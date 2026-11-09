//! Leaves: a page of entries in key order, and a window per entry.
//!
//! Layout: the header, the window column, then the entries, each key beside its value, so a
//! search that ends at a key has loaded its value too. Entries whose size is a power of two
//! up to a cache line start at a multiple of it and never straddle two lines.
//!
//! Leaves keep no fences. Whoever changes a leaf's range holds its parent, which knows the
//! fences, and sets the leaf's prefix from them (`prefix_set`).

const std = @import("std");
const assert = std.debug.assert;
const latch = @import("latch.zig");
const page_header = @import("page.zig");
const windows_column = @import("windows.zig");

const Concurrency = latch.Concurrency;
const Header = page_header.Header;
const Mode = page_header.Mode;
const Restart = latch.Restart;
const Version = latch.Version;

pub fn Leaf(
    comptime Codec: type,
    comptime Value: type,
    comptime page_size_min: u32,
    comptime concurrency: Concurrency,
) type {
    return struct {
        pub const Key = Codec.Key;
        pub const Entry = struct { key: Key, value: Value };
        pub const Window = @Int(.unsigned, Codec.window_size_leaf * 8);

        /// The configured page size, doubled until a leaf holds `capacity_floor` entries.
        pub const page_size: u32 = size: {
            var size = page_size_min;
            while (capacity_for(size) < capacity_floor) size *= 2;
            break :size size;
        };
        pub const capacity: u32 = capacity_for(page_size);
        /// Every leaf but the root holds at least this many entries.
        pub const count_min: u32 = capacity / 4;

        pub const Page = [page_size]u8;
        pub const PagePointer = *align(page_alignment) Page;
        /// A cache line, or the entries' alignment where that is more.
        pub const page_alignment: u32 = @max(page_header.alignment, entry_alignment);

        const capacity_floor = 16;
        const slot_count = windows_column.slot_count(Window, capacity);
        const windows_offset: u32 = @sizeOf(Header);
        const entries_offset: u32 = entries_offset_for(capacity);
        const entry_alignment: u32 = alignment: {
            const size = @sizeOf(Entry);
            if (size > 0 and std.math.isPowerOfTwo(size) and size <= 64) {
                break :alignment @max(size, @alignOf(Entry));
            }
            break :alignment @alignOf(Entry);
        };

        const Self = @This();
        const Latch = latch.Latch(concurrency);
        const access = page_header.Access(concurrency);

        comptime {
            // Window counts and the header's u16 counts stay far from overflow.
            assert(page_size <= 64 * 1024);
            assert(capacity >= capacity_floor);
            assert(capacity < std.math.maxInt(u16));
            assert(count_min >= 4);
            assert(page_size % page_alignment == 0);
            assert(windows_offset % windows_column.slot_alignment_bytes == 0);
            assert(entries_offset + capacity * @sizeOf(Entry) <= page_size);
            assert(entries_offset % entry_alignment == 0);
        }

        fn entries_offset_for(entry_count: u32) u32 {
            const windows_end = windows_offset +
                windows_column.slot_count(Window, entry_count) * @sizeOf(Window);
            return std.mem.alignForward(u32, windows_end, entry_alignment);
        }

        fn capacity_for(size: u32) u32 {
            var entry_count: u32 = size / @max(1, @sizeOf(Entry) + @sizeOf(Window));
            while (entry_count > 0) : (entry_count -= 1) {
                if (entries_offset_for(entry_count) + entry_count * @sizeOf(Entry) <= size) break;
            }
            return @min(entry_count, std.math.maxInt(u16) - 1);
        }

        pub fn header(page: PagePointer) *Header {
            return @ptrCast(page);
        }

        pub fn windows(page: PagePointer) *[slot_count]Window {
            return @ptrCast(@alignCast(page[windows_offset..][0 .. slot_count * @sizeOf(Window)]));
        }

        pub fn entries(page: PagePointer) *[capacity]Entry {
            return @ptrCast(@alignCast(page[entries_offset..][0 .. capacity * @sizeOf(Entry)]));
        }

        /// Makes the page an empty leaf, not the last. Leaves its latch word alone.
        pub fn init(page: PagePointer, prefix_length: u32) void {
            access.store(u16, &header(page).count, 0);
            rightmost_set(page, false);
            access.store(u16, &header(page).prefix_length, prefix_clamp(prefix_length));
            windows_column.pad(Window, windows(page), 0);
        }

        pub fn count(page: PagePointer) u32 {
            const result = header(page).count;
            assert(result <= capacity);
            return result;
        }

        /// The count as a racing reader may see it: possibly torn, so clamped into the page.
        pub fn count_racy(page: PagePointer) u32 {
            return @min(access.load(u16, &header(page).count), capacity);
        }

        pub fn key_at(page: PagePointer, index: u32) Key {
            assert(index < count(page));
            return entries(page)[index].key;
        }

        pub fn key_racy(page: PagePointer, index: u32) Key {
            assert(index < capacity);
            return access.racy(Key, &entries(page)[index].key);
        }

        /// Whether the leaf is the tree's last: it covers every key from its first on.
        pub fn rightmost(page: PagePointer) bool {
            return header(page).rightmost == 1;
        }

        pub fn rightmost_racy(page: PagePointer) bool {
            return access.load(u8, &header(page).rightmost) == 1;
        }

        pub fn rightmost_set(page: PagePointer, value: bool) void {
            access.store(u8, &header(page).rightmost, @intFromBool(value));
        }

        pub fn value_pointer(page: PagePointer, index: u32) *Value {
            assert(index < count(page));
            return &entries(page)[index].value;
        }

        fn prefix_clamp(prefix_length: u32) u16 {
            return @intCast(@min(prefix_length, std.math.maxInt(u16)));
        }

        fn window_of(prefix_length: u32, key: Key) Window {
            return @truncate(Codec.window(key, prefix_length) >> (64 - @bitSizeOf(Window)));
        }

        pub const Position = struct {
            index: u32,
            found: bool,
        };

        /// Where `key` is, or would go. In optimistic mode the page may be changing: the
        /// position stays within the page, and the caller validates `version` before using it.
        pub fn search(
            comptime mode: Mode,
            page: PagePointer,
            version: Version,
            key: Key,
        ) Restart!Position {
            const entry_count = if (mode == .exclusive) count(page) else count_racy(page);
            const prefix_length = access.load(u16, &header(page).prefix_length);
            const result = windows_column.counts(
                Window,
                slot_count,
                windows(page),
                window_of(prefix_length, key),
            );
            if (mode == .exclusive) assert(result.less <= entry_count);

            const start = @min(result.less, entry_count);
            const end = @min(result.less + result.equal, entry_count);
            if (start == end) return .{ .index = start, .found = false };
            return search_ties(mode, page, version, key, prefix_length, start, end);
        }

        /// Binary search over the entries whose windows tie with the key's: every entry before
        /// `start` is smaller than the key, and every entry from `end` on is larger.
        fn search_ties(
            comptime mode: Mode,
            page: PagePointer,
            version: Version,
            key: Key,
            prefix_length: u32,
            start: u32,
            end: u32,
        ) Restart!Position {
            assert(start < end);
            assert(end <= capacity);
            var low = start;
            var high = end;
            while (low < high) {
                const middle = low + (high - low) / 2;
                const middle_key = try key_read(mode, page, version, middle);
                switch (Codec.order_from(middle_key, key, prefix_length)) {
                    .lt => low = middle + 1,
                    .gt => high = middle,
                    .eq => return .{ .index = middle, .found = true },
                }
            }
            return .{ .index = low, .found = false };
        }

        /// A key to compare with. A racing reader validates the page before it may follow an
        /// indirect key's pointer.
        fn key_read(
            comptime mode: Mode,
            page: PagePointer,
            version: Version,
            index: u32,
        ) Restart!Key {
            if (mode == .exclusive) return key_at(page, index);
            const key = access.racy(Key, &entries(page)[index].key);
            if (Codec.indirect) try Latch.validate(&header(page).latch, version);
            return key;
        }

        /// The value of `key`, read without latching the leaf, as a sequence lock reads.
        pub fn get_racy(page: PagePointer, version: Version, key: Key) Restart!?Value {
            const position = try search(.optimistic, page, version, key);
            return get_racy_at(page, version, position);
        }

        /// The first half of `search`: the entries whose windows tie with the key's, which a
        /// batched lookup prefetches before it compares them.
        pub fn candidates_racy(page: PagePointer, key: Key) Candidates {
            const entry_count = count_racy(page);
            const prefix_length = access.load(u16, &header(page).prefix_length);
            const result = windows_column.counts(
                Window,
                slot_count,
                windows(page),
                window_of(prefix_length, key),
            );
            return .{
                .start = @min(result.less, entry_count),
                .end = @min(result.less + result.equal, entry_count),
                .prefix_length = prefix_length,
            };
        }

        pub const Candidates = struct {
            start: u32,
            end: u32,
            prefix_length: u16,
        };

        /// The second half of `search` and `get_racy`: compares the candidates and copies the
        /// value of a match.
        pub fn get_racy_candidates(
            page: PagePointer,
            version: Version,
            key: Key,
            candidates: Candidates,
        ) Restart!?Value {
            assert(candidates.start <= candidates.end);
            assert(candidates.end <= capacity);
            const position: Position = if (candidates.start == candidates.end)
                .{ .index = candidates.start, .found = false }
            else
                try search_ties(
                    .optimistic,
                    page,
                    version,
                    key,
                    candidates.prefix_length,
                    candidates.start,
                    candidates.end,
                );
            return get_racy_at(page, version, position);
        }

        /// Where the entry at `index` lies, for a batched lookup's prefetch.
        pub fn entry_address(page: PagePointer, index: u32) [*]const u8 {
            assert(index < capacity);
            return @ptrCast(&entries(page)[index]);
        }

        /// Bytes from the page's start through its window column: what a search reads first.
        pub const search_prefix_bytes = windows_offset + slot_count * @sizeOf(Window);

        fn get_racy_at(page: PagePointer, version: Version, position: Position) Restart!?Value {
            if (!position.found) {
                try Latch.validate(&header(page).latch, version);
                return null;
            }
            const value = access.racy(Value, &entries(page)[position.index].value);
            try Latch.validate(&header(page).latch, version);
            return value;
        }

        pub const Copied = struct {
            count: u32,
            /// Whether the copy reached the leaf's last entry.
            finished: bool,
        };

        /// Copies entries from `start` on into `target`, as many as fit; the caller validates the
        /// page after.
        pub fn copy_racy(page: PagePointer, start: u32, target: []Entry) Copied {
            const entry_count = count_racy(page);
            if (entry_count <= start) return .{ .count = 0, .finished = true };
            const copied: u32 = @intCast(@min(entry_count - start, target.len));
            access.copy(Entry, target[0..copied], entries(page)[start..][0..copied]);
            return .{ .count = copied, .finished = start + copied == entry_count };
        }

        pub fn insert_at(page: PagePointer, index: u32, key: Key, value: Value) void {
            const entry_count = count(page);
            assert(entry_count < capacity);
            assert(index <= entry_count);
            if (index > 0) assert(Codec.order(entries(page)[index - 1].key, key) == .lt);
            if (index < entry_count) assert(Codec.order(key, entries(page)[index].key) == .lt);

            const slots = entries(page);
            // Appends, as ascending keys make, move nothing: skip the call.
            if (index < entry_count) {
                @memmove(slots[index + 1 .. entry_count + 1], slots[index..entry_count]);
            }
            slots[index] = .{ .key = key, .value = value };
            const prefix_length = header(page).prefix_length;
            windows_column.insert(
                Window,
                windows(page),
                entry_count,
                index,
                window_of(prefix_length, key),
            );
            access.store(u16, &header(page).count, @intCast(entry_count + 1));
        }

        pub fn remove_at(page: PagePointer, index: u32) Entry {
            const entry_count = count(page);
            assert(index < entry_count);
            const slots = entries(page);
            const removed = slots[index];
            @memmove(
                slots[index .. entry_count - 1],
                slots[index + 1 .. entry_count],
            );
            windows_column.remove(Window, windows(page), entry_count, index);
            access.store(u16, &header(page).count, @intCast(entry_count - 1));
            return removed;
        }

        /// Moves the entries from `split_index` on, with their windows, into `right`, an empty
        /// leaf, which takes this leaf's prefix: both halves' keys share it still. Their fences
        /// narrowed, so `prefix_fit` may lengthen either prefix after.
        pub fn move_tail(left: PagePointer, right: PagePointer, split_index: u32) void {
            const left_count = count(left);
            assert(count(right) == 0);
            assert(split_index <= left_count);
            const moved = left_count - split_index;
            @memcpy(entries(right)[0..moved], entries(left)[split_index..left_count]);
            @memcpy(windows(right)[0..moved], windows(left)[split_index..left_count]);
            windows_column.pad(Window, windows(left), split_index);
            access.store(u16, &header(right).prefix_length, header(left).prefix_length);
            access.store(u16, &header(left).count, @intCast(split_index));
            access.store(u16, &header(right).count, @intCast(moved));
        }

        /// Moves every entry of `right` to the end of `left`, its left neighbor. Like every move
        /// between leaves, leaves the windows to `prefix_set`.
        pub fn move_all(left: PagePointer, right: PagePointer) void {
            const left_count = count(left);
            const right_count = count(right);
            assert(left_count + right_count <= capacity);
            @memcpy(entries(left)[left_count..][0..right_count], entries(right)[0..right_count]);
            access.store(u16, &header(left).count, @intCast(left_count + right_count));
            access.store(u16, &header(right).count, 0);
        }

        /// Moves the first `moved` entries of `right` to the end of `left`.
        pub fn move_from_right(left: PagePointer, right: PagePointer, moved: u32) void {
            const left_count = count(left);
            const right_count = count(right);
            assert(moved > 0);
            assert(moved <= right_count);
            assert(left_count + moved <= capacity);
            const right_entries = entries(right);
            @memcpy(entries(left)[left_count..][0..moved], right_entries[0..moved]);
            @memmove(
                right_entries[0 .. right_count - moved],
                right_entries[moved..right_count],
            );
            access.store(u16, &header(left).count, @intCast(left_count + moved));
            access.store(u16, &header(right).count, @intCast(right_count - moved));
        }

        /// Moves the last `moved` entries of `left` to the start of `right`.
        pub fn move_from_left(left: PagePointer, right: PagePointer, moved: u32) void {
            const left_count = count(left);
            const right_count = count(right);
            assert(moved > 0);
            assert(moved <= left_count);
            assert(right_count + moved <= capacity);
            const right_entries = entries(right);
            @memmove(
                right_entries[moved .. right_count + moved],
                right_entries[0..right_count],
            );
            @memcpy(right_entries[0..moved], entries(left)[left_count - moved .. left_count]);
            access.store(u16, &header(left).count, @intCast(left_count - moved));
            access.store(u16, &header(right).count, @intCast(right_count + moved));
        }

        /// Sets the prefix that the leaf's windows skip, and recomputes every window.
        pub fn prefix_set(page: PagePointer, prefix_length: u32) void {
            const prefix = prefix_clamp(prefix_length);
            const entry_count = count(page);
            access.store(u16, &header(page).prefix_length, prefix);
            const column = windows(page);
            for (entries(page)[0..entry_count], column[0..entry_count]) |*entry, *window| {
                window.* = window_of(prefix, entry.key);
            }
            windows_column.pad(Window, column, entry_count);
            assert(std.sort.isSorted(Window, column, {}, std.sort.asc(Window)));
        }

        /// How far a leaf's prefix may trail the one its fences give. Refitting windows reads
        /// every key, which for codecs whose keys live outside the node is a memory access a
        /// key; their leaves keep their windows through splits until the prefix would grow by
        /// a quarter of a window. A shorter prefix stays true, and costs windows only a few of
        /// their many bits.
        pub const prefix_lag_max: u32 = if (Codec.indirect) @bitSizeOf(Window) / 4 else 0;

        /// Lengthens the prefix to `prefix_length` after a split narrowed the leaf's fences,
        /// unless it trails by no more than `prefix_lag_max`.
        pub fn prefix_fit(page: PagePointer, prefix_length: u32) void {
            const prefix_current: u32 = header(page).prefix_length;
            const prefix_target: u32 = prefix_clamp(prefix_length);
            assert(prefix_current <= prefix_target);
            if (prefix_target - prefix_current <= prefix_lag_max) return;
            prefix_set(page, prefix_target);
        }

        /// Checks a quiescent leaf against the fences its parent gives it and the prefix they
        /// set, which the leaf's may trail by `prefix_lag_max`.
        pub fn verify(
            page: PagePointer,
            low: ?Key,
            high: ?Key,
            prefix_expected: u32,
            is_root: bool,
        ) void {
            const entry_count = count(page);
            if (!is_root) assert(entry_count >= count_min);
            const prefix_length = header(page).prefix_length;
            assert(prefix_length <= prefix_clamp(prefix_expected));
            assert(prefix_clamp(prefix_expected) - prefix_length <= prefix_lag_max);
            const column = windows(page);
            for (entries(page)[0..entry_count], 0..) |entry, index| {
                if (index > 0) assert(Codec.order(entries(page)[index - 1].key, entry.key) == .lt);
                if (low) |fence| assert(Codec.order(fence, entry.key) != .gt);
                if (high) |fence| assert(Codec.order(entry.key, fence) == .lt);
                assert(column[index] == window_of(prefix_length, entry.key));
            }
            for (column[entry_count..]) |window| assert(window == std.math.maxInt(Window));
        }
    };
}

const testing = std.testing;
const codec = @import("codec.zig");

test "leaf layouts fit their pages" {
    const U64 = Leaf(codec.Auto(u64), u64, 1024, .multi_threaded);
    try testing.expectEqual(@as(u32, 1024), U64.page_size);
    try testing.expectEqual(@as(u32, 56), U64.capacity);
    try testing.expectEqual(@as(u32, 128), U64.entries_offset);

    const Slice = Leaf(codec.ByteSlice, u64, 1024, .multi_threaded);
    try testing.expect(Slice.capacity >= 30);

    const Large = Leaf(codec.Auto(u64), [200]u8, 1024, .multi_threaded);
    try testing.expect(Large.page_size > 1024);
    try testing.expect(Large.capacity >= 16);
}

test "leaf search, insert, remove and moves" {
    const L = Leaf(codec.Auto(u32), u32, 1024, .single_threaded);
    var pages: [2]L.Page align(L.page_size) = undefined;
    const left: L.PagePointer = &pages[0];
    const right: L.PagePointer = &pages[1];
    L.header(left).* = .empty;
    L.header(right).* = L.header(left).*;
    L.init(left, 0);
    L.init(right, 0);

    var key: u32 = 0;
    while (L.count(left) < L.capacity) : (key += 7) {
        const position = try L.search(.exclusive, left, 0, key);
        try testing.expect(!position.found);
        L.insert_at(left, position.index, key, key * 2);
    }
    L.verify(left, null, null, 0, true);
    const found = try L.search(.exclusive, left, 0, 70);
    try testing.expect(found.found);
    try testing.expectEqual(@as(u32, 140), L.value_pointer(left, found.index).*);

    L.move_tail(left, right, L.capacity / 2);
    const separator = L.key_at(right, 0);
    const left_prefix = codec.Auto(u32).prefix_length(null, separator);
    const right_prefix = codec.Auto(u32).prefix_length(separator, null);
    L.prefix_set(left, left_prefix);
    L.prefix_set(right, right_prefix);
    L.verify(left, null, separator, left_prefix, false);
    L.verify(right, separator, null, right_prefix, false);

    L.move_from_right(left, right, 3);
    L.move_from_left(left, right, 5);
    const moved_separator = L.key_at(right, 0);
    L.prefix_set(left, 0);
    L.prefix_set(right, 0);
    L.verify(left, null, moved_separator, 0, false);
    L.verify(right, moved_separator, null, 0, false);

    const removed = L.remove_at(right, 0);
    try testing.expectEqual(moved_separator, removed.key);
    L.move_all(left, right);
    L.prefix_set(left, 0);
    L.verify(left, null, null, 0, true);
    try testing.expectEqual(L.capacity - 1, L.count(left));
}
