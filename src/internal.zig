//! Internal nodes: separator keys, a window per separator, and the edges between them.
//!
//! Layout: the header, the window column, the separators, the edges, and last the node's
//! fences. A search reads the header line, the windows, and one edge: separators only on a
//! tie, and fences only when the node's children change shape. Edges are 32-bit page
//! indexes. Edge `i` leads to the keys from separator `i - 1` (inclusive) up to separator
//! `i` (exclusive).
//!
//! Separators are copies of keys. A key's removal leaves them in place: they still separate.

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

pub fn Internal(
    comptime Codec: type,
    comptime page_size_min: u32,
    comptime concurrency: Concurrency,
) type {
    return struct {
        pub const Key = Codec.Key;
        pub const Window = @Int(.unsigned, Codec.window_size_internal * 8);

        /// The separators on either side of the node in its ancestors. A missing one is the
        /// tree's border.
        pub const Fences = struct {
            low: ?Key,
            high: ?Key,
        };

        /// The configured page size, doubled until a node holds `capacity_floor` separators.
        pub const page_size: u32 = size: {
            var size = page_size_min;
            while (capacity_for(size) < capacity_floor) size *= 2;
            break :size size;
        };
        pub const capacity: u32 = capacity_for(page_size);
        /// Every internal node but the root holds at least this many separators.
        pub const count_min: u32 = capacity / 4;

        pub const Page = [page_size]u8;
        pub const PagePointer = *align(page_alignment) Page;
        /// A cache line, or the keys' alignment where that is more.
        pub const page_alignment: u32 =
            @max(page_header.alignment, @alignOf(Key), @alignOf(Fences));

        /// Bytes from the page's start through its window column: what a search reads first.
        pub const search_prefix_bytes = @sizeOf(Header) + slot_count * @sizeOf(Window);

        const capacity_floor = 16;
        /// A search reads the header and the whole window column first, so the column holds
        /// the separators whose windows fit in 128 bytes with the header: one cache line on
        /// CPUs with 128-byte lines, two adjacent ones elsewhere. Codecs whose keys live
        /// outside the node take 256, as a tie reads a separator's bytes from memory, which
        /// costs more than a second line of wider windows. The floor of 16 separators comes
        /// first. A wider column puts one more line on every level's critical path; the
        /// page's unused tail costs nothing.
        const search_bytes_max = if (Codec.indirect) 256 else 128;
        const capacity_ceiling = @max(
            capacity_floor,
            (search_bytes_max - @sizeOf(Header)) / @sizeOf(Window),
        );
        const slot_count = windows_column.slot_count(Window, capacity);
        const layout = layout_for(capacity);

        const Latch = latch.Latch(concurrency);
        const access = page_header.Access(concurrency);

        const Layout = struct {
            keys_offset: u32,
            edges_offset: u32,
            fences_offset: u32,
            size: u32,
        };

        comptime {
            // Window counts and the header's u16 counts stay far from overflow.
            assert(page_size <= 64 * 1024);
            assert(capacity >= capacity_floor);
            assert(capacity < std.math.maxInt(u16));
            assert(count_min >= 4);
            assert(layout.size <= page_size);
            assert(page_size % page_alignment == 0);
            assert(layout.keys_offset % @alignOf(Key) == 0);
            assert(layout.edges_offset % @alignOf(u32) == 0);
            assert(layout.fences_offset % @alignOf(Fences) == 0);
        }

        fn layout_for(key_count: u32) Layout {
            const windows_bytes = windows_column.slot_count(Window, key_count) * @sizeOf(Window);
            const keys_offset = std.mem.alignForward(
                u32,
                @sizeOf(Header) + windows_bytes,
                @alignOf(Key),
            );
            const edges_offset = std.mem.alignForward(
                u32,
                keys_offset + key_count * @sizeOf(Key),
                @alignOf(u32),
            );
            const fences_offset = std.mem.alignForward(
                u32,
                edges_offset + (key_count + 1) * @sizeOf(u32),
                @alignOf(Fences),
            );
            return .{
                .keys_offset = keys_offset,
                .edges_offset = edges_offset,
                .fences_offset = fences_offset,
                .size = fences_offset + @sizeOf(Fences),
            };
        }

        fn capacity_for(size: u32) u32 {
            var key_count: u32 = size / (@sizeOf(Key) + @sizeOf(Window) + @sizeOf(u32));
            while (key_count > 0) : (key_count -= 1) {
                if (layout_for(key_count).size <= size) break;
            }
            return @min(key_count, capacity_ceiling);
        }

        pub fn header(page: PagePointer) *Header {
            return @ptrCast(page);
        }

        fn windows(page: PagePointer) *[slot_count]Window {
            const bytes = page[@sizeOf(Header)..][0 .. slot_count * @sizeOf(Window)];
            return @ptrCast(@alignCast(bytes));
        }

        fn keys(page: PagePointer) *[capacity]Key {
            return @ptrCast(@alignCast(page[layout.keys_offset..][0 .. capacity * @sizeOf(Key)]));
        }

        fn edges(page: PagePointer) *[capacity + 1]u32 {
            const bytes = page[layout.edges_offset..][0 .. (capacity + 1) * @sizeOf(u32)];
            return @ptrCast(@alignCast(bytes));
        }

        pub fn fences(page: PagePointer) *Fences {
            return @ptrCast(@alignCast(page[layout.fences_offset..][0..@sizeOf(Fences)]));
        }

        /// Makes the page an internal node with one edge and no separators.
        pub fn init(
            page: PagePointer,
            node_fences: Fences,
            edge_first: u32,
            shared_length: u32,
        ) void {
            access.store(u16, &header(page).count, 0);
            fences(page).* = node_fences;
            edges(page)[0] = edge_first;
            prefix_set(page, shared_length);
        }

        pub fn count(page: PagePointer) u32 {
            const result = header(page).count;
            assert(result <= capacity);
            return result;
        }

        /// The count as a racing reader may see it: possibly stale, so clamped into the page.
        pub fn count_racy(page: PagePointer) u32 {
            return @min(access.load(u16, &header(page).count), capacity);
        }

        pub fn key_at(page: PagePointer, index: u32) Key {
            assert(index < count(page));
            return keys(page)[index];
        }

        pub fn key_racy(page: PagePointer, index: u32) Key {
            assert(index < capacity);
            return access.racy(Key, &keys(page)[index]);
        }

        pub fn edge_at(page: PagePointer, index: u32) u32 {
            assert(index <= count(page));
            return edges(page)[index];
        }

        pub fn edge_racy(page: PagePointer, index: u32) u32 {
            assert(index <= capacity);
            return access.racy(u32, &edges(page)[index]);
        }

        /// Where edge `index` lies, for a batched lookup's prefetch.
        pub fn edge_address(page: PagePointer, index: u32) [*]const u8 {
            assert(index <= capacity);
            return @ptrCast(&edges(page)[index]);
        }

        /// The fences of the child behind edge `index`.
        pub fn child_fences(page: PagePointer, index: u32) Fences {
            const separator_count = count(page);
            assert(index <= separator_count);
            const node_fences = fences(page).*;
            return .{
                .low = if (index == 0) node_fences.low else keys(page)[index - 1],
                .high = if (index == separator_count) node_fences.high else keys(page)[index],
            };
        }

        fn window_of(prefix_length: u32, key: Key) Window {
            return @truncate(Codec.window(key, prefix_length) >> (64 - @bitSizeOf(Window)));
        }

        /// The edge to follow for `key`: how many separators are at most `key`. In optimistic
        /// mode the result stays within the node, and the caller validates `version` before
        /// following it.
        pub fn route(
            comptime mode: Mode,
            page: PagePointer,
            version: Version,
            key: Key,
        ) Restart!u32 {
            const separator_count = if (mode == .exclusive) count(page) else count_racy(page);
            const prefix_length = access.load(u16, &header(page).prefix_length);
            const query = window_of(prefix_length, key);
            const result = windows_column.counts(Window, slot_count, windows(page), query);
            if (mode == .exclusive) assert(result.less <= separator_count);

            const start = @min(result.less, separator_count);
            const end = @min(result.less + result.equal, separator_count);
            if (start == end) return start;
            return route_ties(mode, page, version, key, prefix_length, start, end);
        }

        /// Binary search for the first separator above `key` among those whose windows tie.
        fn route_ties(
            comptime mode: Mode,
            page: PagePointer,
            version: Version,
            key: Key,
            prefix_length: u32,
            start: u32,
            end: u32,
        ) Restart!u32 {
            assert(start < end);
            assert(end <= capacity);
            var low = start;
            var high = end;
            while (low < high) {
                const middle = low + (high - low) / 2;
                const separator = if (mode == .exclusive)
                    key_at(page, middle)
                else
                    key_racy(page, middle);
                // A racing reader validates before an indirect key's pointer may be followed.
                if (mode == .optimistic and Codec.indirect) {
                    try Latch.validate(&header(page).latch, version);
                }
                if (Codec.order_from(separator, key, prefix_length) == .gt) {
                    high = middle;
                } else {
                    low = middle + 1;
                }
            }
            return low;
        }

        /// Adds `separator` at `index`, with `edge` to its right: the right half of the child
        /// at `index`, which kept the keys below the separator.
        pub fn insert_separator(page: PagePointer, index: u32, separator: Key, edge: u32) void {
            const separator_count = count(page);
            assert(separator_count < capacity);
            assert(index <= separator_count);
            if (index > 0) assert(Codec.order(keys(page)[index - 1], separator) == .lt);
            if (index < separator_count) assert(Codec.order(separator, keys(page)[index]) == .lt);

            const key_slots = keys(page);
            const edge_slots = edges(page);
            @memmove(
                key_slots[index + 1 .. separator_count + 1],
                key_slots[index..separator_count],
            );
            @memmove(
                edge_slots[index + 2 .. separator_count + 2],
                edge_slots[index + 1 .. separator_count + 1],
            );
            key_slots[index] = separator;
            edge_slots[index + 1] = edge;
            const window = window_of(header(page).prefix_length, separator);
            windows_column.insert(Window, windows(page), separator_count, index, window);
            access.store(u16, &header(page).count, @intCast(separator_count + 1));
        }

        /// Removes the separator at `index` and the edge to its right.
        pub fn remove_separator(page: PagePointer, index: u32) void {
            const separator_count = count(page);
            assert(index < separator_count);
            const key_slots = keys(page);
            const edge_slots = edges(page);
            @memmove(
                key_slots[index .. separator_count - 1],
                key_slots[index + 1 .. separator_count],
            );
            @memmove(
                edge_slots[index + 1 .. separator_count],
                edge_slots[index + 2 .. separator_count + 1],
            );
            windows_column.remove(Window, windows(page), separator_count, index);
            access.store(u16, &header(page).count, @intCast(separator_count - 1));
        }

        /// Replaces the separator at `index` with one that sorts between the same neighbors.
        pub fn separator_set(page: PagePointer, index: u32, separator: Key) void {
            const separator_count = count(page);
            assert(index < separator_count);
            if (index > 0) assert(Codec.order(keys(page)[index - 1], separator) == .lt);
            if (index + 1 < separator_count) {
                assert(Codec.order(separator, keys(page)[index + 1]) == .lt);
            }
            keys(page)[index] = separator;
            windows(page)[index] = window_of(header(page).prefix_length, separator);
        }

        /// Moves the upper half of a full node into `right`, a fresh page whose contents it
        /// overwrites, and returns the middle separator, which neither keeps. Sets both nodes'
        /// fences.
        pub fn split(left: PagePointer, right: PagePointer, shared_length: u32) Key {
            const separator_count = count(left);
            assert(separator_count == capacity);
            const middle = separator_count / 2;
            const separator = keys(left)[middle];
            const moved = separator_count - middle - 1;
            @memcpy(keys(right)[0..moved], keys(left)[middle + 1 .. separator_count]);
            @memcpy(edges(right)[0 .. moved + 1], edges(left)[middle + 1 .. separator_count + 1]);
            access.store(u16, &header(left).count, @intCast(middle));
            access.store(u16, &header(right).count, @intCast(moved));

            fences(right).* = .{ .low = separator, .high = fences(left).high };
            fences(left).high = separator;
            prefix_set(left, shared_length);
            prefix_set(right, shared_length);
            return separator;
        }

        /// Moves `separator` and everything in `right` to the end of `left`.
        pub fn merge(
            left: PagePointer,
            separator: Key,
            right: PagePointer,
            shared_length: u32,
        ) void {
            const left_count = count(left);
            const right_count = count(right);
            assert(left_count + 1 + right_count <= capacity);
            keys(left)[left_count] = separator;
            @memcpy(keys(left)[left_count + 1 ..][0..right_count], keys(right)[0..right_count]);
            @memcpy(
                edges(left)[left_count + 1 ..][0 .. right_count + 1],
                edges(right)[0 .. right_count + 1],
            );
            access.store(u16, &header(left).count, @intCast(left_count + 1 + right_count));
            access.store(u16, &header(right).count, 0);

            fences(left).high = fences(right).high;
            prefix_set(left, shared_length);
        }

        /// Moves `moved` edges from the start of `right` to the end of `left`, rotating
        /// through the separator between them. Returns the new separator.
        pub fn move_from_right(
            left: PagePointer,
            separator: Key,
            right: PagePointer,
            moved: u32,
            shared_length: u32,
        ) Key {
            const left_count = count(left);
            const right_count = count(right);
            assert(moved > 0);
            assert(moved <= right_count);
            assert(left_count + moved <= capacity);
            const separator_new = keys(right)[moved - 1];
            keys(left)[left_count] = separator;
            @memcpy(keys(left)[left_count + 1 ..][0 .. moved - 1], keys(right)[0 .. moved - 1]);
            @memcpy(edges(left)[left_count + 1 ..][0..moved], edges(right)[0..moved]);
            @memmove(
                keys(right)[0 .. right_count - moved],
                keys(right)[moved..right_count],
            );
            @memmove(
                edges(right)[0 .. right_count - moved + 1],
                edges(right)[moved .. right_count + 1],
            );
            access.store(u16, &header(left).count, @intCast(left_count + moved));
            access.store(u16, &header(right).count, @intCast(right_count - moved));
            fences_rotate(left, right, separator_new, shared_length);
            return separator_new;
        }

        /// Moves `moved` edges from the end of `left` to the start of `right`. Returns the
        /// new separator.
        pub fn move_from_left(
            left: PagePointer,
            separator: Key,
            right: PagePointer,
            moved: u32,
            shared_length: u32,
        ) Key {
            const left_count = count(left);
            const right_count = count(right);
            assert(moved > 0);
            assert(moved <= left_count);
            assert(right_count + moved <= capacity);
            const separator_new = keys(left)[left_count - moved];
            @memmove(
                keys(right)[moved .. right_count + moved],
                keys(right)[0..right_count],
            );
            @memmove(
                edges(right)[moved .. right_count + moved + 1],
                edges(right)[0 .. right_count + 1],
            );
            @memcpy(keys(right)[0 .. moved - 1], keys(left)[left_count - moved + 1 .. left_count]);
            keys(right)[moved - 1] = separator;
            @memcpy(edges(right)[0..moved], edges(left)[left_count - moved + 1 .. left_count + 1]);
            access.store(u16, &header(left).count, @intCast(left_count - moved));
            access.store(u16, &header(right).count, @intCast(right_count + moved));
            fences_rotate(left, right, separator_new, shared_length);
            return separator_new;
        }

        fn fences_rotate(
            left: PagePointer,
            right: PagePointer,
            separator: Key,
            shared_length: u32,
        ) void {
            fences(left).high = separator;
            fences(right).low = separator;
            prefix_set(left, shared_length);
            prefix_set(right, shared_length);
        }

        /// The prefix of a node between `node_fences`, in a tree whose keys all share their
        /// first `shared_length` bits: the prefix that a missing fence leaves to a node on the
        /// tree's border. Prefixes past what the header holds shorten, which keeps them true.
        pub fn prefix_of(node_fences: Fences, shared_length: u32) u16 {
            const fenced = Codec.prefix_length(node_fences.low, node_fences.high);
            return @intCast(@min(@max(fenced, shared_length), std.math.maxInt(u16)));
        }

        /// Sets the prefix from the node's fences, and recomputes every window.
        pub fn prefix_set(page: PagePointer, shared_length: u32) void {
            const prefix_length = prefix_of(fences(page).*, shared_length);
            const separator_count = count(page);
            access.store(u16, &header(page).prefix_length, prefix_length);
            const column = windows(page);
            for (keys(page)[0..separator_count], column[0..separator_count]) |key, *window| {
                window.* = window_of(prefix_length, key);
            }
            windows_column.pad(Window, column, separator_count);
            assert(std.sort.isSorted(Window, column, {}, std.sort.asc(Window)));
        }

        /// Checks a quiescent node against the fences its parent gives it.
        pub fn verify(
            page: PagePointer,
            node_fences: Fences,
            shared_length: u32,
            is_root: bool,
        ) void {
            const separator_count = count(page);
            if (!is_root) assert(separator_count >= count_min);
            const stored = fences(page).*;
            assert(optional_equal(stored.low, node_fences.low));
            assert(optional_equal(stored.high, node_fences.high));
            const prefix_length = header(page).prefix_length;
            assert(prefix_length == prefix_of(stored, shared_length));
            const column = windows(page);
            for (keys(page)[0..separator_count], 0..) |key, index| {
                if (index > 0) assert(Codec.order(keys(page)[index - 1], key) == .lt);
                if (stored.low) |fence| assert(Codec.order(fence, key) == .lt);
                if (stored.high) |fence| assert(Codec.order(key, fence) == .lt);
                assert(column[index] == window_of(prefix_length, key));
            }
            for (column[separator_count..]) |window| assert(window == std.math.maxInt(Window));
        }

        fn optional_equal(a: ?Key, b: ?Key) bool {
            if (a) |a_key| {
                const b_key = b orelse return false;
                return Codec.order(a_key, b_key) == .eq;
            }
            return b == null;
        }
    };
}

const testing = std.testing;
const codec = @import("codec.zig");

test "internal layouts fit their pages" {
    const U64 = Internal(codec.Auto(u64), 1024, .multi_threaded);
    try testing.expectEqual(@as(u32, 1024), U64.page_size);
    try testing.expectEqual(@as(u32, 56), U64.capacity);
    try testing.expectEqual(@as(u32, 128), U64.search_prefix_bytes);
    const Slice = Internal(codec.ByteSlice, 1024, .multi_threaded);
    try testing.expectEqual(@as(u32, 30), Slice.capacity);
    try testing.expectEqual(@as(u32, 256), Slice.search_prefix_bytes);
}

test "internal routing, splits, merges and rotations" {
    const I = Internal(codec.Auto(u64), 1024, .single_threaded);
    var pages: [2]I.Page align(I.page_size) = undefined;
    const left: I.PagePointer = &pages[0];
    const right: I.PagePointer = &pages[1];
    I.header(left).* = .empty;
    I.header(right).* = .empty;
    I.init(left, .{ .low = null, .high = null }, 1000, 0);

    var separator: u64 = 10;
    while (I.count(left) < I.capacity) : (separator += 10) {
        const index = try I.route(.exclusive, left, 0, separator);
        I.insert_separator(left, index, separator, @intCast(1000 + separator));
    }
    I.verify(left, .{ .low = null, .high = null }, 0, true);
    try testing.expectEqual(@as(u32, 0), try I.route(.exclusive, left, 0, 9));
    try testing.expectEqual(@as(u32, 1), try I.route(.exclusive, left, 0, 10));
    try testing.expectEqual(@as(u32, 1), try I.route(.exclusive, left, 0, 19));
    try testing.expectEqual(I.capacity, try I.route(.exclusive, left, 0, std.math.maxInt(u64)));

    const middle = I.split(left, right, 0);
    I.verify(left, .{ .low = null, .high = middle }, 0, false);
    I.verify(right, .{ .low = middle, .high = null }, 0, false);

    const moved_right = I.move_from_left(left, middle, right, 3, 0);
    I.verify(left, .{ .low = null, .high = moved_right }, 0, false);
    I.verify(right, .{ .low = moved_right, .high = null }, 0, false);
    const moved_left = I.move_from_right(left, moved_right, right, 5, 0);
    I.verify(left, .{ .low = null, .high = moved_left }, 0, false);
    I.verify(right, .{ .low = moved_left, .high = null }, 0, false);

    I.remove_separator(left, 0);
    I.remove_separator(right, 0);
    I.merge(left, moved_left, right, 0);
    I.verify(left, .{ .low = null, .high = null }, 0, true);
    try testing.expectEqual(I.capacity - 2, I.count(left));
}
