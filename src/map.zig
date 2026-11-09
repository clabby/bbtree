//! The map: a B+-tree of leaf and internal pages, shared by threads through optimistic lock
//! coupling.
//!
//! Lookups descend without writing anything: at each internal node they read the version,
//! route the key, read the child's version, and check that the node's version has not
//! moved. Leaves are read the same way, as a sequence lock reads, so a hit or a miss writes
//! no shared cache line. Writers descend the same way and latch only the pages they change,
//! by compare-and-swap from the version they read. A writer never waits while it holds a
//! latch: a page that changed or that another writer holds sends it back to the root, which
//! rules out deadlock.
//!
//! Structure changes are preemptive, so each one latches a parent and its children only:
//!
//! - An insert splits every full internal node it passes on the way down, so a leaf that
//!   splits always finds room in its parent.
//! - A removal fixes every child at its minimum before it enters it, by merging it with a
//!   sibling or by moving keys over from one, so the removal itself never underflows a node.
//!
//! Every non-root node stays at least a quarter full, which bounds the pages a map of n
//! entries can need (`options_for`). All pages come from pools sized at initialization.

const std = @import("std");
const assert = std.debug.assert;
const codec = @import("codec.zig");
const latch = @import("latch.zig");
const leaf_node = @import("leaf.zig");
const internal_node = @import("internal.zig");
const page_header = @import("page.zig");
const page_pool = @import("pool.zig");

const Backoff = latch.Backoff;
const Concurrency = latch.Concurrency;
const Version = latch.Version;

pub const Config = struct {
    concurrency: Concurrency = .multi_threaded,
    /// The key codec (`codec.zig`); null picks `codec.Auto(Key)`.
    codec: ?type = null,
    /// Leaf pages hold at least 16 entries; larger entries double the size from here.
    leaf_page_size: u32 = 1024,
    internal_page_size: u32 = 1024,
};

/// Page pool sizes. Every page is allocated at initialization: operations never allocate.
pub const Options = struct {
    leaf_count_max: u32,
    internal_count_max: u32,
};

pub fn Bound(comptime Key: type) type {
    return union(enum) {
        unbounded,
        included: Key,
        excluded: Key,
    };
}

pub fn Map(comptime Key: type, comptime Value: type, comptime config: Config) type {
    return struct {
        /// What every operation reads before the root page, in one cache line.
        head: Head align(std.atomic.cache_line),
        /// The tree's last leaf, where ascending keys go. A hint: the leaf's own flag, read
        /// under its latch, says whether it still is the last. Apart from the root, as appends
        /// change it every few dozen keys.
        rightmost_page: u32 align(std.atomic.cache_line),
        leaves: LeafPool,
        internals: InternalPool,
        /// Entries in the map. Single-threaded maps only: a shared counter would put every
        /// insert and removal on one contended cache line.
        entry_count: EntryCount,

        pub const Codec = config.codec orelse codec.Auto(Key);
        pub const LeafNode = leaf_node.Leaf(Codec, Value, config.leaf_page_size, concurrency);
        pub const InternalNode = internal_node.Internal(
            Codec,
            config.internal_page_size,
            concurrency,
        );
        pub const Entry = LeafNode.Entry;
        pub const KeyBound = Bound(Key);
        pub const Range = struct {
            start: KeyBound = .unbounded,
            end: KeyBound = .unbounded,
        };

        const concurrency = config.concurrency;
        const LeafPool = page_pool.Pool(LeafNode.page_size, LeafNode.page_alignment, concurrency);
        const InternalPool = page_pool.Pool(
            InternalNode.page_size,
            InternalNode.page_alignment,
            concurrency,
        );
        const EntryCount = if (concurrency == .single_threaded) u64 else void;
        const Latch = latch.Latch(concurrency);
        const access = page_header.Access(concurrency);
        const Self = @This();

        /// Leaves lie at most this far below the root: every internal node but the root has at
        /// least five children, and a pool holds at most 2^32 pages.
        pub const height_max = 16;

        const Head = extern struct {
            /// The root page, the tree's height and how many bits every key shares, packed
            /// (`Root`). Every operation reads it. Only the holder of the root page's latch
            /// changes it, or `shared_words`, so the root's version vouches for both.
            root_word: u64,
            leaf_pages: [*]LeafPool.Slot,
            internal_pages: [*]InternalPool.Slot,
            /// The bits that every key in the map shares (`Root.shared_length`), as big-endian
            /// words; bits past them are zero. Every node where a fence is missing, the root
            /// and the nodes along the tree's two borders, skips them as its prefix, and every
            /// operation checks its key against them once, at the root: a key that lacks them
            /// sorts before or after every key in the map.
            shared_words: [shared_word_count]u64,
            /// Whether lookups prefetch the leaf they enter: set for good once the leaves
            /// outgrow the caches (`leaf_prefetch_page_min`).
            leaf_prefetch: bool,
        };

        const Root = packed struct(u64) {
            page: u32,
            height: u8,
            /// Bits that every key shares, or `shared_length_unset` until the first insert.
            shared_length: u16,
            reserved: u8 = 0,
        };

        /// The shared bits the map keeps: at most four words, so that the head fits in the
        /// 64 bytes of the smallest cache line.
        const shared_word_count: u32 = if (Codec.size_max) |size|
            @min(4, std.math.divCeil(u32, size, 8) catch unreachable)
        else
            4;
        const shared_length_max: u16 = if (Codec.size_max) |size|
            @min(shared_word_count * 64, size * 8)
        else
            shared_word_count * 64;
        const shared_length_unset: u16 = std.math.maxInt(u16);

        /// The root page as a descent starts from it, and the shared bits that its version
        /// vouches for.
        const Start = struct {
            node: Cursor,
            shared_length: u16,
        };

        /// A page as a descent saw it: its version, and its height above the leaves.
        const Cursor = struct {
            page: u32,
            version: Version,
            height: u8,
        };

        /// Why an attempt starts over. `Restart`: another thread changed a page the attempt
        /// read. `Restructured`: the attempt changed the tree's shape on its way down.
        const Retry = error{ Restart, Restructured };

        const PutPolicy = enum { replace, keep };

        comptime {
            assert(@sizeOf(Root) == 8);
            assert(shared_length_max < shared_length_unset);
            assert(@sizeOf(Head) <= 64);
            var reach: u128 = 1;
            for (0..height_max - 1) |_| reach = @min(reach * (InternalNode.count_min + 1), 1 << 64);
            assert(reach >= 1 << 32);
        }

        /// Pages per pool that threads changing the tree's shape may hold outside it at once:
        /// a split takes its pages before it links them, and a merge's freed page is out of the
        /// tree a moment before it is back on the free list. A thread holds at most two at a
        /// time, so this covers 1024 threads restructuring at once.
        pub const pages_in_flight_max = 2 * 1024;

        /// Pool sizes that hold `entry_count_max` entries whatever the order of operations:
        /// every leaf but the root keeps at least `LeafNode.count_min` entries, and every
        /// internal node but the root at least `InternalNode.count_min` separators.
        pub fn options_for(entry_count_max: u64) error{CapacityTooLarge}!Options {
            const leaf_count = std.math.divCeil(
                u64,
                entry_count_max,
                LeafNode.count_min,
            ) catch unreachable;
            const internal_count = std.math.divCeil(
                u64,
                leaf_count + 1,
                InternalNode.count_min,
            ) catch unreachable;
            const leaf_count_max = leaf_count + 1 + pages_in_flight_max;
            const internal_count_max = internal_count + height_max + pages_in_flight_max;
            if (leaf_count_max > std.math.maxInt(u32)) return error.CapacityTooLarge;
            if (internal_count_max > std.math.maxInt(u32)) return error.CapacityTooLarge;
            return .{
                .leaf_count_max = @intCast(leaf_count_max),
                .internal_count_max = @intCast(internal_count_max),
            };
        }

        pub fn init(map: *Self, gpa: std.mem.Allocator, options: Options) error{OutOfMemory}!void {
            assert(options.leaf_count_max >= 1);
            assert(options.internal_count_max >= 1);
            try map.leaves.init(gpa, options.leaf_count_max);
            errdefer map.leaves.deinit(gpa);
            try map.internals.init(gpa, options.internal_count_max);
            map.head.leaf_pages = map.leaves.slots.ptr;
            map.head.internal_pages = map.internals.slots.ptr;
            map.head.shared_words = @splat(0);
            map.head.leaf_prefetch = false;

            const root_page = map.leaves.acquire().?;
            LeafNode.init(map.leaf(root_page), 0);
            LeafNode.rightmost_set(map.leaf(root_page), true);
            Latch.unlock(&LeafNode.header(map.leaf(root_page)).latch);
            map.head.root_word = @bitCast(Root{
                .page = root_page,
                .height = 0,
                .shared_length = shared_length_unset,
            });
            map.rightmost_page = root_page;
            map.entry_count = if (concurrency == .single_threaded) 0 else {};
        }

        /// Frees every page. No other thread may use the map any more.
        pub fn deinit(map: *Self, gpa: std.mem.Allocator) void {
            map.internals.deinit(gpa);
            map.leaves.deinit(gpa);
            map.* = undefined;
        }

        /// The value of `key`. Writes nothing shared.
        pub fn get(map: *const Self, key: Key) ?Value {
            var retries: Retries = .{};
            while (true) {
                return map.get_attempt(key) catch |err| {
                    retries.wait(err);
                    continue;
                };
            }
        }

        pub fn contains(map: *const Self, key: Key) bool {
            return map.get(key) != null;
        }

        /// Inserts `key` or replaces its value. Returns the value it replaced.
        pub fn put(map: *Self, key: Key, value: Value) error{MapFull}!?Value {
            return map.put_loop(key, value, .replace);
        }

        /// Inserts `key` unless the map has it. Returns whether it inserted.
        pub fn insert(map: *Self, key: Key, value: Value) error{MapFull}!bool {
            return try map.put_loop(key, value, .keep) == null;
        }

        /// Removes `key`. Returns its value.
        pub fn remove(map: *Self, key: Key) ?Value {
            var retries: Retries = .{};
            while (true) {
                return map.remove_attempt(key) catch |err| {
                    retries.wait(err);
                    continue;
                };
            }
        }

        /// Runs `update_fn` on the value of `key` while its leaf is latched. Returns whether
        /// the map has the key. `update_fn` must not use the map: other threads that read the
        /// leaf wait for it.
        pub fn update(
            map: *Self,
            key: Key,
            context: anytype,
            comptime update_fn: fn (@TypeOf(context), *Value) void,
        ) bool {
            var retries: Retries = .{};
            while (true) {
                return map.update_attempt(key, context, update_fn) catch |err| {
                    retries.wait(err);
                    continue;
                };
            }
        }

        /// A pointer to the value of `key`, valid until the next change to the map.
        pub fn get_ptr(map: *Self, key: Key) ?*Value {
            if (concurrency != .single_threaded) {
                @compileError("get_ptr needs a single_threaded map");
            }
            const node = (map.descend(key) catch unreachable) orelse return null;
            const page = map.leaf(node.page);
            const position = LeafNode.search(.exclusive, page, node.version, key) catch unreachable;
            if (!position.found) return null;
            return LeafNode.value_pointer(page, position.index);
        }

        pub fn count(map: *const Self) u64 {
            if (concurrency != .single_threaded) @compileError("count needs a single_threaded map");
            return map.entry_count;
        }

        /// Iterates over the entries in `range`, in key order, a leaf at a time: each leaf's
        /// entries are a consistent snapshot, but the map may change between leaves.
        pub fn iterator(map: *const Self, range: Range) Iterator {
            return .{
                .map = map,
                .scan = .init(range),
                .buffer = undefined,
                .buffer_count = 0,
                .buffer_index = 0,
            };
        }

        pub const Iterator = struct {
            map: *const Self,
            scan: Scan,
            buffer: [LeafNode.capacity]Entry,
            buffer_count: u32,
            buffer_index: u32,

            pub fn next(iterator_state: *Iterator) ?Entry {
                if (iterator_state.buffer_index == iterator_state.buffer_count) {
                    // A shared map's copy, with its validation and retries, takes the
                    // registers of a caller's loop over `next` when inlined, and the loop's
                    // own state then goes through memory. A single-threaded copy is small.
                    const modifier: std.builtin.CallModifier =
                        if (concurrency == .single_threaded) .always_inline else .never_inline;
                    if (!@call(modifier, fill, .{iterator_state})) return null;
                }
                const entry = iterator_state.buffer[iterator_state.buffer_index];
                iterator_state.buffer_index += 1;
                return entry;
            }

            /// Copies the next leaf's entries into the buffer. Returns false at the range's
            /// end.
            fn fill(iterator_state: *Iterator) bool {
                assert(iterator_state.buffer_index == iterator_state.buffer_count);
                // Each copy takes the rest of a leaf, which fits the buffer, or moves on.
                while (iterator_state.buffer_index == iterator_state.buffer_count) {
                    if (iterator_state.scan.done) return false;
                    const map = iterator_state.map;
                    const buffer = &iterator_state.buffer;
                    iterator_state.buffer_count = iterator_state.scan.copy(map, buffer);
                    iterator_state.buffer_index = 0;
                }
                return true;
            }
        };

        /// Copies the entries in `range` into `target`, in key order, until it fills, and returns
        /// how many. Each leaf's entries are a consistent snapshot, but the map may change
        /// between leaves. To continue, scan again from just after the last key copied.
        pub fn scan(map: *const Self, range: Range, target: []Entry) usize {
            var position: Scan = .init(range);
            var entry_count: usize = 0;
            while (entry_count < target.len and !position.done) {
                entry_count += position.copy(map, target[entry_count..]);
            }
            assert(entry_count <= target.len);
            return entry_count;
        }

        /// A range scan between its copies: where the next copy starts, and the parent of the
        /// leaf copied last, through which the next leaf is usually one step away.
        pub const Scan = struct {
            start: KeyBound,
            end: KeyBound,
            parent: ?ScanParent,
            done: bool,

            pub fn init(range: Range) Scan {
                return .{ .start = range.start, .end = range.end, .parent = null, .done = false };
            }

            /// Copies entries of one leaf into `target`: from the scan's start on, up to the
            /// range's end and `target`'s length. Returns how many.
            pub fn copy(scan_state: *Scan, map: *const Self, target: []Entry) u32 {
                assert(!scan_state.done);
                assert(target.len > 0);
                const leaf_copy = map.scan_leaf(scan_state.start, &scan_state.parent, target);
                assert(leaf_copy.entry_count <= target.len);
                if (scan_state.end != .unbounded) {
                    for (target[0..leaf_copy.entry_count], 0..) |entry, index| {
                        if (beyond_end(entry.key, scan_state.end)) {
                            scan_state.done = true;
                            return @intCast(index);
                        }
                    }
                }
                scan_state.advance(leaf_copy, target);
                return leaf_copy.entry_count;
            }

            /// Moves the start past a copy: to just after its last key if the leaf has more,
            /// or else to the leaf's upper fence, where the next leaf starts.
            fn advance(scan_state: *Scan, leaf_copy: LeafCopy, target: []const Entry) void {
                if (!leaf_copy.leaf_finished) {
                    assert(leaf_copy.entry_count == target.len);
                    scan_state.start = .{ .excluded = target[leaf_copy.entry_count - 1].key };
                    // The next copy takes up the same leaf again, through a descent.
                    scan_state.parent = null;
                    return;
                }
                const high = leaf_copy.high orelse {
                    scan_state.done = true;
                    return;
                };
                // The upper fence lies above every key the leaf covers, the start included,
                // so each copy moves on.
                switch (scan_state.start) {
                    .unbounded => {},
                    .included, .excluded => |start| assert(Codec.order(start, high) == .lt),
                }
                scan_state.start = .{ .included = high };
                scan_state.done = beyond_end(high, scan_state.end);
            }
        };

        fn beyond_end(key: Key, end: KeyBound) bool {
            return switch (end) {
                .unbounded => false,
                .included => |bound| Codec.order(key, bound) == .gt,
                .excluded => |bound| Codec.order(key, bound) != .lt,
            };
        }

        /// Paces an operation's attempts.
        const Retries = struct {
            backoff: Backoff = .{},
            restructure_count: u32 = 0,

            fn wait(retries: *Retries, err: Retry) void {
                switch (err) {
                    error.Restructured => {
                        retries.restructure_count += 1;
                        // Alone, an operation restructures each level at most twice: once on
                        // its way down, once more when a change below fills or drains it. An
                        // insert may shrink the shared bits once before.
                        if (concurrency == .single_threaded) {
                            assert(retries.restructure_count <= 2 * height_max + 3);
                        }
                    },
                    error.Restart => {
                        // A single thread never finds a page changed under it.
                        assert(concurrency == .multi_threaded);
                        retries.backoff.wait();
                    },
                }
            }
        };

        fn leaf(map: *const Self, page: u32) LeafNode.PagePointer {
            assert(page < map.leaves.page_count_max());
            return &map.head.leaf_pages[page].page;
        }

        fn internal(map: *const Self, page: u32) InternalNode.PagePointer {
            assert(page < map.internals.page_count_max());
            return &map.head.internal_pages[page].page;
        }

        fn latch_word(map: *const Self, page: u32, height: u8) *u64 {
            if (height == 0) return &LeafNode.header(map.leaf(page)).latch;
            return &InternalNode.header(map.internal(page)).latch;
        }

        fn root_load(map: *const Self) Root {
            if (concurrency == .single_threaded) return @bitCast(map.head.root_word);
            return @bitCast(@atomicLoad(u64, &map.head.root_word, .acquire));
        }

        /// Publishes a new root, or new shared bits. Callers hold the root page's latch, and
        /// store before they let go of it, so a reader that sees the old root unlatched also
        /// sees the new value.
        fn root_store(map: *Self, root: Root) void {
            assert(root.height <= height_max);
            assert(root.shared_length <= shared_length_max or
                root.shared_length == shared_length_unset);
            if (concurrency == .single_threaded) {
                map.head.root_word = @bitCast(root);
            } else {
                @atomicStore(u64, &map.head.root_word, @bitCast(root), .release);
            }
        }

        /// The shared bits, for a writer that sets a node's prefix. A writer holding a node
        /// on the tree's border holds them still, as shrinking them takes every border node.
        /// Elsewhere both fences share more bits anyway, whichever value it reads.
        fn shared_length(map: *const Self) u32 {
            const length = map.root_load().shared_length;
            return if (length == shared_length_unset) 0 else length;
        }

        fn rightmost_load(map: *const Self) u32 {
            if (concurrency == .single_threaded) return map.rightmost_page;
            return @atomicLoad(u32, &map.rightmost_page, .monotonic);
        }

        fn rightmost_store(map: *Self, page: u32) void {
            if (concurrency == .single_threaded) {
                map.rightmost_page = page;
            } else {
                @atomicStore(u32, &map.rightmost_page, page, .monotonic);
            }
        }

        fn root_read(map: *const Self) Retry!Start {
            const root = map.root_load();
            const version = Latch.version_read(map.latch_word(root.page, root.height));
            if (map.root_load() != root) return error.Restart;
            return .{
                .node = .{ .page = root.page, .version = version, .height = root.height },
                .shared_length = root.shared_length,
            };
        }

        /// How `key` sorts against the bits every key shares: `.eq` if it has them, or else
        /// before or after every key in the map. Reads the bits racily: the root's version,
        /// read with `shared_length`, vouches for them.
        fn shared_order(map: *const Self, length: u16, key: Key) std.math.Order {
            if (length == 0) return .eq;
            if (length == shared_length_unset) return .lt;
            assert(length <= shared_length_max);
            inline for (0..shared_word_count) |index| {
                const offset: u32 = index * 64;
                if (offset >= length) return .eq;
                const key_word = Codec.window(key, offset) & word_mask(length - offset);
                const shared_word = access.racy(u64, &map.head.shared_words[index]);
                if (key_word != shared_word) return std.math.order(key_word, shared_word);
            }
            return .eq;
        }

        /// The leading `bits` bits of a word.
        fn word_mask(bits: u32) u64 {
            if (bits == 0) return 0;
            if (bits >= 64) return std.math.maxInt(u64);
            return ~(@as(u64, std.math.maxInt(u64)) >> @intCast(bits));
        }

        /// Checks that the root is still at the version a descent started from, so that a
        /// key the shared bits ruled out was absent then.
        fn root_validate(map: *const Self, root: Cursor) Retry!void {
            try Latch.validate(map.latch_word(root.page, root.height), root.version);
        }

        /// Steps from an internal node to the child behind `edge`. The node's version vouches for
        /// the edge before the child is touched, and again after the child's version is read,
        /// so the child was still the node's child then.
        fn child_read(map: *const Self, node: Cursor, edge: u32) Retry!Cursor {
            assert(node.height > 0);
            const node_word = map.latch_word(node.page, node.height);
            try Latch.validate(node_word, node.version);
            const version = Latch.version_read(map.latch_word(edge, node.height - 1));
            try Latch.validate(node_word, node.version);
            return .{ .page = edge, .version = version, .height = node.height - 1 };
        }

        /// The leaf for `key`, read optimistically, or null if `key` lacks the bits every key
        /// shares and so lies in no leaf. Inline, as the loop of every lookup: out of line, its
        /// cursor would travel back through memory.
        inline fn descend(map: *const Self, key: Key) Retry!?Cursor {
            const start = try map.root_read();
            if (map.shared_order(start.shared_length, key) != .eq) {
                try map.root_validate(start.node);
                return null;
            }
            var node = start.node;
            while (node.height > 0) {
                node = (try map.step(node, key, .lookup)).child;
            }
            return node;
        }

        const Step = struct {
            edge_index: u32,
            child: Cursor,
        };

        /// What a descent is for. A lookup reads a whole entry it finds in a leaf; a change
        /// writes a few lines near its key.
        const Purpose = enum { lookup, change };

        /// One step of a descent: routes `key` through an internal node, and reads the child
        /// it leads to.
        inline fn step(
            map: *const Self,
            node: Cursor,
            key: Key,
            comptime purpose: Purpose,
        ) Retry!Step {
            const page = map.internal(node.page);
            const edge_index = try InternalNode.route(.optimistic, page, node.version, key);
            const edge = InternalNode.edge_racy(page, edge_index);
            if (purpose == .lookup and node.height == 1) map.leaf_prefetch(edge);
            return .{ .edge_index = edge_index, .child = try map.child_read(node, edge) };
        }

        /// Below this many bytes of leaves, lookups leave a leaf's lines to its search: the
        /// leaves stay in the outer caches, where a prefetch only adds traffic.
        const leaf_prefetch_bytes_min = 128 << 20;
        const leaf_prefetch_page_min = leaf_prefetch_bytes_min / LeafNode.page_size;
        const leaf_prefetch_bytes = @min(LeafNode.page_size, 2048);

        /// Starts loading the leaf a lookup is about to enter, once the leaves outgrow the
        /// caches. A lookup reads the leaf's window column, then the entry the windows lead
        /// to: two misses, one after the other, which the prefetch overlaps. Lines past the
        /// column are streamed, so that they don't flush the internal nodes out of the outer
        /// caches. Changes go without: the extra traffic costs them more than it saves. The
        /// edge was read racily, so its page goes unchecked: a prefetch never faults.
        fn leaf_prefetch(map: *const Self, page: u32) void {
            if (!access.load(bool, &map.head.leaf_prefetch)) return;
            const lines = map.leaf_unchecked(page);
            @prefetch(lines, .{ .rw = .read, .locality = 3, .cache = .data });
            const line_bytes = std.atomic.cache_line;
            comptime var line: u32 = line_bytes;
            inline while (line < leaf_prefetch_bytes) : (line += line_bytes) {
                @prefetch(lines + line, .{ .rw = .read, .locality = 0, .cache = .data });
            }
        }

        /// Where a leaf page lies, for an index read racily that only a prefetch uses: any
        /// index gives an address, and a prefetch never faults.
        fn leaf_unchecked(map: *const Self, page: u32) [*]const u8 {
            const offset = @as(usize, page) * @sizeOf(LeafPool.Slot);
            return @ptrFromInt(@intFromPtr(map.head.leaf_pages) + offset);
        }

        fn leaf_prefetch_enable(map: *Self) void {
            // Read first: every lookup reads this line, so writes stop once the flag is seen.
            if (access.load(bool, &map.head.leaf_prefetch)) return;
            access.store(bool, &map.head.leaf_prefetch, true);
        }

        inline fn get_attempt(map: *const Self, key: Key) Retry!?Value {
            const node = try map.descend(key) orelse return null;
            return try LeafNode.get_racy(map.leaf(node.page), node.version, key);
        }

        /// Lookups in flight at once in `get_many`.
        const lane_count = 64;
        /// Below this many bytes of leaves the tree stays in cache, and lookups go one at a time.
        const lane_leaf_bytes_min = 1 << 20;

        /// Looks up every key of `keys` into `values`. Lookups run in lanes that overlap
        /// their cache misses: a lane prefetches the line it reads next, and the others run
        /// while it arrives. A descent's misses are chained, so this is the only way to overlap
        /// them.
        pub fn get_many(map: *const Self, keys: []const Key, values: []?Value) void {
            assert(keys.len == values.len);
            const leaf_page_count: u64 = map.leaves.page_count_used.load(.monotonic);
            if (leaf_page_count * LeafNode.page_size < lane_leaf_bytes_min) {
                for (keys, values) |key, *value| value.* = map.get(key);
                return;
            }
            var start: usize = 0;
            while (start < keys.len) : (start += lane_count) {
                const end = @min(start + lane_count, keys.len);
                map.lanes_run(keys[start..end], values[start..end]);
            }
        }

        /// One lookup of a batch, as a state machine that stops after each prefetch.
        const Lane = struct {
            node: Cursor,
            stage: Stage,
            edge_index: u32,
            edge: u32,
            candidates: LeafNode.Candidates,

            const Stage = enum {
                /// The node's edge line is on its way: read the edge, prefetch the child.
                follow,
                /// The child's search lines are on their way: step down and search it.
                enter,
                /// The leaf's candidate entry is on its way: compare it and copy the value.
                finish,
                done,
            };
        };

        fn lanes_run(map: *const Self, keys: []const Key, values: []?Value) void {
            assert(keys.len <= lane_count);
            assert(keys.len == values.len);
            var lanes: [lane_count]Lane = undefined;
            var active_count: usize = 0;
            for (lanes[0..keys.len], keys, values) |*lane, key, *value| {
                value.* = null;
                map.lane_start(lane, key);
                active_count += @intFromBool(lane.stage != .done);
            }
            // Every round moves each active lane one stage on, or back to the root after a
            // restart; a lane stays at most two stages per level of the tree.
            while (active_count > 0) {
                for (lanes[0..keys.len], keys, values) |*lane, key, *value| {
                    if (lane.stage == .done) continue;
                    map.lane_step(lane, key, value) catch {
                        assert(concurrency == .multi_threaded);
                        map.lane_start(lane, key);
                        if (lane.stage == .done) active_count -= 1;
                        continue;
                    };
                    if (lane.stage == .done) active_count -= 1;
                }
            }
        }

        /// Starts a lane at the root, or finishes it there if its key lacks the shared bits.
        fn lane_start(map: *const Self, lane: *Lane, key: Key) void {
            var retries: Retries = .{};
            while (true) {
                const start = map.root_read() catch |err| {
                    retries.wait(err);
                    continue;
                };
                if (map.shared_order(start.shared_length, key) != .eq) {
                    map.root_validate(start.node) catch |err| {
                        retries.wait(err);
                        continue;
                    };
                    lane.stage = .done;
                    return;
                }
                map.lane_search(lane, start.node, key) catch |err| {
                    retries.wait(err);
                    continue;
                };
                return;
            }
        }

        fn lane_step(map: *const Self, lane: *Lane, key: Key, value: *?Value) Retry!void {
            switch (lane.stage) {
                .follow => {
                    const page = map.internal(lane.node.page);
                    lane.edge = InternalNode.edge_racy(page, lane.edge_index);
                    try Latch.validate(&InternalNode.header(page).latch, lane.node.version);
                    if (lane.node.height == 1) {
                        prefetch(map.leaf(lane.edge), LeafNode.search_prefix_bytes);
                    } else {
                        prefetch(map.internal(lane.edge), InternalNode.search_prefix_bytes);
                    }
                    lane.stage = .enter;
                },
                .enter => try map.lane_search(lane, try map.child_read(lane.node, lane.edge), key),
                .finish => {
                    value.* = try LeafNode.get_racy_candidates(
                        map.leaf(lane.node.page),
                        lane.node.version,
                        key,
                        lane.candidates,
                    );
                    lane.stage = .done;
                },
                .done => unreachable,
            }
        }

        /// Searches the node the lane stepped onto, and prefetches what the search leads to.
        fn lane_search(map: *const Self, lane: *Lane, node: Cursor, key: Key) Retry!void {
            lane.node = node;
            if (node.height == 0) {
                const page = map.leaf(node.page);
                lane.candidates = LeafNode.candidates_racy(page, key);
                if (lane.candidates.start < lane.candidates.end) {
                    prefetch(LeafNode.entry_address(page, lane.candidates.start), @sizeOf(Entry));
                }
                lane.stage = .finish;
            } else {
                const page = map.internal(node.page);
                lane.edge_index = try InternalNode.route(.optimistic, page, node.version, key);
                prefetch(InternalNode.edge_address(page, lane.edge_index), 4);
                lane.stage = .follow;
            }
        }

        fn prefetch(address: [*]const u8, comptime byte_count: u32) void {
            const line_bytes = std.atomic.cache_line;
            comptime var offset: u32 = 0;
            inline while (offset < byte_count) : (offset += line_bytes) {
                @prefetch(address + offset, .{ .rw = .read, .locality = 3, .cache = .data });
            }
        }

        fn update_attempt(
            map: *Self,
            key: Key,
            context: anytype,
            comptime update_fn: fn (@TypeOf(context), *Value) void,
        ) Retry!bool {
            const node = try map.descend(key) orelse return false;
            const page = map.leaf(node.page);
            const word = &LeafNode.header(page).latch;
            try Latch.try_lock(word, node.version);
            const position = LeafNode.search(.exclusive, page, node.version, key) catch unreachable;
            if (!position.found) {
                Latch.unlock_unchanged(word);
                return false;
            }
            update_fn(context, LeafNode.value_pointer(page, position.index));
            Latch.unlock(word);
            return true;
        }

        fn put_loop(
            map: *Self,
            key: Key,
            value: Value,
            comptime policy: PutPolicy,
        ) error{MapFull}!?Value {
            var retries: Retries = .{};
            while (true) {
                return map.put_attempt(key, value, policy) catch |err| switch (err) {
                    error.MapFull => return error.MapFull,
                    error.Restart, error.Restructured => |retry| {
                        retries.wait(retry);
                        continue;
                    },
                };
            }
        }

        fn put_attempt(
            map: *Self,
            key: Key,
            value: Value,
            comptime policy: PutPolicy,
        ) (Retry || error{MapFull})!?Value {
            if (map.append_attempt(key, value)) return null;
            const start = try map.root_read();
            if (map.shared_order(start.shared_length, key) != .eq) {
                try map.shared_shrink(start.node, key);
                return error.Restructured;
            }
            var parent: ?Cursor = null;
            var edge_index: u32 = 0;
            var node = start.node;
            while (node.height > 0) {
                const page = map.internal(node.page);
                if (InternalNode.count_racy(page) == InternalNode.capacity) {
                    try map.split_internal(parent, edge_index, node);
                    return error.Restructured;
                }
                const next = try map.step(node, key, .change);
                edge_index = next.edge_index;
                parent = node;
                node = next.child;
            }
            return map.put_leaf(parent, edge_index, node, key, value, policy);
        }

        /// A node on the tree's border as a shrink of the shared bits latched it, with its
        /// fences.
        const Border = struct {
            page: u32,
            height: u8,
            fences: InternalNode.Fences,
        };

        const Borders = [1 + 2 * height_max]Border;

        /// Shrinks the shared bits to those that `key` has too, for an insert of a key that
        /// lacks them; the first insert sets them to the key's own. The root and the nodes
        /// along the tree's two borders take the shared bits as their prefix where a fence is
        /// missing, so the shrink latches every one of them, without waiting, before it
        /// changes anything, and then refits their prefixes.
        fn shared_shrink(map: *Self, root: Cursor, key: Key) Retry!void {
            var borders: Borders = undefined;
            var border_count: u32 = 0;
            errdefer for (borders[0..border_count]) |border| {
                Latch.unlock_unchanged(map.latch_word(border.page, border.height));
            };
            try Latch.try_lock(map.latch_word(root.page, root.height), root.version);
            borders[0] = .{
                .page = root.page,
                .height = root.height,
                .fences = .{ .low = null, .high = null },
            };
            border_count = 1;
            try map.borders_lock(&borders, &border_count);

            var root_new = map.root_load();
            var words: [shared_word_count]u64 = undefined;
            root_new.shared_length = map.shared_after(root_new.shared_length, key, &words);
            for (&map.head.shared_words, words) |*target, word| access.store(u64, target, word);
            map.root_store(root_new);
            for (borders[0..border_count]) |border| {
                map.border_prefix_set(border, root_new.shared_length);
            }
            for (borders[0..border_count]) |border| {
                Latch.unlock(map.latch_word(border.page, border.height));
            }
        }

        /// The shared bits once `key` joins the map's keys: those it shares with them, or for
        /// the first key, as many of its own as the map keeps. Writes their words to `words`.
        fn shared_after(
            map: *const Self,
            length_before: u16,
            key: Key,
            words: *[shared_word_count]u64,
        ) u16 {
            const first = length_before == shared_length_unset;
            var length: u32 = if (first) shared_length_max else length_before;
            for (words, 0..) |*word, index| {
                const offset: u32 = @intCast(index * 64);
                const key_word = Codec.window(key, offset) & word_mask(length -| offset);
                if (!first) {
                    const difference = key_word ^ map.head.shared_words[index];
                    if (difference != 0) length = @min(length, offset + @clz(difference));
                }
                word.* = key_word & word_mask(length -| offset);
            }
            assert(length < length_before);
            return @intCast(length);
        }

        /// Latches the nodes along both borders below the root, which `borders` holds first,
        /// top down and without waiting, and records them in `borders`.
        fn borders_lock(map: *Self, borders: *Borders, border_count: *u32) Retry!void {
            assert(border_count.* == 1);
            var left = borders[0];
            var right = borders[0];
            while (left.height > 0) {
                assert(right.height == left.height);
                const right_edge = InternalNode.count(map.internal(right.page));
                const left_child = map.border_child(left, 0);
                const right_child = map.border_child(right, right_edge);
                try map.border_lock(borders, border_count, left_child);
                if (right_child.page != left_child.page) {
                    try map.border_lock(borders, border_count, right_child);
                }
                left = left_child;
                right = right_child;
            }
        }

        fn border_child(map: *const Self, parent: Border, edge_index: u32) Border {
            const page = map.internal(parent.page);
            return .{
                .page = InternalNode.edge_at(page, edge_index),
                .height = parent.height - 1,
                .fences = InternalNode.child_fences(page, edge_index),
            };
        }

        fn border_lock(
            map: *Self,
            borders: *Borders,
            border_count: *u32,
            border: Border,
        ) Retry!void {
            const word = map.latch_word(border.page, border.height);
            try Latch.try_lock(word, try Latch.version_try_read(word));
            borders[border_count.*] = border;
            border_count.* += 1;
        }

        fn border_prefix_set(map: *Self, border: Border, length: u32) void {
            if (border.height == 0) {
                const prefix_length = InternalNode.prefix_of(border.fences, length);
                LeafNode.prefix_set(map.leaf(border.page), prefix_length);
            } else {
                InternalNode.prefix_set(map.internal(border.page), length);
            }
        }

        /// Appends `key` to the tree's last leaf if it sorts after every key there, as ascending
        /// keys do: one latch, no descent. Returns false where that does not apply, or when
        /// the leaf is busy, and leaves the insert to a descent.
        fn append_attempt(map: *Self, key: Key, value: Value) bool {
            const page = map.leaf(map.rightmost_load());
            const word = &LeafNode.header(page).latch;
            const version = Latch.version_try_read(word) catch return false;
            if (!LeafNode.rightmost_racy(page)) return false;
            const entry_count = LeafNode.count_racy(page);
            if (entry_count == 0) return false;
            if (entry_count == LeafNode.capacity) return false;
            const last = LeafNode.key_racy(page, entry_count - 1);
            if (Codec.indirect) Latch.validate(word, version) catch return false;
            if (Codec.order(last, key) != .lt) return false;
            // The latch confirms the flag, the count and the last key read above.
            Latch.try_lock(word, version) catch return false;
            assert(LeafNode.rightmost(page));
            // The leaf lies on the tree's border, so holding it holds the shared bits still.
            if (map.shared_order(map.root_load().shared_length, key) != .eq) {
                Latch.unlock_unchanged(word);
                return false;
            }
            LeafNode.insert_at(page, entry_count, key, value);
            Latch.unlock(word);
            if (concurrency == .single_threaded) map.entry_count += 1;
            return true;
        }

        fn put_leaf(
            map: *Self,
            parent: ?Cursor,
            edge_index: u32,
            node: Cursor,
            key: Key,
            value: Value,
            comptime policy: PutPolicy,
        ) (Retry || error{MapFull})!?Value {
            const page = map.leaf(node.page);
            const word = &LeafNode.header(page).latch;
            try Latch.try_lock(word, node.version);
            const position = LeafNode.search(.exclusive, page, node.version, key) catch unreachable;
            if (position.found) {
                const slot = LeafNode.value_pointer(page, position.index);
                const previous = slot.*;
                if (policy == .replace) {
                    slot.* = value;
                    Latch.unlock(word);
                } else {
                    Latch.unlock_unchanged(word);
                }
                return previous;
            }
            if (LeafNode.count(page) < LeafNode.capacity) {
                LeafNode.insert_at(page, position.index, key, value);
                Latch.unlock(word);
            } else {
                try map.split_leaf(parent, edge_index, node, position.index, key, value);
            }
            if (concurrency == .single_threaded) map.entry_count += 1;
            return null;
        }

        /// Splits a full, latched leaf and inserts `key`. Lets go of every page it latched,
        /// the leaf included, whether it succeeds or not.
        fn split_leaf(
            map: *Self,
            parent: ?Cursor,
            edge_index: u32,
            node: Cursor,
            position: u32,
            key: Key,
            value: Value,
        ) (Retry || error{MapFull})!void {
            const leaf_word = map.latch_word(node.page, 0);
            assert(Latch.is_locked(leaf_word));
            errdefer Latch.unlock_unchanged(leaf_word);

            const parent_cursor = parent orelse return map.split_root_leaf(
                node,
                position,
                key,
                value,
            );
            const parent_word = map.latch_word(parent_cursor.page, parent_cursor.height);
            try Latch.try_lock(parent_word, parent_cursor.version);
            errdefer Latch.unlock_unchanged(parent_word);
            const parent_page = map.internal(parent_cursor.page);
            assert(InternalNode.count(parent_page) < InternalNode.capacity);

            const right = map.leaves.acquire() orelse return error.MapFull;
            map.split_leaf_into(parent_page, edge_index, node.page, right, position, key, value);
            Latch.unlock(map.latch_word(right, 0));
            Latch.unlock(leaf_word);
            Latch.unlock(parent_word);
        }

        fn split_root_leaf(
            map: *Self,
            node: Cursor,
            position: u32,
            key: Key,
            value: Value,
        ) error{MapFull}!void {
            assert(node.height == 0);
            const root_page = map.internals.acquire() orelse return error.MapFull;
            const right = map.leaves.acquire() orelse {
                Latch.unlock(map.latch_word(root_page, 1));
                map.internals.release(root_page);
                return error.MapFull;
            };
            const root = map.internal(root_page);
            InternalNode.init(root, .{ .low = null, .high = null }, node.page, map.shared_length());
            map.split_leaf_into(root, 0, node.page, right, position, key, value);
            Latch.unlock(map.latch_word(right, 0));
            Latch.unlock(map.latch_word(root_page, 1));
            map.root_store(map.root_with(root_page, 1));
            Latch.unlock(map.latch_word(node.page, 0));
        }

        /// Moves the upper part of a full leaf into the fresh leaf `right`, links it into the
        /// parent, and inserts `key` into whichever half covers it. Inserting at the end, as
        /// ascending keys do, leaves the right leaf only its minimum, so that ascending
        /// inserts fill leaves three-quarters full rather than half.
        fn split_leaf_into(
            map: *Self,
            parent: InternalNode.PagePointer,
            edge_index: u32,
            left_page: u32,
            right_page: u32,
            position: u32,
            key: Key,
            value: Value,
        ) void {
            const left = map.leaf(left_page);
            const right = map.leaf(right_page);
            const entry_count = LeafNode.count(left);
            assert(entry_count == LeafNode.capacity);
            assert(position <= entry_count);
            const split_index =
                if (position == entry_count) entry_count - LeafNode.count_min else entry_count / 2;

            LeafNode.init(right, 0);
            LeafNode.move_tail(left, right, split_index);
            // Pools hand out a page past those in use only when none is free, so a page's
            // index stays below the most leaves the map has held at once.
            if (right_page >= leaf_prefetch_page_min) map.leaf_prefetch_enable();
            // A split writes its fresh page at once, and a page the pool has never handed out
            // is cold: start loading the next one for the next split.
            if (map.leaves.fresh_next()) |next| {
                const lines: [*]const u8 = @ptrCast(map.leaf(next));
                comptime var line: u32 = 0;
                inline while (line < LeafNode.page_size) : (line += std.atomic.cache_line) {
                    @prefetch(lines + line, .{ .rw = .write, .locality = 3, .cache = .data });
                }
            }
            if (LeafNode.rightmost(left)) {
                LeafNode.rightmost_set(left, false);
                LeafNode.rightmost_set(right, true);
                map.rightmost_store(right_page);
            }
            InternalNode.insert_separator(
                parent,
                edge_index,
                LeafNode.key_at(right, 0),
                right_page,
            );
            map.leaf_prefix_fit(parent, edge_index, left);
            map.leaf_prefix_fit(parent, edge_index + 1, right);
            if (position <= split_index) {
                LeafNode.insert_at(left, position, key, value);
            } else {
                LeafNode.insert_at(right, position - split_index, key, value);
            }
        }

        fn leaf_prefix_set(
            map: *const Self,
            parent: InternalNode.PagePointer,
            edge_index: u32,
            page: LeafNode.PagePointer,
        ) void {
            const fences = InternalNode.child_fences(parent, edge_index);
            LeafNode.prefix_set(page, InternalNode.prefix_of(fences, map.shared_length()));
        }

        fn leaf_prefix_fit(
            map: *const Self,
            parent: InternalNode.PagePointer,
            edge_index: u32,
            page: LeafNode.PagePointer,
        ) void {
            const fences = InternalNode.child_fences(parent, edge_index);
            LeafNode.prefix_fit(page, InternalNode.prefix_of(fences, map.shared_length()));
        }

        /// The root as it is, moved to another page and height. For the holder of the root
        /// page's latch, under which the shared bits stay as they are.
        fn root_with(map: *const Self, page: u32, height: u8) Root {
            var root = map.root_load();
            root.page = page;
            root.height = height;
            return root;
        }

        /// Splits a full internal node, latching it and its parent at the versions the descent
        /// read; a root splits under a new root.
        fn split_internal(
            map: *Self,
            parent: ?Cursor,
            edge_index: u32,
            node: Cursor,
        ) (Retry || error{MapFull})!void {
            const parent_word: ?*u64 = if (parent) |cursor|
                map.latch_word(cursor.page, cursor.height)
            else
                null;
            if (parent) |cursor| try Latch.try_lock(parent_word.?, cursor.version);
            errdefer if (parent_word) |word| Latch.unlock_unchanged(word);
            const node_word = map.latch_word(node.page, node.height);
            try Latch.try_lock(node_word, node.version);
            errdefer Latch.unlock_unchanged(node_word);

            const page = map.internal(node.page);
            assert(InternalNode.count(page) == InternalNode.capacity);
            const right = map.internals.acquire() orelse return error.MapFull;
            const length = map.shared_length();
            if (parent) |cursor| {
                const parent_page = map.internal(cursor.page);
                assert(InternalNode.count(parent_page) < InternalNode.capacity);
                const separator = InternalNode.split(page, map.internal(right), length);
                InternalNode.insert_separator(parent_page, edge_index, separator, right);
                Latch.unlock(map.latch_word(right, node.height));
                Latch.unlock(node_word);
                Latch.unlock(parent_word.?);
                return;
            }
            const root_page = map.internals.acquire() orelse {
                Latch.unlock(map.latch_word(right, node.height));
                map.internals.release(right);
                return error.MapFull;
            };
            const root = map.internal(root_page);
            InternalNode.init(root, .{ .low = null, .high = null }, node.page, length);
            const separator = InternalNode.split(page, map.internal(right), length);
            InternalNode.insert_separator(root, 0, separator, right);
            Latch.unlock(map.latch_word(right, node.height));
            Latch.unlock(map.latch_word(root_page, node.height + 1));
            map.root_store(map.root_with(root_page, node.height + 1));
            Latch.unlock(node_word);
        }

        fn remove_attempt(map: *Self, key: Key) Retry!?Value {
            const start = try map.root_read();
            var node = start.node;
            if (node.height > 0 and InternalNode.count_racy(map.internal(node.page)) == 0) {
                try map.collapse_root(node);
                return error.Restructured;
            }
            if (map.shared_order(start.shared_length, key) != .eq) {
                try map.root_validate(node);
                return null;
            }
            const height_root = node.height;
            while (node.height > 0) {
                const next = try map.step(node, key, .change);
                if (map.count_racy(next.child) <= count_min(next.child.height)) {
                    try map.rebalance(node, next.edge_index, next.child);
                    return error.Restructured;
                }
                node = next.child;
            }
            return map.remove_leaf(node, key, node.height == height_root);
        }

        fn remove_leaf(map: *Self, node: Cursor, key: Key, is_root: bool) Retry!?Value {
            const page = map.leaf(node.page);
            const word = &LeafNode.header(page).latch;
            try Latch.try_lock(word, node.version);
            const position = LeafNode.search(.exclusive, page, node.version, key) catch unreachable;
            if (!position.found) {
                Latch.unlock_unchanged(word);
                return null;
            }
            if (!is_root) assert(LeafNode.count(page) > LeafNode.count_min);
            const removed = LeafNode.remove_at(page, position.index);
            Latch.unlock(word);
            if (concurrency == .single_threaded) map.entry_count -= 1;
            return removed.value;
        }

        fn count_racy(map: *const Self, node: Cursor) u32 {
            if (node.height == 0) return LeafNode.count_racy(map.leaf(node.page));
            return InternalNode.count_racy(map.internal(node.page));
        }

        fn count_min(height: u8) u32 {
            return if (height == 0) LeafNode.count_min else InternalNode.count_min;
        }

        /// Lowers the tree when the root has a single child.
        fn collapse_root(map: *Self, node: Cursor) Retry!void {
            assert(node.height > 0);
            const word = map.latch_word(node.page, node.height);
            try Latch.try_lock(word, node.version);
            const page = map.internal(node.page);
            assert(InternalNode.count(page) == 0);
            map.root_store(map.root_with(InternalNode.edge_at(page, 0), node.height - 1));
            Latch.unlock(word);
            map.internals.release(node.page);
        }

        /// Brings a child at its minimum above it: merges it with a sibling if both fit in one
        /// node, or else evens the two out. Latches the parent, the child and the sibling
        /// without waiting.
        fn rebalance(map: *Self, parent: Cursor, edge_index: u32, child: Cursor) Retry!void {
            const parent_word = map.latch_word(parent.page, parent.height);
            try Latch.try_lock(parent_word, parent.version);
            errdefer Latch.unlock_unchanged(parent_word);
            const child_word = map.latch_word(child.page, child.height);
            try Latch.try_lock(child_word, child.version);
            errdefer Latch.unlock_unchanged(child_word);

            const parent_page = map.internal(parent.page);
            assert(InternalNode.count(parent_page) > 0);
            assert(map.count_racy(child) <= count_min(child.height));
            const sibling_index = if (edge_index > 0) edge_index - 1 else edge_index + 1;
            const sibling = InternalNode.edge_at(parent_page, sibling_index);
            const sibling_word = map.latch_word(sibling, child.height);
            try Latch.try_lock(sibling_word, try Latch.version_try_read(sibling_word));

            const separator_index = @min(edge_index, sibling_index);
            const left = if (edge_index < sibling_index) child.page else sibling;
            const right = if (edge_index < sibling_index) sibling else child.page;
            const merged = if (child.height == 0)
                map.rebalance_leaves(parent_page, separator_index, left, right)
            else
                map.rebalance_internals(parent_page, separator_index, left, right);
            Latch.unlock(sibling_word);
            Latch.unlock(child_word);
            Latch.unlock(parent_word);
            if (merged) {
                if (child.height == 0) map.leaves.release(right) else map.internals.release(right);
            }
        }

        fn rebalance_leaves(
            map: *Self,
            parent: InternalNode.PagePointer,
            separator_index: u32,
            left_page: u32,
            right_page: u32,
        ) bool {
            const left = map.leaf(left_page);
            const right = map.leaf(right_page);
            const left_count = LeafNode.count(left);
            const right_count = LeafNode.count(right);
            if (left_count + right_count <= LeafNode.capacity) {
                LeafNode.move_all(left, right);
                if (LeafNode.rightmost(right)) {
                    LeafNode.rightmost_set(right, false);
                    LeafNode.rightmost_set(left, true);
                    map.rightmost_store(left_page);
                }
                InternalNode.remove_separator(parent, separator_index);
                map.leaf_prefix_set(parent, separator_index, left);
                return true;
            }
            const left_target = (left_count + right_count) / 2;
            if (left_count < left_target) {
                LeafNode.move_from_right(left, right, left_target - left_count);
            } else {
                LeafNode.move_from_left(left, right, left_count - left_target);
            }
            InternalNode.separator_set(parent, separator_index, LeafNode.key_at(right, 0));
            map.leaf_prefix_set(parent, separator_index, left);
            map.leaf_prefix_set(parent, separator_index + 1, right);
            assert(LeafNode.count(left) > LeafNode.count_min);
            assert(LeafNode.count(right) > LeafNode.count_min);
            return false;
        }

        fn rebalance_internals(
            map: *Self,
            parent: InternalNode.PagePointer,
            separator_index: u32,
            left_page: u32,
            right_page: u32,
        ) bool {
            const left = map.internal(left_page);
            const right = map.internal(right_page);
            const left_count = InternalNode.count(left);
            const right_count = InternalNode.count(right);
            const separator = InternalNode.key_at(parent, separator_index);
            const length = map.shared_length();
            if (left_count + 1 + right_count <= InternalNode.capacity) {
                InternalNode.merge(left, separator, right, length);
                InternalNode.remove_separator(parent, separator_index);
                return true;
            }
            const left_target = (left_count + right_count) / 2;
            const separator_new = if (left_count < left_target)
                InternalNode.move_from_right(
                    left,
                    separator,
                    right,
                    left_target - left_count,
                    length,
                )
            else
                InternalNode.move_from_left(
                    left,
                    separator,
                    right,
                    left_count - left_target,
                    length,
                );
            InternalNode.separator_set(parent, separator_index, separator_new);
            assert(InternalNode.count(left) > InternalNode.count_min);
            assert(InternalNode.count(right) > InternalNode.count_min);
            return false;
        }

        /// What a scan copied from one leaf.
        const LeafCopy = struct {
            entry_count: u32,
            /// Whether the copy reached the leaf's last entry.
            leaf_finished: bool,
            /// The leaf's upper fence, where the next leaf starts; null for the last leaf.
            high: ?Key,
        };

        /// A scanned leaf's parent as the scan saw it, the edge it took there, and the parent's
        /// own upper fence.
        const ScanParent = struct {
            node: Cursor,
            edge_index: u32,
            high: ?Key,
        };

        /// Copies the leaf after the one copied last, one step from their parent, or else the
        /// leaf that covers `start` through a descent. Leaves `parent` at the copied leaf's.
        fn scan_leaf(
            map: *const Self,
            start: KeyBound,
            parent: *?ScanParent,
            target: []Entry,
        ) LeafCopy {
            if (parent.*) |previous| {
                parent.* = null;
                // A parent that changed, or ran out of edges, leaves it to the descent.
                if (map.scan_next_attempt(previous, parent, target)) |stepped| {
                    if (stepped) |leaf_copy| return leaf_copy;
                } else |_| {}
            }
            var retries: Retries = .{};
            while (true) {
                return map.scan_attempt(start, parent, target) catch |err| {
                    retries.wait(err);
                    continue;
                };
            }
        }

        /// Copies the entries of the leaf that covers `start`, from `start` on.
        fn scan_attempt(
            map: *const Self,
            start: KeyBound,
            parent_out: *?ScanParent,
            target: []Entry,
        ) Retry!LeafCopy {
            const root = try map.root_read();
            var node = root.node;
            // A start that lacks the shared bits lies before or after every key. Before, the
            // scan takes the first leaf, along the left border: shrinking the bits latches
            // every node there, so the leaf's version vouches that its keys had them too.
            const bound: KeyBound = switch (start) {
                .unbounded => .unbounded,
                .included, .excluded => |key| switch (map.shared_order(root.shared_length, key)) {
                    .eq => start,
                    .lt => .unbounded,
                    .gt => {
                        try map.root_validate(node);
                        return .{ .entry_count = 0, .leaf_finished = true, .high = null };
                    },
                },
            };
            var high: ?Key = null;
            var parent: ?ScanParent = null;
            while (node.height > 0) {
                const page = map.internal(node.page);
                const edge_index = switch (bound) {
                    .unbounded => 0,
                    .included, .excluded => |key| try InternalNode.route(
                        .optimistic,
                        page,
                        node.version,
                        key,
                    ),
                };
                if (node.height == 1) {
                    parent = .{ .node = node, .edge_index = edge_index, .high = high };
                    // The scan reads most of this leaf, and the next ones that `target` has
                    // room for: their misses overlap rather than come one leaf at a time.
                    for (0..scan_leaves_ahead(target.len) + 1) |ahead| {
                        map.scan_prefetch(page, edge_index + @as(u32, @intCast(ahead)));
                    }
                }
                if (edge_index < InternalNode.count_racy(page)) {
                    high = InternalNode.key_racy(page, edge_index);
                }
                node = try map.child_read(node, InternalNode.edge_racy(page, edge_index));
            }
            const page = map.leaf(node.page);
            const first: u32 = switch (bound) {
                .unbounded => 0,
                .included => |key| included: {
                    const position = try LeafNode.search(.optimistic, page, node.version, key);
                    break :included position.index;
                },
                .excluded => |key| excluded: {
                    const position = try LeafNode.search(.optimistic, page, node.version, key);
                    break :excluded position.index + @intFromBool(position.found);
                },
            };
            const copied = LeafNode.copy_racy(page, first, target);
            try Latch.validate(&LeafNode.header(page).latch, node.version);
            parent_out.* = parent;
            return .{ .entry_count = copied.count, .leaf_finished = copied.finished, .high = high };
        }

        /// Copies the leaf after the one a scan through `previous` copied, one step from their
        /// parent, if the parent has not changed since and has an edge after it. Its keys all
        /// lie at or above the separator between the two, the previous leaf's upper fence.
        ///
        /// The parent's version vouches for its count and the edge before the leaf is touched,
        /// as a page the pool has never handed out holds no header yet. Both sit at addresses
        /// fixed before they are read, so an address dependency orders that check after them,
        /// with no fence. Checked again with the leaf's after the copy, behind one fence, the
        /// parent's version vouches that the leaf was still its child when the leaf's version
        /// was read: freeing or moving a leaf changes its parent.
        fn scan_next_attempt(
            map: *const Self,
            previous: ScanParent,
            parent_out: *?ScanParent,
            target: []Entry,
        ) Retry!?LeafCopy {
            assert(previous.node.height == 1);
            const page = map.internal(previous.node.page);
            const parent_word = &InternalNode.header(page).latch;
            // The parent comes from an earlier copy, which even a single thread may have
            // changed since. A shared map's check after the copy covers that too.
            if (concurrency == .single_threaded and
                !Latch.is_current(parent_word, previous.node.version)) return null;
            const edge_index = previous.edge_index + 1;
            const count_loaded = access.load(u16, &InternalNode.header(page).count);
            const separator_count = @min(count_loaded, InternalNode.capacity);
            if (edge_index > separator_count) return null;
            const high = if (edge_index < separator_count)
                InternalNode.key_racy(page, edge_index)
            else
                previous.high;
            const edge = InternalNode.edge_racy(page, edge_index);
            try Latch.validate_after(parent_word, previous.node.version, count_loaded ^ edge);
            const leaf_page = map.leaf(edge);
            const leaf_word = &LeafNode.header(leaf_page).latch;
            const version = Latch.version_read(leaf_word);
            map.scan_prefetch(page, edge_index + scan_leaves_ahead(target.len));
            const copied = LeafNode.copy_racy(leaf_page, 0, target);
            try Latch.validate_both(leaf_word, version, parent_word, previous.node.version);
            parent_out.* = .{
                .node = previous.node,
                .edge_index = edge_index,
                .high = previous.high,
            };
            return .{ .entry_count = copied.count, .leaf_finished = copied.finished, .high = high };
        }

        /// Leaves past the one being copied that a scan prefetches: as many as `target_length`
        /// entries likely fill, at least one, so that a copy finds the next leaf on its way.
        fn scan_leaves_ahead(target_length: usize) u32 {
            const leaf_entries_typical = LeafNode.capacity / 2;
            return @intCast(std.math.clamp(target_length / leaf_entries_typical, 1, 4));
        }

        /// Starts loading the whole leaf behind the parent's edge `edge_index`, if it has one.
        /// The edge is read without validation, so it is only prefetched, never followed.
        fn scan_prefetch(
            map: *const Self,
            parent: InternalNode.PagePointer,
            edge_index: u32,
        ) void {
            if (edge_index > InternalNode.count_racy(parent)) return;
            const edge = InternalNode.edge_racy(parent, edge_index);
            prefetch(map.leaf_unchecked(edge), LeafNode.page_size);
        }

        /// Checks every invariant of a quiescent map: key order, fences, prefixes and windows,
        /// the shared bits, node fill, uniform leaf depth, and that the pools account for
        /// every page.
        pub fn verify(map: *const Self) void {
            const Frame = struct {
                page: u32,
                height: u8,
                fences: InternalNode.Fences,
                edge_next: u32,
            };
            var path: [height_max + 1]Frame = undefined;
            const root = map.root_load();
            const length = map.shared_length();
            for (map.head.shared_words, 0..) |word, index| {
                assert(word & ~word_mask(length -| @as(u32, @intCast(index * 64))) == 0);
            }
            var depth: u32 = 0;
            var leaf_count: u32 = 0;
            var internal_count: u32 = 0;
            var entry_total: u64 = 0;
            var rightmost_seen: ?u32 = null;
            path[0] = .{
                .page = root.page,
                .height = root.height,
                .fences = .{ .low = null, .high = null },
                .edge_next = 0,
            };
            while (true) {
                const frame = &path[depth];
                if (frame.height == 0) {
                    const page = map.leaf(frame.page);
                    const prefix_length = InternalNode.prefix_of(frame.fences, length);
                    const fences = frame.fences;
                    LeafNode.verify(page, fences.low, fences.high, prefix_length, depth == 0);
                    for (0..LeafNode.count(page)) |index| {
                        const key = LeafNode.key_at(page, @intCast(index));
                        assert(map.shared_order(root.shared_length, key) == .eq);
                    }
                    assert(rightmost_seen == null);
                    if (LeafNode.rightmost(page)) rightmost_seen = frame.page;
                    assert(LeafNode.rightmost(page) == (frame.fences.high == null));
                    entry_total += LeafNode.count(page);
                    leaf_count += 1;
                } else if (frame.edge_next == 0) {
                    const page = map.internal(frame.page);
                    InternalNode.verify(page, frame.fences, length, depth == 0);
                    internal_count += 1;
                }
                if (frame.height > 0 and
                    frame.edge_next <= InternalNode.count(map.internal(frame.page)))
                {
                    const page = map.internal(frame.page);
                    path[depth + 1] = .{
                        .page = InternalNode.edge_at(page, frame.edge_next),
                        .height = frame.height - 1,
                        .fences = InternalNode.child_fences(page, frame.edge_next),
                        .edge_next = 0,
                    };
                    frame.edge_next += 1;
                    depth += 1;
                    assert(depth <= height_max);
                    continue;
                }
                if (depth == 0) break;
                depth -= 1;
            }
            assert(rightmost_seen.? == map.rightmost_page);
            assert(leaf_count == map.leaves.page_count_in_use());
            assert(internal_count == map.internals.page_count_in_use());
            if (root.shared_length == shared_length_unset) assert(entry_total == 0);
            if (concurrency == .single_threaded) assert(entry_total == map.entry_count);
        }
    };
}
