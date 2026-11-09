//! Differential tests: random operations on a map and on a sorted array, which must agree
//! after every operation, with the tree's invariants checked as it grows and shrinks.

const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;
const bbtree = @import("bbtree.zig");
const stress = @import("test_options").stress;

/// A sorted array of entries: slow, simple, and obviously right.
fn Model(comptime Map: type) type {
    const Key = Map.Codec.Key;
    const Value = @FieldType(Map.Entry, "value");
    return struct {
        entries: std.ArrayList(Map.Entry) = .empty,

        const Self = @This();

        fn position(model: *const Self, key: Key) struct { index: usize, found: bool } {
            var low: usize = 0;
            var high: usize = model.entries.items.len;
            while (low < high) {
                const middle = low + (high - low) / 2;
                switch (Map.Codec.order(model.entries.items[middle].key, key)) {
                    .lt => low = middle + 1,
                    .gt => high = middle,
                    .eq => return .{ .index = middle, .found = true },
                }
            }
            return .{ .index = low, .found = false };
        }

        fn get(model: *const Self, key: Key) ?Value {
            const found = model.position(key);
            return if (found.found) model.entries.items[found.index].value else null;
        }

        fn put(model: *Self, gpa: std.mem.Allocator, key: Key, value: Value) !?Value {
            const found = model.position(key);
            if (found.found) {
                const previous = model.entries.items[found.index].value;
                model.entries.items[found.index].value = value;
                return previous;
            }
            try model.entries.insert(gpa, found.index, .{ .key = key, .value = value });
            return null;
        }

        fn remove(model: *Self, key: Key) ?Value {
            const found = model.position(key);
            if (!found.found) return null;
            return model.entries.orderedRemove(found.index).value;
        }
    };
}

fn KeySource(comptime Key: type) type {
    return struct {
        /// Backing storage for slice keys: the map borrows key bytes for its whole life.
        arena: std.heap.ArenaAllocator,
        random: std.Random,
        /// Keys come from a small space, so operations often hit existing keys.
        space: u64,
        /// With phases, inserted keys share long prefixes at first and ever shorter ones
        /// each phase, while lookups, removals and scan bounds come from any phase, so keys
        /// that lack the shared bits arrive while the tree is deep. Zero means no phases.
        phase_length: u64 = 0,
        key_count: u64 = 0,

        const Self = @This();
        const level_count = 4;

        /// A key for an insert.
        fn key(source: *Self) !Key {
            if (source.phase_length == 0) return source.key_at(level_count - 1);
            source.key_count += 1;
            const current = @min(level_count - 1, source.key_count / source.phase_length);
            if (source.random.uintLessThan(u32, 4) == 0) {
                return source.key_at(source.random.uintAtMost(u32, current));
            }
            return source.key_at(@intCast(current));
        }

        /// A key for anything else.
        fn probe(source: *Self) !Key {
            if (source.phase_length == 0) return source.key();
            return source.key_at(source.random.uintLessThan(u32, level_count));
        }

        /// A key whose prefix shortens with `level`: phased keys at level 0 share the most.
        fn key_at(source: *Self, level: u32) !Key {
            assert(level < level_count);
            const ordinal = source.random.uintLessThan(u64, source.space);
            if (source.phase_length != 0) return source.key_phased(level, ordinal);
            switch (@typeInfo(Key)) {
                .int => {
                    const scattered = ordinal *% 0x9e37_79b9_7f4a_7c15;
                    const wide = (@as(u128, scattered) << 64) | scattered;
                    return @bitCast(@as(@Int(.unsigned, @bitSizeOf(Key)), @truncate(wide)));
                },
                .array => {
                    var bytes: Key = @splat(0);
                    std.mem.writeInt(
                        u64,
                        bytes[bytes.len - 8 ..][0..8],
                        ordinal *% 0x9e37_79b9_7f4a_7c15,
                        .big,
                    );
                    bytes[0] = @truncate(ordinal % 3);
                    return bytes;
                },
                .pointer => {
                    // Shared prefixes and varied lengths, as paths and URLs have.
                    const prefix = "https://example.com/path/";
                    const text = try std.fmt.allocPrint(source.arena.allocator(), "{s}{d}/{x}", .{
                        prefix[0 .. ordinal % prefix.len],
                        ordinal % 97,
                        ordinal,
                    });
                    return text;
                },
                else => @compileError("unsupported key type"),
            }
        }

        fn key_phased(source: *Self, level: u32, ordinal: u64) !Key {
            const scattered = ordinal *% 0x9e37_79b9_7f4a_7c15;
            switch (@typeInfo(Key)) {
                .int, .array => {
                    const bits = @bitSizeOf(Key);
                    comptime assert(bits <= 128);
                    const base: u128 = 0xa5c3_5a3c_96e1_1e69_d2b4_4b2d_87f0_0f78;
                    const free_bits: u7 = @intCast(bits * (level + 1) / level_count - 1);
                    const mask = (@as(u128, 2) << free_bits) -% 1;
                    const random = (@as(u128, scattered) << 64) | (scattered ^ 0x5555);
                    const Bits = @Int(.unsigned, bits);
                    const value: Bits = @truncate((base & ~mask) | (random & mask));
                    if (@typeInfo(Key) == .int) return @bitCast(value);
                    var bytes: Key = undefined;
                    std.mem.writeInt(Bits, &bytes, value, .big);
                    return bytes;
                },
                .pointer => {
                    const prefixes = [level_count][]const u8{
                        "tenant/0042/users/",
                        "tenant/0042/",
                        "tenant/",
                        "",
                    };
                    // Half sort after the longer prefixes: '~' follows every letter and digit.
                    const after = if (ordinal % 2 == 0) "~" else "";
                    return std.fmt.allocPrint(
                        source.arena.allocator(),
                        "{s}{s}{x}",
                        .{ prefixes[level], after, scattered >> 40 },
                    );
                },
                else => @compileError("unsupported key type"),
            }
        }
    };
}

fn expect_iteration_matches(
    comptime Map: type,
    map: *const Map,
    model: anytype,
    random: std.Random,
    source: anytype,
) !void {
    const Key = Map.Codec.Key;
    const items = model.entries.items;
    var start_bound: Map.KeyBound = .unbounded;
    var end_bound: Map.KeyBound = .unbounded;
    var expected_start: usize = 0;
    var expected_end: usize = items.len;
    if (random.boolean()) {
        const key: Key = try source.probe();
        const found = model.position(key);
        if (random.boolean()) {
            start_bound = .{ .included = key };
            expected_start = found.index;
        } else {
            start_bound = .{ .excluded = key };
            expected_start = found.index + @intFromBool(found.found);
        }
    }
    if (random.boolean()) {
        const key: Key = try source.probe();
        const found = model.position(key);
        if (random.boolean()) {
            end_bound = .{ .included = key };
            expected_end = found.index + @intFromBool(found.found);
        } else {
            end_bound = .{ .excluded = key };
            expected_end = found.index;
        }
    }
    var iterator = map.iterator(.{ .start = start_bound, .end = end_bound });
    var index = expected_start;
    while (iterator.next()) |entry| : (index += 1) {
        try testing.expect(index < expected_end);
        try testing.expectEqual(Map.Codec.order(items[index].key, entry.key), .eq);
        try testing.expectEqual(items[index].value, entry.value);
    }
    try testing.expectEqual(@max(expected_start, expected_end), @max(index, expected_end));

    // A bulk scan into a buffer of any size copies the same entries, cut short by the buffer.
    var buffer: [200]Map.Entry = undefined;
    const target = buffer[0..random.uintAtMost(usize, buffer.len)];
    const copied = map.scan(.{ .start = start_bound, .end = end_bound }, target);
    const expected_count = @min(target.len, expected_end -| expected_start);
    try testing.expectEqual(expected_count, copied);
    for (target[0..copied], items[expected_start..][0..copied]) |entry, item| {
        try testing.expectEqual(Map.Codec.order(item.key, entry.key), .eq);
        try testing.expectEqual(item.value, entry.value);
    }
}

fn run_differential(comptime Map: type, seed: u64, operation_count_base: u32, space: u64) !void {
    try run_differential_phased(Map, seed, operation_count_base, space, 0);
}

fn run_differential_phased(
    comptime Map: type,
    seed: u64,
    operation_count_base: u32,
    space: u64,
    phase_count: u32,
) !void {
    const operation_count = operation_count_base * stress;
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    var source: KeySource(Map.Codec.Key) = .{
        .arena = .init(gpa),
        .random = random,
        .space = space,
        // Inserts take about four in ten operations.
        .phase_length = if (phase_count == 0) 0 else operation_count * 4 / 10 / phase_count,
    };
    defer source.arena.deinit();
    var model: Model(Map) = .{};
    defer model.entries.deinit(gpa);

    var map: Map = undefined;
    try map.init(gpa, try Map.options_for(space));
    defer map.deinit(gpa);

    // Phases that favor inserts, then removals, so the tree grows and shrinks.
    for (0..operation_count) |operation| {
        try differential_step(Map, &map, &model, &source, operation);
        if (operation % 1_000 == 0) map.verify();
    }
    map.verify();
    for (model.entries.items) |entry| try testing.expectEqual(entry.value, map.remove(entry.key).?);
    map.verify();
    var iterator = map.iterator(.{});
    try testing.expectEqual(@as(?Map.Entry, null), iterator.next());
}

/// One random operation on both the map and the model, which must agree.
fn differential_step(
    comptime Map: type,
    map: *Map,
    model: *Model(Map),
    source: *KeySource(Map.Codec.Key),
    operation: usize,
) !void {
    const gpa = testing.allocator;
    const random = source.random;
    const growing = (operation / 2_000) % 2 == 0;
    const value: u64 = random.int(u64);
    switch (random.uintLessThan(u32, 10)) {
        0...3 => if (growing) {
            const key = try source.key();
            try testing.expectEqual(try model.put(gpa, key, value), try map.put(key, value));
        } else {
            const key = try source.probe();
            try testing.expectEqual(model.remove(key), map.remove(key));
        },
        4 => {
            const key = try source.key();
            const absent = model.get(key) == null;
            try testing.expectEqual(absent, try map.insert(key, value));
            if (absent) _ = try model.put(gpa, key, value);
        },
        5, 6 => {
            const key = try source.probe();
            try testing.expectEqual(model.remove(key), map.remove(key));
        },
        7 => {
            const key = try source.probe();
            const present = map.update(key, value, struct {
                fn add(delta: u64, slot: *u64) void {
                    slot.* +%= delta;
                }
            }.add);
            try testing.expectEqual(model.get(key) != null, present);
            if (present) _ = try model.put(gpa, key, model.get(key).? +% value);
        },
        8 => {
            const key = try source.probe();
            try testing.expectEqual(model.get(key), map.get(key));
        },
        else => if (operation % 50 == 0) {
            try expect_iteration_matches(Map, map, model, random, source);
        } else {
            const key = try source.probe();
            try testing.expectEqual(model.get(key) != null, map.contains(key));
        },
    }
}

test "differential: integer keys" {
    try run_differential(
        bbtree.Map(u64, u64, .{ .concurrency = .single_threaded }),
        1,
        40_000,
        20_000,
    );
    try run_differential(bbtree.Map(u64, u64, .{}), 2, 40_000, 20_000);
    try run_differential(
        bbtree.Map(i32, u64, .{ .concurrency = .single_threaded }),
        3,
        20_000,
        5_000,
    );
    try run_differential(bbtree.Map(u128, u64, .{}), 4, 20_000, 5_000);
}

test "differential: byte keys" {
    try run_differential(
        bbtree.Map([16]u8, u64, .{ .concurrency = .single_threaded }),
        5,
        30_000,
        10_000,
    );
    try run_differential(
        bbtree.Map([]const u8, u64, .{ .concurrency = .single_threaded }),
        6,
        30_000,
        10_000,
    );
    try run_differential(bbtree.Map([]const u8, u64, .{}), 7, 30_000, 10_000);
}

test "differential: small pages make deep trees" {
    const config: bbtree.Config = .{
        .concurrency = .single_threaded,
        .leaf_page_size = 256,
        .internal_page_size = 256,
    };
    try run_differential(bbtree.Map(u64, u64, config), 8, 300_000, 150_000);
    try run_differential(bbtree.Map([]const u8, u64, config), 9, 100_000, 50_000);
}

test "differential: keys that lack the shared bits arrive while the tree is deep" {
    const config: bbtree.Config = .{
        .concurrency = .single_threaded,
        .leaf_page_size = 256,
        .internal_page_size = 256,
    };
    try run_differential_phased(bbtree.Map(u64, u64, config), 10, 60_000, 30_000, 4);
    try run_differential_phased(bbtree.Map(i32, u64, config), 11, 40_000, 20_000, 4);
    try run_differential_phased(bbtree.Map(u128, u64, .{}), 12, 40_000, 20_000, 4);
    try run_differential_phased(bbtree.Map([16]u8, u64, config), 13, 40_000, 20_000, 4);
    try run_differential_phased(bbtree.Map([]const u8, u64, config), 14, 60_000, 30_000, 4);
    try run_differential_phased(bbtree.Map([]const u8, u64, .{}), 15, 40_000, 20_000, 4);
}

test "ascending and descending inserts" {
    const Map = bbtree.Map(u64, u64, .{ .concurrency = .single_threaded });
    const gpa = testing.allocator;
    var map: Map = undefined;
    try map.init(gpa, try Map.options_for(100_000));
    defer map.deinit(gpa);

    for (0..50_000) |key| try testing.expectEqual(@as(?u64, null), try map.put(key, key));
    map.verify();
    // Ascending inserts leave leaves at least three-quarters full.
    try testing.expect(
        map.leaves.page_count_in_use() <= 50_000 / (Map.LeafNode.capacity * 3 / 4) + 1,
    );
    var key: u64 = 100_000;
    while (key > 50_000) : (key -= 1) _ = try map.put(key, key);
    map.verify();
    try testing.expectEqual(@as(u64, 100_000), map.count());
    for (0..100_001) |probe| try testing.expectEqual(
        if (probe == 50_000) null else @as(?u64, probe),
        map.get(probe),
    );
}

test "a full map refuses inserts and stays intact" {
    const Map = bbtree.Map(u64, u64, .{ .concurrency = .single_threaded });
    const gpa = testing.allocator;
    var map: Map = undefined;
    try map.init(gpa, .{ .leaf_count_max = 3, .internal_count_max = 1 });
    defer map.deinit(gpa);

    var inserted: u64 = 0;
    while (map.put(inserted *% 0x9e37_79b9_7f4a_7c15, inserted)) |previous| : (inserted += 1) {
        try testing.expectEqual(@as(?u64, null), previous);
    } else |err| try testing.expectEqual(error.MapFull, err);
    try testing.expect(inserted >= 2 * Map.LeafNode.capacity);
    map.verify();
    try testing.expectEqual(inserted, map.count());
    for (0..inserted) |ordinal| try testing.expectEqual(
        @as(?u64, ordinal),
        map.get(ordinal *% 0x9e37_79b9_7f4a_7c15),
    );
}

test "batched lookups agree with single lookups" {
    inline for (.{
        bbtree.Concurrency.single_threaded,
        bbtree.Concurrency.multi_threaded,
    }) |concurrency| {
        const Map = bbtree.Map(u64, u64, .{ .concurrency = concurrency });
        const gpa = testing.allocator;
        var map: Map = undefined;
        try map.init(gpa, try Map.options_for(100_000));
        defer map.deinit(gpa);

        // Enough leaves that lookups run in lanes.
        for (0..60_000) |ordinal| _ = try map.put(ordinal *% 0x9e37_79b9_7f4a_7c15, ordinal);
        var prng = std.Random.DefaultPrng.init(0x1a2e);
        var keys: [100]u64 = undefined;
        var values: [100]?u64 = undefined;
        for (0..50) |_| {
            for (&keys) |*key| key.* = prng.random().uintLessThan(
                u64,
                120_000,
            ) *% 0x9e37_79b9_7f4a_7c15;
            map.get_many(&keys, &values);
            for (keys, values) |key, value| try testing.expectEqual(map.get(key), value);
        }
    }
}

test "pool sizes beyond 2^32 pages are refused" {
    const Map = bbtree.Map(u64, u64, .{});
    try testing.expectError(error.CapacityTooLarge, Map.options_for(std.math.maxInt(u64)));
    const options = try Map.options_for(1_000);
    try testing.expect(options.leaf_count_max > 1_000 / Map.LeafNode.count_min);
}

test "get_ptr updates in place" {
    const Map = bbtree.Map([4]u8, u32, .{ .concurrency = .single_threaded });
    const gpa = testing.allocator;
    var map: Map = undefined;
    try map.init(gpa, try Map.options_for(16));
    defer map.deinit(gpa);
    _ = try map.put("abcd".*, 1);
    map.get_ptr("abcd".*).?.* += 41;
    try testing.expectEqual(@as(?u32, 42), map.get("abcd".*));
    try testing.expectEqual(@as(?*u32, null), map.get_ptr("abce".*));
}

test "values aligned past a cache line keep their alignment" {
    const Value = struct { word: u64 align(256) };
    const Map = bbtree.Map(u64, Value, .{ .concurrency = .single_threaded });
    const gpa = testing.allocator;
    var map: Map = undefined;
    try map.init(gpa, try Map.options_for(200));
    defer map.deinit(gpa);
    for (0..200) |key| _ = try map.put(key, .{ .word = key });
    map.verify();
    for (0..200) |key| {
        const slot = map.get_ptr(key).?;
        try testing.expect(std.mem.isAligned(@intFromPtr(slot), 256));
        try testing.expectEqual(@as(u64, key), slot.word);
    }
}

test "partial copies resume after their last key, and iterators survive mutation" {
    const Map = bbtree.Map(u64, u64, .{
        .concurrency = .single_threaded,
        .leaf_page_size = 256,
        .internal_page_size = 256,
    });
    var map: Map = undefined;
    try map.init(testing.allocator, try Map.options_for(20_000));
    defer map.deinit(testing.allocator);
    for (0..4_000) |ordinal| _ = try map.put(ordinal * 2, ordinal * 2);

    // Every buffer size, so that copies end at every position within a leaf.
    var buffer: [100]Map.Entry = undefined;
    for (1..buffer.len) |capacity| {
        const range: Map.Range = .{ .start = .{ .excluded = 101 }, .end = .{ .included = 7_501 } };
        var scan: Map.Scan = .init(range);
        var expected: u64 = 102;
        while (!scan.done) {
            const copied = scan.copy(&map, buffer[0..capacity]);
            for (buffer[0..copied]) |entry| {
                try testing.expectEqual(expected, entry.key);
                expected += 2;
            }
        }
        try testing.expectEqual(@as(u64, 7_502), expected);
    }

    // Odd keys come and go between calls; the even keys must all come out, in order.
    var iterator = map.iterator(.{});
    var previous: ?u64 = null;
    var even_count: u64 = 0;
    var call_count: u64 = 0;
    while (iterator.next()) |entry| : (call_count += 1) {
        if (previous) |key| try testing.expect(key < entry.key);
        previous = entry.key;
        if (entry.key % 2 == 0) {
            try testing.expectEqual(even_count * 2, entry.key);
            even_count += 1;
        }
        for (0..16) |step| {
            const key = ((call_count * 17 + step * 31) % 4_000) * 2 + 1;
            if (call_count % 2 == 0) _ = try map.put(key, key) else _ = map.remove(key);
        }
    }
    try testing.expectEqual(@as(u64, 4_000), even_count);
    map.verify();
}
