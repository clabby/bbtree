//! Concurrency tests: threads share one map, and each checks what it can know for sure.
//!
//! - Owned keys: each thread writes only its own keys, so it knows their values exactly, and
//!   every read of an own key, every range scan's own keys included, must match its model,
//!   while other threads split and merge the same pages.
//! - Stable keys: keys inserted before the threads start and never removed must be found,
//!   with their values, by every lookup and every scan, however the pages around them change.
//! - Counters: concurrent increments under `update`, interleaved with inserts and removals
//!   that move the counters between leaves, must not lose a single increment.
//! - Shrinking shared bits: keys that lack the bits every key shares arrive while threads
//!   look up, append and scan, and every stable key stays where it was.
//!
//! Small pages make structure changes frequent. Each test ends by checking the tree's
//! invariants with the threads joined.

const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;
const bbtree = @import("bbtree.zig");
const stress = @import("test_options").stress;

const thread_count = 8;
const small_pages: bbtree.Config = .{ .leaf_page_size = 256, .internal_page_size = 256 };

/// A bijection on 64-bit words, so that dense ordinals spread over the key space while the
/// owner of a key stays recoverable.
const Scramble = struct {
    const multiplier: u64 = 0x9e37_79b9_7f4a_7c15;
    const inverse: u64 = 0xf1de_83e1_9937_733d;

    comptime {
        assert(multiplier *% inverse == 1);
    }

    fn key(owner_index: u64, ordinal: u64) u64 {
        return ((ordinal << 8) | owner_index) *% multiplier;
    }

    fn owner(key_value: u64) u64 {
        return (key_value *% inverse) & 0xff;
    }
};

fn OwnedKeys(comptime Map: type) type {
    return struct {
        const ordinal_count = 3_000;

        fn run(map: *Map, owner: u64, seed: u64, operation_count: u32) !void {
            var prng = std.Random.DefaultPrng.init(seed);
            const random = prng.random();
            var model: [ordinal_count]?u64 = @splat(null);
            for (0..operation_count) |operation| {
                const ordinal = random.uintLessThan(u64, ordinal_count);
                const key = Scramble.key(owner, ordinal);
                const value = random.int(u64);
                switch (random.uintLessThan(u32, 8)) {
                    0, 1, 2 => {
                        try testing.expectEqual(model[ordinal], try map.put(key, value));
                        model[ordinal] = value;
                    },
                    3, 4 => {
                        try testing.expectEqual(model[ordinal], map.remove(key));
                        model[ordinal] = null;
                    },
                    5 => _ = map.get(Scramble.key(random.uintLessThan(u64, thread_count), ordinal)),
                    6 => try testing.expectEqual(model[ordinal], map.get(key)),
                    else => if (operation % 16 == 0) try expect_scan_matches(
                        map,
                        owner,
                        &model,
                        random,
                    ),
                }
            }
            for (model, 0..) |expected, ordinal| {
                try testing.expectEqual(expected, map.get(Scramble.key(owner, ordinal)));
            }
        }

        fn expect_scan_matches(
            map: *const Map,
            owner: u64,
            model: *const [ordinal_count]?u64,
            random: std.Random,
        ) !void {
            const start = random.int(u64);
            const end = start +| random.uintAtMost(u64, std.math.maxInt(u64) / 8);
            var iterator = map.iterator(.{
                .start = .{ .included = start },
                .end = .{ .excluded = end },
            });
            var previous: ?u64 = null;
            var seen_count: u32 = 0;
            while (iterator.next()) |entry| {
                if (previous) |key| try testing.expect(key < entry.key);
                previous = entry.key;
                try testing.expect(entry.key >= start);
                try testing.expect(entry.key < end);
                if (Scramble.owner(entry.key) != owner) continue;
                const ordinal = (entry.key *% Scramble.inverse) >> 8;
                try testing.expectEqual(model[ordinal], entry.value);
                seen_count += 1;
            }
            var expected_count: u32 = 0;
            for (model, 0..) |value, ordinal| {
                const key = Scramble.key(owner, ordinal);
                if (value != null and key >= start and key < end) expected_count += 1;
            }
            try testing.expectEqual(expected_count, seen_count);
        }
    };
}

fn run_threads(comptime function: anytype, arguments: anytype) !void {
    var threads: [thread_count]std.Thread = undefined;
    var errors: [thread_count]?anyerror = @splat(null);
    const Runner = struct {
        fn run(slot: *?anyerror, owner: u64, captured: @TypeOf(arguments)) void {
            @call(.auto, function, .{owner} ++ captured) catch |err| {
                slot.* = err;
            };
        }
    };
    for (&threads, &errors, 0..) |*thread, *slot, owner| {
        thread.* = try std.Thread.spawn(.{}, Runner.run, .{ slot, owner, arguments });
    }
    for (threads) |thread| thread.join();
    for (errors) |slot| if (slot) |err| return err;
}

test "threads that own their keys always read their own writes" {
    const Map = bbtree.Map(u64, u64, small_pages);
    var map: Map = undefined;
    const entry_count_max = thread_count * OwnedKeys(Map).ordinal_count;
    try map.init(testing.allocator, try Map.options_for(entry_count_max));
    defer map.deinit(testing.allocator);

    const Worker = struct {
        fn run(owner: u64, shared: *Map) !void {
            try OwnedKeys(Map).run(shared, owner, 0x51 + owner, 30_000 * stress);
        }
    };
    try run_threads(Worker.run, .{&map});
    map.verify();
}

test "stable byte keys stay visible while their neighbors churn" {
    const Map = bbtree.Map([]const u8, u64, small_pages);
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();

    const stable_count = 2_000;
    const churn_count = 4_000;
    const keys = try arena.allocator().alloc([]const u8, stable_count + churn_count * thread_count);
    for (keys, 0..) |*key, index| {
        key.* = try std.fmt.allocPrint(
            arena.allocator(),
            "/srv/data/{d}/item-{d}",
            .{ index % 13, index },
        );
    }
    var map: Map = undefined;
    try map.init(gpa, try Map.options_for(keys.len));
    defer map.deinit(gpa);
    for (keys[0..stable_count], 0..) |key, index| _ = try map.put(key, index);

    const Worker = struct {
        fn run(owner: u64, shared: *Map, all_keys: []const []const u8) !void {
            var prng = std.Random.DefaultPrng.init(0xab + owner);
            const random = prng.random();
            const own = all_keys[stable_count + owner * churn_count ..][0..churn_count];
            for (0..20_000 * stress) |operation| {
                const churn = own[random.uintLessThan(usize, churn_count)];
                if (random.boolean()) _ = try shared.put(churn, 0) else _ = shared.remove(churn);
                const stable = random.uintLessThan(usize, stable_count);
                try testing.expectEqual(@as(?u64, stable), shared.get(all_keys[stable]));
                if (operation % 16 == 0) try expect_stable_batch(shared, all_keys, random);
                if (operation % 64 == 0) try expect_stable_in_scan(shared, all_keys, random);
            }
        }

        fn expect_stable_batch(
            shared: *const Map,
            all_keys: []const []const u8,
            random: std.Random,
        ) !void {
            var indexes: [24]usize = undefined;
            var batch: [24][]const u8 = undefined;
            var values: [24]?u64 = undefined;
            for (&indexes, &batch) |*index, *key| {
                index.* = random.uintLessThan(usize, stable_count);
                key.* = all_keys[index.*];
            }
            shared.get_many(&batch, &values);
            for (indexes, values) |index, value| try testing.expectEqual(@as(?u64, index), value);
        }

        fn expect_stable_in_scan(
            shared: *const Map,
            all_keys: []const []const u8,
            random: std.Random,
        ) !void {
            const first = random.uintLessThan(usize, stable_count);
            var iterator = shared.iterator(.{ .start = .{ .included = all_keys[first] } });
            var previous: []const u8 = "";
            var stable_seen: u32 = 0;
            for (0..200) |_| {
                const entry = iterator.next() orelse break;
                try testing.expect(std.mem.order(u8, previous, entry.key) == .lt);
                previous = entry.key;
                if (entry.value != 0) stable_seen += 1;
            }
            try testing.expect(stable_seen > 0);
        }
    };
    try run_threads(Worker.run, .{ &map, @as([]const []const u8, keys) });
    map.verify();
    for (
        keys[0..stable_count],
        0..,
    ) |key, index| try testing.expectEqual(@as(?u64, index), map.get(key));
}

test "concurrent increments survive structure changes" {
    const Map = bbtree.Map(u64, u64, small_pages);
    const counter_count = 64;
    const counter_spacing = 1_000_000;
    const churn_per_counter = 2_000;
    const increment_count = 20_000 * stress;
    var map: Map = undefined;
    try map.init(testing.allocator, try Map.options_for(counter_count * (churn_per_counter + 1)));
    defer map.deinit(testing.allocator);
    for (0..counter_count) |counter| _ = try map.put(counter * counter_spacing, 0);

    const Worker = struct {
        fn run(owner: u64, shared: *Map) !void {
            var prng = std.Random.DefaultPrng.init(0xc0 + owner);
            const random = prng.random();
            for (0..increment_count) |_| {
                const counter = random.uintLessThan(u64, counter_count) * counter_spacing;
                try testing.expectEqual(true, shared.update(counter, @as(u64, 1), increment));
                // Churn between the counters moves them between leaves.
                const gap = random.uintLessThan(u64, counter_count) * counter_spacing;
                const churn = gap + 1 + random.uintLessThan(u64, churn_per_counter);
                if (random.uintLessThan(u32, 3) != 0) {
                    _ = try shared.put(churn, owner);
                } else {
                    _ = shared.remove(churn);
                }
            }
        }

        fn increment(amount: u64, slot: *u64) void {
            slot.* += amount;
        }
    };
    try run_threads(Worker.run, .{&map});
    map.verify();
    var total: u64 = 0;
    for (0..counter_count) |counter| total += map.get(counter * 1_000_000).?;
    try testing.expectEqual(@as(u64, thread_count * increment_count), total);
}

test "lookups, appends and scans stay right while the shared bits shrink" {
    const Map = bbtree.Map([]const u8, u64, small_pages);
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();

    const keys = try ShrinkKeys.init(arena.allocator());
    const entry_count_max = ShrinkKeys.stable_count + ShrinkKeys.append_count + 16;
    for (0..8 * stress) |_| {
        var map: Map = undefined;
        try map.init(gpa, try Map.options_for(entry_count_max));
        defer map.deinit(gpa);
        for (keys.stable, 0..) |key, index| _ = try map.put(key, index);
        try run_threads(ShrinkKeys.run, .{ &map, &keys });
        map.verify();
        for (keys.stable, 0..) |key, index| try testing.expectEqual(@as(?u64, index), map.get(key));
        for (ShrinkKeys.outsiders) |key| {
            try testing.expectEqual(@as(?u64, ShrinkKeys.outsider_value), map.get(key));
        }
    }
}

/// Stable keys behind a long shared prefix, keys appended after them, and outsiders that lack
/// ever more of the prefix, on both sides of the stable keys.
const ShrinkKeys = struct {
    stable: []const []const u8,
    appended: []const []const u8,

    const Map = bbtree.Map([]const u8, u64, small_pages);
    const stable_count = 1_500;
    const append_count = 4_000;
    const operation_count = 4_000;
    const outsider_value: u64 = 1 << 40;
    const append_value: u64 = 1 << 32;
    const outsiders = [_][]const u8{
        "tenant/0042/users/~",
        "tenant/0042/a",
        "tenant/0042/~",
        "tenant/0",
        "tenant/~",
        "t",
        "a",
        "~",
        "",
    };
    /// Scans start here, before every stable key, and end before the first outsider above
    /// them: neither bound has the stable keys' prefix.
    const scan_start = "tenant/0042/a";
    const scan_end = "tenant/0042/users/~";

    fn init(allocator: std.mem.Allocator) !ShrinkKeys {
        const stable = try allocator.alloc([]const u8, stable_count);
        for (stable, 0..) |*key, index| {
            key.* = try std.fmt.allocPrint(allocator, "tenant/0042/users/{d:0>5}", .{index});
        }
        const appended = try allocator.alloc([]const u8, append_count);
        for (appended, 0..) |*key, index| {
            key.* = try std.fmt.allocPrint(allocator, "tenant/0042/users/zz{d:0>6}", .{index});
        }
        return .{ .stable = stable, .appended = appended };
    }

    /// Thread 0 inserts the outsiders, spread over its operations; thread 1 appends; the
    /// others read. Every thread checks the stable keys.
    fn run(owner: u64, map: *Map, keys: *const ShrinkKeys) !void {
        var prng = std.Random.DefaultPrng.init(0x5f + owner);
        const random = prng.random();
        const spacing = operation_count / outsiders.len;
        for (0..operation_count) |operation| {
            if (owner == 0 and operation % spacing == 0 and operation / spacing < outsiders.len) {
                _ = try map.put(outsiders[operation / spacing], outsider_value);
            }
            if (owner == 1) _ = try map.put(keys.appended[operation], append_value + operation);
            const index = random.uintLessThan(usize, stable_count);
            try testing.expectEqual(@as(?u64, index), map.get(keys.stable[index]));
            try testing.expectEqual(@as(?u64, null), map.get("tenant/0042/b"));
            if (map.get(outsiders[random.uintLessThan(usize, outsiders.len)])) |value| {
                try testing.expectEqual(outsider_value, value);
            }
            if (operation % 128 == owner) try expect_scans(map, keys);
        }
    }

    fn expect_scans(map: *const Map, keys: *const ShrinkKeys) !void {
        var iterator = map.iterator(.{
            .start = .{ .included = scan_start },
            .end = .{ .excluded = scan_end },
        });
        var previous: []const u8 = "";
        var stable_seen: u64 = 0;
        while (iterator.next()) |entry| {
            try testing.expect(std.mem.order(u8, previous, entry.key) == .lt);
            try testing.expect(std.mem.order(u8, scan_start, entry.key) != .gt);
            try testing.expect(std.mem.order(u8, entry.key, scan_end) == .lt);
            previous = entry.key;
            if (entry.value < stable_count) {
                try testing.expectEqual(stable_seen, entry.value);
                try testing.expectEqualStrings(keys.stable[stable_seen], entry.key);
                stable_seen += 1;
            }
        }
        try testing.expectEqual(@as(u64, stable_count), stable_seen);

        // From after every stable key, a scan finds none of them.
        var after = map.iterator(.{ .start = .{ .excluded = "tenant/0042/~" } });
        while (after.next()) |entry| {
            try testing.expect(std.mem.order(u8, "tenant/0042/~", entry.key) == .lt);
            try testing.expect(entry.value >= stable_count);
        }
    }
};

test "scans with small buffers see every stable key while leaves split and merge" {
    const Map = bbtree.Map(u64, u64, small_pages);
    var map: Map = undefined;
    try map.init(testing.allocator, try Map.options_for(16_000));
    defer map.deinit(testing.allocator);
    // Even keys are stable; writers churn the odd keys between them.
    for (0..4_000) |ordinal| _ = try map.put(ordinal * 2, ordinal * 2);

    const Writer = struct {
        fn run(shared: *Map, stop: *const std.atomic.Value(bool), seed: u64) !void {
            var prng = std.Random.DefaultPrng.init(seed);
            var iteration: u64 = 0;
            while (!stop.load(.monotonic)) : (iteration += 1) {
                const key = prng.random().uintLessThan(u64, 4_000) * 2 + 1;
                if (iteration % 4 < 2) _ = try shared.put(key, key) else _ = shared.remove(key);
            }
        }
    };
    var stop = std.atomic.Value(bool).init(false);
    var writers: [4]std.Thread = undefined;
    for (&writers, 0..) |*writer, index| {
        writer.* = try std.Thread.spawn(.{}, Writer.run, .{ &map, &stop, 0x5ca + index });
    }
    defer {
        stop.store(true, .monotonic);
        for (writers) |writer| writer.join();
    }

    var buffer: [100]Map.Entry = undefined;
    for (0..300 * stress) |round| {
        const range: Map.Range = .{ .start = .{ .included = 100 }, .end = .{ .excluded = 7_900 } };
        var scan: Map.Scan = .init(range);
        var previous: ?u64 = null;
        var expected: u64 = 100;
        while (!scan.done) {
            const copied = scan.copy(&map, buffer[0 .. round % buffer.len + 1]);
            for (buffer[0..copied]) |entry| {
                try testing.expect(entry.key >= 100);
                try testing.expect(entry.key < 7_900);
                if (previous) |key| try testing.expect(key < entry.key);
                previous = entry.key;
                try testing.expectEqual(entry.key, entry.value);
                if (entry.key % 2 == 0) {
                    try testing.expectEqual(expected, entry.key);
                    expected += 2;
                }
            }
        }
        try testing.expectEqual(@as(u64, 7_900), expected);
    }
}
