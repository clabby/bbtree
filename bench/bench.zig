//! The Zig side of the comparison with Rust's ordered maps. bench/rust builds this program,
//! runs it once per workload, and puts its cells beside the Rust maps' in one table per
//! workload. Both sides run the same workloads key for key: keys are scrambled ordinals, and
//! random orders come from SplitMix64 with the same seeds. The maps run natively here, with
//! nothing between the timer and their calls.
//!
//!     bench u64 <entries>
//!     bench bytes <entries>
//!     bench threads <entries> <operations per thread>
//!
//! Each line of output is one cell: map, column and value, separated by tabs. Single-thread
//! cells are nanoseconds per operation, thread cells millions of operations per second.

const std = @import("std");
const assert = std.debug.assert;
const bbtree = @import("bbtree");

const Shared = bbtree.Map(u64, u64, .{});
const Single = bbtree.Map(u64, u64, .{ .concurrency = .single_threaded });
const Bytes = bbtree.Map([]const u8, u64, .{});
const BytesSingle = bbtree.Map([]const u8, u64, .{ .concurrency = .single_threaded });

/// Odd, so multiplying by it permutes u64: ordinals give distinct keys that look random.
const scramble: u64 = 0x9e37_79b9_7f4a_7c15;
const batch_size = 64;
const scan_length = 100;
const thread_counts = [_]u32{ 1, 4, 8, 16 };
const mix_thread_count = 16;

const pages: std.mem.Allocator = .{ .ptr = &recycler, .vtable = &Recycler.vtable };
var recycler: Recycler = .{};

/// Hands each map memory the process has touched before: fresh memory is written before a
/// map gets it, and up to eight freed pools go to the next maps that ask for as much. The
/// Zig maps' timings then leave out first-touch page faults, which the Rust side's allocator
/// avoids for nodes it recycles from maps it dropped.
const Recycler = struct {
    freed: [8]?[]u8 = @splat(null),

    const vtable: std.mem.Allocator.VTable = .{
        .alloc = alloc,
        .resize = std.mem.Allocator.noResize,
        .remap = std.mem.Allocator.noRemap,
        .free = free,
    };

    fn alloc(
        context: *anyopaque,
        length: usize,
        alignment: std.mem.Alignment,
        return_address: usize,
    ) ?[*]u8 {
        const self: *Recycler = @ptrCast(@alignCast(context));
        for (&self.freed) |*slot| {
            const memory = slot.* orelse continue;
            if (memory.len != length or !alignment.check(@intFromPtr(memory.ptr))) continue;
            slot.* = null;
            return memory.ptr;
        }
        const fresh = std.heap.page_allocator.rawAlloc(length, alignment, return_address) orelse
            return null;
        @memset(fresh[0..length], 0);
        return fresh;
    }

    fn free(
        context: *anyopaque,
        memory: []u8,
        alignment: std.mem.Alignment,
        return_address: usize,
    ) void {
        const self: *Recycler = @ptrCast(@alignCast(context));
        for (&self.freed) |*slot| {
            if (slot.* != null) continue;
            slot.* = memory;
            return;
        }
        std.heap.page_allocator.rawFree(memory, alignment, return_address);
    }
};

fn key(ordinal: u64) u64 {
    return ordinal *% scramble;
}

/// SplitMix64, as bench/rust draws it.
const Rng = struct {
    state: u64,

    fn next(rng: *Rng) u64 {
        rng.state +%= scramble;
        var z = rng.state;
        z = (z ^ (z >> 30)) *% 0xbf58_476d_1ce4_e5b9;
        z = (z ^ (z >> 27)) *% 0x94d0_49bb_1331_11eb;
        return z ^ (z >> 31);
    }

    /// Uniform below `bound`, without a division.
    fn below(rng: *Rng, bound: u64) u64 {
        assert(bound > 0);
        return @intCast((@as(u128, rng.next()) * bound) >> 64);
    }
};

/// The ordinals below `count` in a random order: Fisher-Yates, as bench/rust shuffles.
fn permutation(gpa: std.mem.Allocator, count: u64, seed: u64) ![]u64 {
    const ordinals = try gpa.alloc(u64, count);
    for (ordinals, 0..) |*ordinal, index| ordinal.* = index;
    var rng: Rng = .{ .state = seed };
    var index = count;
    while (index > 1) {
        index -= 1;
        std.mem.swap(u64, &ordinals[index], &ordinals[rng.below(index + 1)]);
    }
    return ordinals;
}

const Report = struct {
    writer: *std.Io.Writer,

    fn cell(report: Report, map: []const u8, column: []const u8, value: f64) !void {
        try report.writer.print("{s}\t{s}\t{d:.2}\n", .{ map, column, value });
    }
};

const Clock = struct {
    io: std.Io,
    start: std.Io.Timestamp,

    fn begin(io: std.Io) Clock {
        return .{ .io = io, .start = std.Io.Timestamp.now(io, .awake) };
    }

    fn elapsed_ns(clock: Clock) f64 {
        const duration = clock.start.durationTo(std.Io.Timestamp.now(clock.io, .awake));
        return @floatFromInt(duration.nanoseconds);
    }

    fn per_operation(clock: Clock, operation_count: usize) f64 {
        assert(operation_count > 0);
        return clock.elapsed_ns() / @as(f64, @floatFromInt(operation_count));
    }
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    if (argv.len < 3) return error.Usage;
    const workload = argv[1];
    const entry_count = try std.fmt.parseInt(u64, argv[2], 10);
    assert(entry_count > 0);

    var buffer: [4096]u8 = undefined;
    var stdout: std.Io.File.Writer = .init(.stdout(), init.io, &buffer);
    const report: Report = .{ .writer = &stdout.interface };
    warm_up(init.io);
    if (std.mem.eql(u8, workload, "u64")) {
        const order = try permutation(arena, entry_count, 1);
        try single_thread(Shared, "bbtree-zig", report, init.io, order);
        try single_thread(Single, "bbtree-zig (single-threaded)", report, init.io, order);
    } else if (std.mem.eql(u8, workload, "bytes")) {
        try bytes(report, init.io, arena, entry_count);
    } else if (std.mem.eql(u8, workload, "threads")) {
        if (argv.len < 4) return error.Usage;
        const operation_count = try std.fmt.parseInt(u64, argv[3], 10);
        try threads(report, init.io, entry_count, operation_count);
    } else return error.Usage;
    try stdout.interface.flush();
}

/// Spins for 300 ms before the first timed cell, so that the scheduler and the clock settle on
/// a busy core: a process that just started may run on an efficiency core, or slowly.
fn warm_up(io: std.Io) void {
    const clock: Clock = .begin(io);
    var state: u64 = 1;
    while (clock.elapsed_ns() < 300 * std.time.ns_per_ms) {
        for (0..10_000) |_| state = state *% scramble +% 1;
    }
    std.mem.doNotOptimizeAway(state);
}

/// Every single-thread cell of one map, in the order of bench/rust's u64 table.
fn single_thread(
    comptime Map: type,
    name: []const u8,
    report: Report,
    io: std.Io,
    order: []const u64,
) !void {
    var map: Map = undefined;
    try map.init(pages, try Map.options_for(order.len));
    const insert = try time_inserts(Map, &map, io, order.len, key);
    try report.cell(name, "insert random", insert);
    try report.cell(name, "get hit", time_lookups(Map, &map, io, order, 0));
    try report.cell(name, "get miss", time_lookups(Map, &map, io, order, order.len));
    try report.cell(name, "batched get hit (per key)", time_batches(Map, &map, io, order));
    // A row shows the scan beside the removal, as in bench/rust; the scan runs first.
    const scan = time_scans(Map, &map, io, order[0 .. order.len / 10]);
    try report.cell(name, "remove random", time_removes(Map, &map, io, order));
    try report.cell(name, "scan 100 (per scan)", scan);
    map.deinit(pages);

    try map.init(pages, try Map.options_for(order.len));
    defer map.deinit(pages);
    const ascending = try time_inserts(Map, &map, io, order.len, ordinal_itself);
    try report.cell(name, "insert ascending", ascending);
}

fn ordinal_itself(ordinal: u64) u64 {
    return ordinal;
}

fn time_inserts(
    comptime Map: type,
    map: *Map,
    io: std.Io,
    count: usize,
    comptime key_of: fn (u64) u64,
) !f64 {
    const clock: Clock = .begin(io);
    for (0..count) |ordinal| {
        const previous = try map.put(key_of(ordinal), ordinal);
        std.mem.doNotOptimizeAway(previous);
    }
    return clock.per_operation(count);
}

/// Looks up the keys of `order`, each shifted by `offset`: zero for hits, the entry count
/// for misses.
fn time_lookups(
    comptime Map: type,
    map: *const Map,
    io: std.Io,
    order: []const u64,
    offset: u64,
) f64 {
    var hit_count: usize = 0;
    const clock: Clock = .begin(io);
    for (order) |ordinal| {
        const value = map.get(key(ordinal + offset));
        std.mem.doNotOptimizeAway(value);
        hit_count += @intFromBool(value != null);
    }
    const result = clock.per_operation(order.len);
    assert(hit_count == if (offset == 0) order.len else 0);
    return result;
}

fn time_batches(comptime Map: type, map: *const Map, io: std.Io, order: []const u64) f64 {
    const batch_count = order.len / batch_size;
    var keys: [batch_size]u64 = undefined;
    var values: [batch_size]?u64 = undefined;
    var hit_count: usize = 0;
    const clock: Clock = .begin(io);
    for (0..batch_count) |batch| {
        for (&keys, order[batch * batch_size ..][0..batch_size]) |*batch_key, ordinal| {
            batch_key.* = key(ordinal);
        }
        map.get_many(&keys, &values);
        for (values) |value| hit_count += @intFromBool(value != null);
    }
    const result = clock.per_operation(batch_count * batch_size);
    assert(hit_count == batch_count * batch_size);
    return result;
}

fn time_scans(comptime Map: type, map: *const Map, io: std.Io, starts: []const u64) f64 {
    var sum: u64 = 0;
    var buffer: [scan_length]Map.Entry = undefined;
    const clock: Clock = .begin(io);
    for (starts) |ordinal| {
        const count = map.scan(.{ .start = .{ .included = key(ordinal) } }, &buffer);
        for (buffer[0..count]) |entry| sum +%= entry.value;
    }
    const result = clock.per_operation(starts.len);
    std.mem.doNotOptimizeAway(sum);
    return result;
}

fn time_removes(comptime Map: type, map: *Map, io: std.Io, order: []const u64) f64 {
    var removed_count: usize = 0;
    const clock: Clock = .begin(io);
    for (order) |ordinal| {
        const value = map.remove(key(ordinal));
        std.mem.doNotOptimizeAway(value);
        removed_count += @intFromBool(value != null);
    }
    const result = clock.per_operation(order.len);
    assert(removed_count == order.len);
    return result;
}

/// A key like `user/0517/9e3779b97f4a7c15012`, as bench/rust makes it: 26 to 40 bytes behind a
/// shared `user/` prefix, distinct by the 16 hex digits of the scrambled ordinal.
fn byte_key(arena: std.mem.Allocator, ordinal: u64) ![]const u8 {
    const scrambled = key(ordinal);
    const padding = "0123456789abcd"[0 .. scrambled % 15];
    return std.fmt.allocPrint(arena, "user/{d:0>4}/{x:0>16}{s}", .{
        scrambled >> 54,
        scrambled,
        padding,
    });
}

fn bytes(report: Report, io: std.Io, arena: std.mem.Allocator, entry_count: u64) !void {
    const keys = try arena.alloc([]const u8, entry_count);
    const misses = try arena.alloc([]const u8, entry_count);
    for (keys, misses, 0..) |*hit_key, *miss_key, ordinal| {
        hit_key.* = try byte_key(arena, ordinal);
        miss_key.* = try byte_key(arena, entry_count + ordinal);
    }
    const order = try permutation(arena, entry_count, 2);
    try bytes_cells(Bytes, "bbtree-zig", report, io, keys, misses, order);
    try bytes_cells(BytesSingle, "bbtree-zig (single-threaded)", report, io, keys, misses, order);
}

fn bytes_cells(
    comptime Map: type,
    name: []const u8,
    report: Report,
    io: std.Io,
    keys: []const []const u8,
    misses: []const []const u8,
    order: []const u64,
) !void {
    var map: Map = undefined;
    try map.init(pages, try Map.options_for(keys.len));
    defer map.deinit(pages);
    var clock: Clock = .begin(io);
    for (keys, 0..) |hit_key, value| std.mem.doNotOptimizeAway(try map.put(hit_key, value));
    try report.cell(name, "insert random", clock.per_operation(keys.len));

    var hit_count: usize = 0;
    clock = .begin(io);
    for (order) |ordinal| hit_count += @intFromBool(map.get(keys[ordinal]) != null);
    try report.cell(name, "get hit", clock.per_operation(order.len));
    assert(hit_count == order.len);

    hit_count = 0;
    clock = .begin(io);
    for (misses) |miss_key| hit_count += @intFromBool(map.get(miss_key) != null);
    try report.cell(name, "get miss", clock.per_operation(misses.len));
    assert(hit_count == 0);
}

fn threads(report: Report, io: std.Io, entry_count: u64, operation_count: u64) !void {
    var map: Shared = undefined;
    try map.init(pages, try Shared.options_for(entry_count + mix_thread_count * operation_count));
    defer map.deinit(pages);
    for (0..entry_count) |ordinal| _ = try map.put(key(ordinal), ordinal);

    var column_buffer: [32]u8 = undefined;
    for (thread_counts) |thread_count| {
        const column = try std.fmt.bufPrint(&column_buffer, "get ×{d}", .{thread_count});
        const throughput =
            try time_threads(&map, io, entry_count, operation_count, thread_count, false);
        try report.cell("bbtree-zig", column, throughput);
    }
    const mix = try time_threads(&map, io, entry_count, operation_count, mix_thread_count, true);
    try report.cell("bbtree-zig", "90% get / 10% insert ×16", mix);
}

/// Millions of operations per second, with each thread running `operation_count` random lookups
/// of the prebuilt keys, or with `mixed`, one in ten of them inserts of new keys.
fn time_threads(
    map: *Shared,
    io: std.Io,
    entry_count: u64,
    operation_count: u64,
    thread_count: u32,
    mixed: bool,
) !f64 {
    assert(thread_count <= mix_thread_count);
    var start: Start = .{};
    var workers: [mix_thread_count]std.Thread = undefined;
    for (workers[0..thread_count], 0..) |*worker, index| {
        worker.* = try std.Thread.spawn(.{}, Start.work, .{
            &start, map, index, entry_count, operation_count, mixed,
        });
    }
    while (start.ready.load(.acquire) < thread_count) std.atomic.spinLoopHint();
    const clock: Clock = .begin(io);
    start.go.store(true, .release);
    for (workers[0..thread_count]) |worker| worker.join();
    const seconds = clock.elapsed_ns() / 1e9;
    return @as(f64, @floatFromInt(thread_count * operation_count)) / seconds / 1e6;
}

/// Holds every worker at the line until all have started, as bench/rust's barrier does.
const Start = struct {
    ready: std.atomic.Value(u32) = .init(0),
    go: std.atomic.Value(bool) = .init(false),

    fn work(
        start: *Start,
        map: *Shared,
        index: usize,
        entry_count: u64,
        operation_count: u64,
        mixed: bool,
    ) void {
        var rng: Rng = .{ .state = index + 1 };
        var insert_next = entry_count + index * operation_count;
        var get_count: u64 = 0;
        var hit_count: u64 = 0;
        _ = start.ready.fetchAdd(1, .acq_rel);
        while (!start.go.load(.acquire)) std.atomic.spinLoopHint();
        for (0..operation_count) |_| {
            if (mixed and rng.next() % 10 == 0) {
                _ = map.put(key(insert_next), insert_next) catch unreachable;
                insert_next += 1;
            } else {
                get_count += 1;
                hit_count += @intFromBool(map.get(key(rng.below(entry_count))) != null);
            }
        }
        assert(hit_count == get_count);
    }
};
