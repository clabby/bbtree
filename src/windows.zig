//! Window columns: the array of per-key windows that a node searches.
//!
//! A column has a slot for every key a node can hold, rounded up to whole vectors. Slots past
//! the node's count hold the largest window, so a search can count over every slot, with no
//! branch on the count: padding never compares less than a query. Equal counts can include
//! padding; callers clamp them to the count.

const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;

/// The smallest vector the columns are padded to: 16 bytes, the width that every SIMD
/// instruction set has.
pub const slot_alignment_bytes = 16;

pub fn slot_count(comptime Window: type, comptime capacity: u32) u32 {
    const lanes = slot_alignment_bytes / @sizeOf(Window);
    return std.mem.alignForward(u32, capacity, lanes);
}

/// How the windows of a node compare with a query's.
pub const Counts = struct {
    less: u32,
    equal: u32,
};

/// Counts the slots whose window is less than, and equal to, the query's. Reads the slots
/// volatile, so that a reader racing a writer reads what the hardware holds; the caller
/// validates the node before using the counts.
pub inline fn counts(
    comptime Window: type,
    comptime slots: u32,
    windows: *const [slots]Window,
    query: Window,
) Counts {
    comptime assert(slots % (slot_alignment_bytes / @sizeOf(Window)) == 0);
    if (comptime builtin.cpu.arch.isX86()) return counts_masked(Window, slots, windows, query);
    return counts_summed(Window, slots, windows, query);
}

/// x86: each vector compare yields a bit mask, and a population count adds it up.
inline fn counts_masked(
    comptime Window: type,
    comptime slots: u32,
    windows: *const [slots]Window,
    query: Window,
) Counts {
    var less: u32 = 0;
    var equal: u32 = 0;
    comptime var offset: u32 = 0;
    inline while (offset < slots) {
        const lanes = comptime chunk_lanes(Window, slots - offset);
        const Vector = @Vector(lanes, Window);
        const Mask = @Int(.unsigned, lanes);
        const source: *align(@alignOf(Window)) const volatile Vector =
            @ptrCast(windows[offset..][0..lanes]);
        const loaded = source.*;
        const query_vector: Vector = @splat(query);
        less += @popCount(@as(Mask, @bitCast(loaded < query_vector)));
        equal += @popCount(@as(Mask, @bitCast(loaded == query_vector)));
        offset += lanes;
    }
    return .{ .less = less, .equal = equal };
}

/// Elsewhere (NEON): a compare yields all-ones lanes, which accumulate in a vector, and one
/// horizontal add at the end sums them. Turning each compare into a bit mask costs NEON a
/// horizontal add per vector.
inline fn counts_summed(
    comptime Window: type,
    comptime slots: u32,
    windows: *const [slots]Window,
    query: Window,
) Counts {
    const lanes = slot_alignment_bytes / @sizeOf(Window);
    const Vector = @Vector(lanes, Window);
    const query_vector: Vector = @splat(query);
    const ones: Vector = @splat(1);
    const zeros: Vector = @splat(0);
    var less: Vector = zeros;
    var equal: Vector = zeros;
    comptime var offset: u32 = 0;
    inline while (offset < slots) : (offset += lanes) {
        const source: *align(@alignOf(Window)) const volatile Vector =
            @ptrCast(windows[offset..][0..lanes]);
        const loaded = source.*;
        less += @select(Window, loaded < query_vector, ones, zeros);
        equal += @select(Window, loaded == query_vector, ones, zeros);
    }
    // Each lane counts at most slots / lanes; the sums widen, as a column may hold more
    // windows than a lane's type counts.
    const Wide = @Vector(lanes, u32);
    const less_wide: Wide = @intCast(less);
    const equal_wide: Wide = @intCast(equal);
    return .{ .less = @reduce(.Add, less_wide), .equal = @reduce(.Add, equal_wide) };
}

/// The widest vector that fits the remaining slots, up to the CPU's preferred width.
fn chunk_lanes(comptime Window: type, comptime remaining: u32) u32 {
    const preferred = std.simd.suggestVectorLength(Window) orelse
        slot_alignment_bytes / @sizeOf(Window);
    var lanes: u32 = @max(preferred, slot_alignment_bytes / @sizeOf(Window));
    while (lanes > remaining) lanes /= 2;
    assert(lanes * @sizeOf(Window) >= slot_alignment_bytes);
    return lanes;
}

/// Opens slot `index` for a new window, shifting the windows after it one slot right.
pub fn insert(
    comptime Window: type,
    windows: []Window,
    count: u32,
    index: u32,
    window: Window,
) void {
    assert(index <= count);
    assert(count < windows.len);
    assert(windows[count] == std.math.maxInt(Window));
    if (index < count) @memmove(windows[index + 1 .. count + 1], windows[index..count]);
    windows[index] = window;
}

/// Closes slot `index`, shifting the windows after it one slot left.
pub fn remove(comptime Window: type, windows: []Window, count: u32, index: u32) void {
    assert(index < count);
    assert(count <= windows.len);
    @memmove(windows[index .. count - 1], windows[index + 1 .. count]);
    windows[count - 1] = std.math.maxInt(Window);
}

/// Sets every slot from `count` on to the largest window.
pub fn pad(comptime Window: type, windows: []Window, count: u32) void {
    assert(count <= windows.len);
    @memset(windows[count..], std.math.maxInt(Window));
}

const testing = std.testing;

fn expect_counts_match_scan(comptime Window: type, comptime slots: u32, random: std.Random) !void {
    var windows: [slots]Window align(16) = undefined;
    const count = random.uintAtMost(u32, slots);
    for (windows[0..count]) |*window| window.* = random.uintAtMost(Window, 8);
    std.mem.sort(Window, windows[0..count], {}, std.sort.asc(Window));
    pad(Window, &windows, count);

    const query = random.uintAtMost(Window, 9);
    var less: u32 = 0;
    var equal: u32 = 0;
    for (windows) |window| {
        less += @intFromBool(window < query);
        equal += @intFromBool(window == query);
    }
    const result = counts(Window, slots, &windows, query);
    try testing.expectEqual(less, result.less);
    try testing.expectEqual(equal, result.equal);
}

test "vector counts match a scalar scan" {
    var prng = std.Random.DefaultPrng.init(0xc0de);
    for (0..500) |_| {
        try expect_counts_match_scan(u16, 56, prng.random());
        try expect_counts_match_scan(u16, 72, prng.random());
        try expect_counts_match_scan(u32, 40, prng.random());
        try expect_counts_match_scan(u64, 32, prng.random());
        try expect_counts_match_scan(u64, 2, prng.random());
    }
}

test "insert and remove keep the padding" {
    var windows: [8]u16 = undefined;
    pad(u16, &windows, 0);
    insert(u16, &windows, 0, 0, 5);
    insert(u16, &windows, 1, 0, 3);
    insert(u16, &windows, 2, 2, 9);
    try testing.expectEqualSlices(u16, &.{ 3, 5, 9, 0xffff }, windows[0..4]);
    remove(u16, &windows, 3, 1);
    try testing.expectEqualSlices(u16, &.{ 3, 9, 0xffff, 0xffff }, windows[0..4]);
}
