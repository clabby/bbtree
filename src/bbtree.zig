//! bbtree: an ordered map that threads share, after Xiangpeng Hao's bbtree.
//!
//! ```zig
//! const Map = bbtree.Map(u64, u64, .{});
//! var map: Map = undefined;
//! try map.init(gpa, try Map.options_for(1_000_000));
//! defer map.deinit(gpa);
//!
//! _ = try map.put(1, 10);
//! assert(map.get(1).? == 10);
//! ```
//!
//! The map is a B+-tree of 1 KiB pages. Nodes search a few bytes of each key, its window,
//! with SIMD compares, and read full keys only on ties. Readers share the tree through
//! optimistic lock coupling and never write to it. Every page comes from pools sized at
//! initialization, which is also what makes reads without epochs safe: a page is never
//! returned to the allocator while the map lives.

const map = @import("map.zig");

pub const codec = @import("codec.zig");
pub const Map = map.Map;
pub const Config = map.Config;
pub const Options = map.Options;
pub const Bound = map.Bound;
pub const Concurrency = @import("latch.zig").Concurrency;

test {
    _ = @import("codec.zig");
    _ = @import("latch.zig");
    _ = @import("pool.zig");
    _ = @import("windows.zig");
    _ = @import("leaf.zig");
    _ = @import("internal.zig");
    _ = @import("map_test.zig");
    _ = @import("stress_test.zig");
}
