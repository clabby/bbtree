//! Page pools: every page a map uses, allocated once when the map is initialized.
//!
//! A map never returns a page to the allocator while it lives. A page it no longer needs
//! goes on its pool's free list, and keeps its kind, its size and its latch word. So a reader
//! that reaches a page through a stale index still reads a page of the kind it expects, and
//! its version check fails: the page was locked, and so moved past the version the reader
//! saw, before it was freed. This is what lets readers go without epochs or hazard pointers.
//!
//! Pages are named by 32-bit indexes, which halve the size of an internal node's edges.
//!
//! Each page is followed by an unused cache line, so that pages start an odd number of lines
//! apart. Every search starts with a page's first line, its header and the start of its window
//! column; spaced this way, those lines spread over every set of each cache, rather than
//! competing for the few sets that addresses a power of two apart share.
//!
//! The free list is a stack behind a spin lock. Pages come and go only when nodes split and
//! merge, so the lock is rarely contended, and it rules out the ABA problem of a lock-free
//! stack, whose tag a stalled pop can see wrap around. Its holder never waits for anything
//! else, so it cannot take part in a deadlock with page latches. A free page links to the
//! next one in the four bytes after its header, where a page in use starts its window column:
//! the header stays as the page's last owner left it.

const std = @import("std");
const assert = std.debug.assert;
const latch = @import("latch.zig");
const page_header = @import("page.zig");
const Concurrency = latch.Concurrency;
const Header = page_header.Header;

/// Pages of `page_size` bytes, each aligned to `page_alignment`: a cache line, or more for
/// entries that need it.
pub fn Pool(
    comptime page_size: u32,
    comptime page_alignment: u32,
    comptime concurrency: Concurrency,
) type {
    comptime assert(page_alignment >= page_header.alignment);
    comptime assert(std.math.isPowerOfTwo(page_alignment));
    comptime assert(page_size % page_alignment == 0);
    comptime assert(page_size >= @sizeOf(Header) + @sizeOf(u32));

    return struct {
        slots: []Slot,
        /// Pages handed out at least once. Pages past it have never been written.
        page_count_used: std.atomic.Value(u32) align(std.atomic.cache_line),
        /// Guards `free_first` and the links of the pages on the free list.
        free_lock: std.atomic.Value(bool) align(std.atomic.cache_line),
        /// The free list's first page plus one, or zero when it is empty.
        free_first: u32,

        pub const Page = [page_size]u8;
        /// A page and the cache line that staggers the next one.
        pub const Slot = extern struct {
            page: Page align(page_alignment),
            padding: [page_alignment]u8,
        };
        const Self = @This();
        const Latch = latch.Latch(concurrency);

        pub fn init(pool: *Self, gpa: std.mem.Allocator, page_count: u32) error{OutOfMemory}!void {
            assert(page_count > 0);
            pool.* = .{
                .slots = try gpa.alloc(Slot, page_count),
                .page_count_used = .init(0),
                .free_lock = .init(false),
                .free_first = 0,
            };
            assert(pool.slots.len == page_count);
        }

        pub fn deinit(pool: *Self, gpa: std.mem.Allocator) void {
            gpa.free(pool.slots);
            pool.* = undefined;
        }

        pub fn page_count_max(pool: *const Self) u32 {
            return @intCast(pool.slots.len);
        }

        pub fn page(pool: *const Self, index: u32) *align(page_alignment) Page {
            assert(index < pool.slots.len);
            return &pool.slots[index].page;
        }

        pub fn header(pool: *const Self, index: u32) *Header {
            return @ptrCast(pool.page(index));
        }

        /// The never-used page that `acquire` hands out next once the free list is empty, if
        /// any is left. A hint, read without the lock.
        pub fn fresh_next(pool: *const Self) ?u32 {
            const used = pool.page_count_used.load(.monotonic);
            return if (used < pool.slots.len) used else null;
        }

        /// A page for the caller alone, latched, its contents undefined; null when every page
        /// is in use.
        pub fn acquire(pool: *Self) ?u32 {
            if (pool.free_pop()) |index| {
                Latch.lock_fresh(&pool.header(index).latch);
                return index;
            }
            var used = pool.page_count_used.load(.monotonic);
            while (used < pool.slots.len) {
                used = pool.page_count_used.cmpxchgWeak(
                    used,
                    used + 1,
                    .monotonic,
                    .monotonic,
                ) orelse {
                    const fresh = pool.header(used);
                    fresh.* = .empty;
                    Latch.lock_fresh(&fresh.latch);
                    return used;
                };
            }
            return null;
        }

        /// Returns a page that nothing in the tree leads to any more. The caller let go of it
        /// after its last change, so its version is past any a reader holds.
        pub fn release(pool: *Self, index: u32) void {
            assert(index < pool.page_count_used.load(.monotonic));
            assert(!Latch.is_locked(&pool.header(index).latch));
            pool.free_lock_acquire();
            defer pool.free_lock_release();
            pool.free_next(index).* = pool.free_first;
            pool.free_first = index + 1;
        }

        fn free_pop(pool: *Self) ?u32 {
            pool.free_lock_acquire();
            defer pool.free_lock_release();
            if (pool.free_first == 0) return null;
            const first = pool.free_first - 1;
            pool.free_first = pool.free_next(first).*;
            assert(pool.free_first <= pool.slots.len);
            return first;
        }

        fn free_lock_acquire(pool: *Self) void {
            if (concurrency == .single_threaded) {
                assert(!pool.free_lock.raw);
                pool.free_lock.raw = true;
                return;
            }
            var backoff: latch.Backoff = .{};
            while (pool.free_lock.cmpxchgWeak(false, true, .acquire, .monotonic) != null) {
                backoff.wait();
            }
        }

        fn free_lock_release(pool: *Self) void {
            if (concurrency == .single_threaded) {
                assert(pool.free_lock.raw);
                pool.free_lock.raw = false;
                return;
            }
            assert(pool.free_lock.load(.monotonic));
            pool.free_lock.store(false, .release);
        }

        fn free_next(pool: *const Self, index: u32) *u32 {
            return @ptrCast(@alignCast(pool.page(index)[@sizeOf(Header)..][0..@sizeOf(u32)]));
        }

        /// Pages in use: handed out and not on the free list. For quiescent checks only.
        pub fn page_count_in_use(pool: *const Self) u32 {
            var free_count: u32 = 0;
            var cursor: u32 = pool.free_first;
            while (cursor != 0) : (free_count += 1) {
                assert(free_count < pool.slots.len);
                cursor = pool.free_next(cursor - 1).*;
            }
            const used = pool.page_count_used.load(.acquire);
            assert(free_count <= used);
            return used - free_count;
        }
    };
}

const testing = std.testing;

test "pool hands out each page once and reuses released pages" {
    const L = latch.Latch(.multi_threaded);
    var pool: Pool(256, page_header.alignment, .multi_threaded) = undefined;
    try pool.init(testing.allocator, 3);
    defer pool.deinit(testing.allocator);

    var pages: [3]u32 = undefined;
    for (&pages) |*index| index.* = pool.acquire().?;
    try testing.expectEqual(@as(?u32, null), pool.acquire());
    try testing.expectEqual(@as(u32, 3), pool.page_count_in_use());

    const version_before = pool.header(pages[1]).latch;
    L.unlock(&pool.header(pages[1]).latch);
    pool.release(pages[1]);
    try testing.expectEqual(@as(u32, 2), pool.page_count_in_use());

    const reused = pool.acquire().?;
    try testing.expectEqual(pages[1], reused);
    try testing.expect(pool.header(reused).latch > version_before);
    try testing.expect(L.is_locked(&pool.header(reused).latch));
}

test "pool survives concurrent acquire and release" {
    const Concurrent = Pool(256, page_header.alignment, .multi_threaded);
    const L = latch.Latch(.multi_threaded);
    var pool: Concurrent = undefined;
    try pool.init(testing.allocator, 64);
    defer pool.deinit(testing.allocator);

    const Worker = struct {
        fn run(shared: *Concurrent) void {
            var held: [8]u32 = undefined;
            for (0..2_000) |_| {
                var count: usize = 0;
                while (count < held.len) : (count += 1) {
                    held[count] = shared.acquire() orelse break;
                    // A page belongs to one holder: its latch word is ours to bump.
                    assert(L.is_locked(&shared.header(held[count]).latch));
                }
                for (held[0..count]) |index| {
                    L.unlock(&shared.header(index).latch);
                    shared.release(index);
                }
            }
        }
    };
    var threads: [4]std.Thread = undefined;
    for (&threads) |*thread| thread.* = try std.Thread.spawn(.{}, Worker.run, .{&pool});
    for (threads) |thread| thread.join();
    try testing.expectEqual(@as(u32, 0), pool.page_count_in_use());
}
