//! What every page shares: a header at offset zero, and racing reads of page contents.

const std = @import("std");
const assert = std.debug.assert;
const Concurrency = @import("latch.zig").Concurrency;

/// What a page's address is aligned to: a cache line, so that no line straddles two pages.
pub const alignment = std.atomic.cache_line;

/// The first 16 bytes of every leaf and internal page.
pub const Header = extern struct {
    /// The page's latch word (`latch.zig`). Page pools never reset it.
    latch: u64,
    /// Keys in the page.
    count: u16,
    /// Leading encoding bits that every key routed to the page shares, which its windows skip.
    prefix_length: u16,
    /// Leaves: one if the leaf is the tree's last, which covers every key from its first on.
    rightmost: u8,
    reserved: [3]u8,

    pub const empty: Header = .{
        .latch = 0,
        .count = 0,
        .prefix_length = 0,
        .rightmost = 0,
        .reserved = @splat(0),
    };
};

comptime {
    assert(@sizeOf(Header) == 16);
    assert(@offsetOf(Header, "latch") == 0);
}

/// How a node operation reads its page: holding the page's latch, or racing writers and
/// validating the page's version before trusting what it read.
pub const Mode = enum { exclusive, optimistic };

pub fn Access(comptime concurrency: Concurrency) type {
    return struct {
        /// Reads a header field that writers may be changing. Writers store these fields
        /// with `store`, so a racing reader sees one value or the other, never a mix.
        pub fn load(comptime T: type, pointer: *const T) T {
            if (concurrency == .single_threaded) return pointer.*;
            return @atomicLoad(T, pointer, .monotonic);
        }

        pub fn store(comptime T: type, pointer: *T, value: T) void {
            if (concurrency == .single_threaded) {
                pointer.* = value;
            } else {
                @atomicStore(T, pointer, value, .monotonic);
            }
        }

        /// Reads page contents that writers may be changing with plain stores. Volatile, so the
        /// result is whatever the hardware saw; the caller validates the page's version before
        /// trusting it.
        pub fn racy(comptime T: type, pointer: *const T) T {
            if (concurrency == .single_threaded) return pointer.*;
            const volatile_pointer: *const volatile T = pointer;
            return volatile_pointer.*;
        }

        /// Copies page contents that writers may be changing, as `racy` reads them, in the
        /// widest words the type's size allows: a racy read of a whole struct can go through
        /// a temporary on the stack. Single-threaded maps copy the same way, as a loop of
        /// 16-byte loads beats a call to memcpy for a leaf's few hundred bytes.
        pub fn copy(comptime T: type, target: []T, source: []const T) void {
            assert(target.len == source.len);
            const Word = comptime word: {
                if (@sizeOf(T) % 16 == 0 and @alignOf(T) >= 8) break :word @Vector(2, u64);
                for (.{ u64, u32, u16 }) |Word| {
                    if (@sizeOf(T) % @sizeOf(Word) == 0 and @alignOf(T) >= @alignOf(Word)) {
                        break :word Word;
                    }
                }
                break :word u8;
            };
            const word_alignment = @min(@alignOf(T), @alignOf(Word));
            const word_count = source.len * @sizeOf(T) / @sizeOf(Word);
            const from: [*]align(word_alignment) const volatile Word = @ptrCast(source.ptr);
            const to: [*]align(word_alignment) Word = @ptrCast(target.ptr);
            // Four words a round: volatile loads stay one per word, so the loop unrolls by
            // hand to keep them in flight together.
            var index: usize = 0;
            while (index + 4 <= word_count) : (index += 4) {
                const words = [4]Word{
                    from[index],
                    from[index + 1],
                    from[index + 2],
                    from[index + 3],
                };
                for (words, 0..) |word, offset| to[index + offset] = word;
            }
            while (index < word_count) : (index += 1) to[index] = from[index];
        }
    };
}
