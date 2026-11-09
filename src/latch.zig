//! Version latches: one word per page that lets readers read a page without writing to it.
//!
//! A writer takes a page by compare-and-swap from the version it read, changes the page in
//! place, and lets it go with a higher version. A reader reads the version, reads the page,
//! and checks that the version has not moved. Readers never write the word, so a page that
//! many threads read stays in every reader's cache (optimistic lock coupling, as in Leis et
//! al., "The ART of Practical Synchronization").
//!
//! The word is even while no writer holds the page and odd while one does. Locking and every
//! unlock that follows a change each add one, so the word never returns to a value a reader
//! may have seen, even across freeing and reusing the page.
//!
//! A racing reader may see a page half-written. It must neither crash nor act on what it
//! read before the version check succeeds: indexes derived from racing reads stay within
//! the page, and pointers read from a page are followed only after a successful check.

const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;

pub const Concurrency = enum {
    /// One thread at a time. Latch operations reduce to assertions.
    single_threaded,
    /// Threads share the map through a shared pointer.
    multi_threaded,
};

/// A latch word's value, read while no writer held the page.
pub const Version = u64;

/// Returned when an optimistic read or a lock attempt finds that a page changed. The
/// operation lets go of what it holds and starts over from the root.
pub const Restart = error{Restart};

pub fn Latch(comptime concurrency: Concurrency) type {
    return struct {
        /// The version of a page, once no writer holds it. Waits while one does: callers
        /// that hold a page themselves must use `version_try_read` instead.
        pub fn version_read(word: *const u64) Version {
            if (concurrency == .single_threaded) {
                assert(word.* % 2 == 0);
                return word.*;
            }
            var backoff: Backoff = .{};
            while (true) {
                const version = @atomicLoad(u64, word, .acquire);
                if (version % 2 == 0) return version;
                backoff.wait();
            }
        }

        /// The version of a page, or a restart if a writer holds it.
        pub fn version_try_read(word: *const u64) Restart!Version {
            if (concurrency == .single_threaded) return version_read(word);
            const version = @atomicLoad(u64, word, .acquire);
            if (version % 2 == 1) return error.Restart;
            return version;
        }

        /// Succeeds if the page is still at `version`: everything read from it since
        /// `version` was read is what the page held.
        pub fn validate(word: *const u64, version: Version) Restart!void {
            assert(version % 2 == 0);
            if (concurrency == .single_threaded) {
                assert(word.* == version);
                return;
            }
            fence_loads();
            if (@atomicLoad(u64, word, .monotonic) != version) return error.Restart;
        }

        /// `validate` for two pages at once, behind one fence.
        pub fn validate_both(
            word: *const u64,
            version: Version,
            other_word: *const u64,
            other_version: Version,
        ) Restart!void {
            assert(version % 2 == 0);
            assert(other_version % 2 == 0);
            if (concurrency == .single_threaded) {
                assert(word.* == version);
                assert(other_word.* == other_version);
                return;
            }
            fence_loads();
            if (@atomicLoad(u64, word, .monotonic) != version) return error.Restart;
            if (@atomicLoad(u64, other_word, .monotonic) != other_version) return error.Restart;
        }

        /// `validate` for a check that vouches only for loaded values that all feed
        /// `loaded`. The check's address depends on `loaded`, which orders it after those
        /// loads as `validate`'s fence orders it after every earlier one. The loads must be
        /// at addresses fixed before them: an address chosen by branches on other racing loads
        /// leaves those loads unordered.
        pub fn validate_after(word: *const u64, version: Version, loaded: usize) Restart!void {
            assert(version % 2 == 0);
            if (concurrency == .single_threaded) {
                assert(word.* == version);
                return;
            }
            const ordered: *const u64 = @ptrFromInt(@intFromPtr(word) + zero_after(loaded));
            if (@atomicLoad(u64, ordered, .monotonic) != version) return error.Restart;
        }

        /// Whether the page is still at `version`, for a position kept between operations,
        /// which even a single thread may have changed since.
        pub fn is_current(word: *const u64, version: Version) bool {
            if (concurrency == .single_threaded) return word.* == version;
            fence_loads();
            return @atomicLoad(u64, word, .monotonic) == version;
        }

        /// Takes the page if it is still at `version`. Never waits.
        pub fn try_lock(word: *u64, version: Version) Restart!void {
            assert(version % 2 == 0);
            if (concurrency == .single_threaded) {
                assert(word.* == version);
                word.* = version + 1;
                return;
            }
            if (@cmpxchgStrong(u64, word, version, version + 1, .acquire, .monotonic) != null) {
                return error.Restart;
            }
            fence_stores();
        }

        /// Takes a page that no other thread can reach: fresh from a page pool.
        pub fn lock_fresh(word: *u64) void {
            if (concurrency == .single_threaded) {
                assert(word.* % 2 == 0);
                word.* += 1;
                return;
            }
            const version = @atomicLoad(u64, word, .monotonic);
            assert(version % 2 == 0);
            @atomicStore(u64, word, version + 1, .monotonic);
            fence_stores();
        }

        /// Lets go of a page that the holder changed: readers that read it meanwhile restart.
        pub fn unlock(word: *u64) void {
            if (concurrency == .single_threaded) {
                assert(word.* % 2 == 1);
                word.* += 1;
                return;
            }
            const version = @atomicLoad(u64, word, .monotonic);
            assert(version % 2 == 1);
            @atomicStore(u64, word, version + 1, .release);
        }

        /// Lets go of a page that the holder did not change: readers keep what they read.
        pub fn unlock_unchanged(word: *u64) void {
            if (concurrency == .single_threaded) {
                assert(word.* % 2 == 1);
                word.* -= 1;
                return;
            }
            const version = @atomicLoad(u64, word, .monotonic);
            assert(version % 2 == 1);
            @atomicStore(u64, word, version - 1, .release);
        }

        pub fn is_locked(word: *const u64) bool {
            if (concurrency == .single_threaded) return word.* % 2 == 1;
            return @atomicLoad(u64, word, .monotonic) % 2 == 1;
        }
    };
}

/// Orders a reader's racing loads before the load that validates them, as a seqlock reader
/// does. Zig has no fence builtin, so each architecture spells its own. On Arm the loads to
/// order include ones a search chose by branches, and a branch orders no later load, so only
/// a barrier orders them all.
fn fence_loads() void {
    switch (builtin.cpu.arch) {
        .x86, .x86_64 => asm volatile ("" ::: .{ .memory = true }),
        .aarch64, .aarch64_be => asm volatile ("dmb ishld" ::: .{ .memory = true }),
        .riscv64, .riscv32 => asm volatile ("fence r, r" ::: .{ .memory = true }),
        else => @compileError("bbtree: multi_threaded maps need a load fence for this CPU"),
    }
}

/// Zero, computed from `value` in a way the compiler cannot see through, so that an address
/// offset by it depends on the loads that produced `value`. Arm and RISC-V order a load
/// after the loads its address depends on; x86 orders every load after earlier ones, and the
/// dependency only keeps the compiler from reordering them.
fn zero_after(value: usize) usize {
    return switch (builtin.cpu.arch) {
        .aarch64, .aarch64_be => asm ("eor %[zero], %[value], %[value]"
            : [zero] "=r" (-> usize),
            : [value] "r" (value),
        ),
        .riscv64, .riscv32 => asm ("xor %[zero], %[value], %[value]"
            : [zero] "=r" (-> usize),
            : [value] "r" (value),
        ),
        .x86, .x86_64 => asm ("xor %[zero], %[zero]"
            : [zero] "=r" (-> usize),
            : [value] "r" (value),
        ),
        else => @compileError("bbtree: multi_threaded maps need a load dependency for this CPU"),
    };
}

/// Orders the lock's store before the stores that change the page: a reader that sees any
/// of the changes also sees the page locked when it validates.
fn fence_stores() void {
    switch (builtin.cpu.arch) {
        .x86, .x86_64 => asm volatile ("" ::: .{ .memory = true }),
        .aarch64, .aarch64_be => asm volatile ("dmb ishst" ::: .{ .memory = true }),
        .riscv64, .riscv32 => asm volatile ("fence w, w" ::: .{ .memory = true }),
        else => @compileError("bbtree: multi_threaded maps need a store fence for this CPU"),
    }
}

/// Waiting for another thread: a short exponential spin, then yielding to the scheduler,
/// since the thread we wait for may be descheduled.
pub const Backoff = struct {
    step: u32 = 0,

    const spin_step_max = 6;
    const step_max = 16;

    pub fn wait(backoff: *Backoff) void {
        assert(backoff.step <= step_max);
        if (backoff.step <= spin_step_max) {
            const spin_count = @as(u32, 1) << @intCast(backoff.step);
            for (0..spin_count) |_| std.atomic.spinLoopHint();
        } else {
            std.Thread.yield() catch {};
        }
        backoff.step = @min(backoff.step + 1, step_max);
    }
};

const testing = std.testing;

test "latch versions only grow" {
    inline for (.{ Concurrency.single_threaded, Concurrency.multi_threaded }) |concurrency| {
        const L = Latch(concurrency);
        var word: u64 = 0;
        const first = L.version_read(&word);
        try L.validate(&word, first);
        try L.try_lock(&word, first);
        try testing.expect(L.is_locked(&word));
        L.unlock_unchanged(&word);
        try L.validate(&word, first);
        try L.try_lock(&word, first);
        L.unlock(&word);
        const second = L.version_read(&word);
        try testing.expect(second > first);
        if (concurrency == .multi_threaded) {
            try testing.expectError(error.Restart, L.validate(&word, first));
            try testing.expectError(error.Restart, L.try_lock(&word, first));
        }
        L.lock_fresh(&word);
        try testing.expect(L.is_locked(&word));
        L.unlock(&word);
        try testing.expect(L.version_read(&word) > second);
    }
}
