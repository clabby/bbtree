//! Key codecs: how a map reads the order-preserving byte encoding of its keys.
//!
//! Nodes do not search their full keys. Each node keeps a window per key: a few bytes of
//! the key's encoding, taken right after the bits that every key routed to the node shares
//! (the node's prefix). Comparing windows decides most searches with a handful of vector
//! instructions over one or two cache lines; only keys whose windows tie with the query
//! are compared in full. Prefixes count bits, not bytes, so that a window starts where the
//! node's keys start to differ and spreads them over all of its values.
//!
//! An encoding is a string of bytes that sorts as its key does. Codecs read it as a string
//! of bits followed by zeros without end, so that every window of a key has 64 bits.
//!
//! A codec is a namespace (a struct type with no fields) that declares:
//!
//! - `Key`: the key type.
//! - `window_size_leaf`, `window_size_internal`: bytes of each window that leaves and
//!   internal nodes keep: 2, 4 or 8.
//! - `indirect`: whether comparing two keys reads memory outside the node, as byte slices
//!   do. Optimistic readers validate a node before following such a key.
//! - `size_max: ?u32`: the most bytes an encoding has, or null without a bound.
//! - `window(key, offset) u64`: bits `offset..offset + 64` of the encoding. Windows of sorted
//!   keys whose encodings share their first `offset` bits are sorted.
//! - `order(a, b)` and `order_from(a, b, offset)`: the key order; the second may assume that
//!   the encodings share their first `offset` bits.
//! - `prefix_length(low, high) u32`: how many leading bits the encodings of `low` and
//!   `high` share, where a null `low` stands for an encoding of zeros and a null `high` for
//!   one of ones. Every key between two fences shares this many bits with both.

const std = @import("std");
const assert = std.debug.assert;
const Order = std.math.Order;

/// The codec for a key type that bbtree recognizes: integers whose width is a whole number
/// of bytes, byte arrays, and byte slices.
pub fn Auto(comptime Key: type) type {
    if (Key == []const u8) return ByteSlice;
    switch (@typeInfo(Key)) {
        .int => return Integer(Key),
        .array => |array| if (array.child == u8) return ByteArray(array.len),
        else => {},
    }
    @compileError("bbtree: no codec for " ++ @typeName(Key) ++ "; set Config.codec");
}

/// Integers encode as their big-endian bytes, with the sign bit flipped for signed types so
/// that negative numbers sort first.
pub fn Integer(comptime Int: type) type {
    const info = @typeInfo(Int).int;
    comptime assert(info.bits % 8 == 0);
    comptime assert(info.bits >= 8);
    comptime assert(info.bits <= 128);

    return struct {
        pub const Key = Int;
        pub const window_size_leaf = 2;
        pub const window_size_internal = 2;
        pub const indirect = false;
        pub const size_max: ?u32 = info.bits / 8;

        const Unsigned = @Int(.unsigned, info.bits);
        const sign_mask: Unsigned = if (info.signedness == .signed) 1 << (info.bits - 1) else 0;

        fn encode(key: Int) Unsigned {
            const bits: Unsigned = @bitCast(key);
            return bits ^ sign_mask;
        }

        pub fn window(key: Int, offset: u32) u64 {
            if (offset >= info.bits) return 0;
            const shifted = encode(key) << @intCast(offset);
            if (info.bits >= 64) return @truncate(shifted >> (info.bits - 64));
            return @as(u64, shifted) << (64 - info.bits);
        }

        pub fn order(a: Int, b: Int) Order {
            return std.math.order(a, b);
        }

        pub fn order_from(a: Int, b: Int, offset: u32) Order {
            _ = offset;
            return std.math.order(a, b);
        }

        pub fn prefix_length(low: ?Int, high: ?Int) u32 {
            const low_encoded: Unsigned = if (low) |key| encode(key) else 0;
            const high_encoded: Unsigned =
                if (high) |key| encode(key) else std.math.maxInt(Unsigned);
            const length: u32 = @clz(low_encoded ^ high_encoded);
            assert(length <= info.bits);
            return length;
        }
    };
}

/// Byte arrays encode as themselves. They live inside the node, so ties compare without
/// leaving it, and two-byte windows suffice.
pub fn ByteArray(comptime length: u32) type {
    comptime assert(length > 0);

    return struct {
        pub const Key = [length]u8;
        pub const window_size_leaf = 2;
        pub const window_size_internal = 2;
        pub const indirect = false;
        pub const size_max: ?u32 = length;

        pub fn window(key: Key, offset: u32) u64 {
            return bytes_window(&key, offset);
        }

        pub fn order(a: Key, b: Key) Order {
            return bytes_order(&a, &b);
        }

        pub fn order_from(a: Key, b: Key, offset: u32) Order {
            const start = @min(offset / 8, length);
            return bytes_order(a[start..], b[start..]);
        }

        pub fn prefix_length(low: ?Key, high: ?Key) u32 {
            return bytes_fences_prefix_length(
                if (low) |*key| key else null,
                if (high) |*key| key else null,
            );
        }
    };
}

/// Byte slices encode as their bytes. The map stores the slice, not the bytes: the bytes
/// must stay valid and unchanged until the map is deinitialized, even after their key is
/// removed, because separators in internal nodes are copies of keys. Ties read the bytes from
/// wherever they lie, so slices keep wider windows than keys stored in the node: four bytes
/// in leaves, whose column then fits two lines, and eight in internal nodes, where keys that
/// share a field, as a shard number, would otherwise tie on every level.
pub const ByteSlice = struct {
    pub const Key = []const u8;
    pub const window_size_leaf = 4;
    pub const window_size_internal = 8;
    pub const indirect = true;
    pub const size_max: ?u32 = null;

    pub fn window(key: Key, offset: u32) u64 {
        return bytes_window(key, offset);
    }

    pub fn order(a: Key, b: Key) Order {
        return bytes_order(a, b);
    }

    pub fn order_from(a: Key, b: Key, offset: u32) Order {
        const start = @min(offset / 8, a.len, b.len);
        return bytes_order(a[start..], b[start..]);
    }

    pub fn prefix_length(low: ?Key, high: ?Key) u32 {
        return bytes_fences_prefix_length(low, high);
    }
};

/// Bits `offset..offset + 64` of `bytes`, followed by zeros.
fn bytes_window(bytes: []const u8, offset: u32) u64 {
    const byte_offset = offset / 8;
    const word = bytes_word(bytes, byte_offset);
    const shift: u6 = @intCast(offset % 8);
    if (shift == 0) return word;
    const next: u64 = if (byte_offset + 8 < bytes.len) bytes[byte_offset + 8] else 0;
    return (word << shift) | (next >> @intCast(@as(u7, 8) - shift));
}

/// Bytes `offset..offset + 8` of `bytes`, big-endian, followed by zeros.
fn bytes_word(bytes: []const u8, offset: usize) u64 {
    if (offset >= bytes.len) return 0;
    const rest = bytes[offset..];
    if (rest.len >= 8) return std.mem.readInt(u64, rest[0..8], .big);

    var buffer: [8]u8 = @splat(0);
    @memcpy(buffer[0..rest.len], rest);
    return std.mem.readInt(u64, &buffer, .big);
}

/// Lexicographic order, eight bytes at a time: keys are usually short, and a call into
/// memcmp costs more than the comparison.
fn bytes_order(a: []const u8, b: []const u8) Order {
    const common = @min(a.len, b.len);
    var index: usize = 0;
    while (index + 8 <= common) : (index += 8) {
        const a_word = std.mem.readInt(u64, a[index..][0..8], .big);
        const b_word = std.mem.readInt(u64, b[index..][0..8], .big);
        if (a_word != b_word) return std.math.order(a_word, b_word);
    }
    while (index < common) : (index += 1) {
        if (a[index] != b[index]) return std.math.order(a[index], b[index]);
    }
    return std.math.order(a.len, b.len);
}

/// `prefix_length` for byte encodings: an empty string stands for a missing lower fence, as
/// its zeros sort first. A missing upper fence stands for ones without end.
fn bytes_fences_prefix_length(low: ?[]const u8, high: ?[]const u8) u32 {
    const high_bytes = high orelse return bytes_leading_ones(low orelse return 0);
    return bytes_prefix_length(low orelse &.{}, high_bytes);
}

/// Leading bits that the encodings of `a` and `b` share, counting the zeros that follow
/// the shorter one. Encodings that differ only in their trailing zeros share every bit.
fn bytes_prefix_length(a: []const u8, b: []const u8) u32 {
    const length = @max(a.len, b.len);
    var offset: usize = 0;
    while (offset < length) : (offset += 8) {
        const difference = bytes_word(a, offset) ^ bytes_word(b, offset);
        if (difference != 0) return bits_clamp(offset * 8 + @clz(difference));
    }
    return std.math.maxInt(u32);
}

fn bytes_leading_ones(bytes: []const u8) u32 {
    var offset: usize = 0;
    while (offset < bytes.len) : (offset += 8) {
        const word = bytes_word(bytes, offset);
        if (word != std.math.maxInt(u64)) return bits_clamp(offset * 8 + @clz(~word));
    }
    return bits_clamp(bytes.len * 8);
}

/// A shorter prefix is always a correct one, so lengths past u32 clamp.
fn bits_clamp(bits: usize) u32 {
    return @intCast(@min(bits, std.math.maxInt(u32)));
}

const testing = std.testing;

/// Checks the codec contract on sorted triples: windows taken past a shared prefix are
/// monotone, and the prefix length is shared by every key between the fences.
fn expect_codec_contract(
    comptime Codec: type,
    low: Codec.Key,
    middle: Codec.Key,
    high: Codec.Key,
) !void {
    try testing.expect(Codec.order(low, middle) != .gt);
    try testing.expect(Codec.order(middle, high) != .gt);
    const prefix = Codec.prefix_length(low, high);
    try testing.expect(Codec.prefix_length(low, middle) >= prefix);
    try testing.expect(Codec.prefix_length(middle, high) >= prefix);
    try testing.expect(Codec.prefix_length(null, middle) >= Codec.prefix_length(null, high));
    try testing.expect(Codec.prefix_length(middle, null) >= Codec.prefix_length(low, null));
    try testing.expect(Codec.window(low, prefix) <= Codec.window(middle, prefix));
    try testing.expect(Codec.window(middle, prefix) <= Codec.window(high, prefix));
    try testing.expectEqual(Codec.order(low, high), Codec.order_from(low, high, prefix));
    // A window's bits are the encoding's: the window one bit on is this one shifted. Fences
    // whose encodings differ only in trailing zeros share every bit, and have no next bit.
    if (prefix < std.math.maxInt(u32)) {
        const next = Codec.window(middle, prefix + 1);
        try testing.expectEqual(Codec.window(middle, prefix) << 1, next & ~@as(u64, 1));
    }
}

test "integer codecs order like their keys" {
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const random = prng.random();
    inline for (.{ u8, u16, u32, u64, u128, i8, i32, i64, i128 }) |Int| {
        const Codec = Auto(Int);
        for (0..2_000) |_| {
            var keys = [3]Int{ random.int(Int), random.int(Int), random.int(Int) };
            std.mem.sort(Int, &keys, {}, std.sort.asc(Int));
            try expect_codec_contract(Codec, keys[0], keys[1], keys[2]);
        }
        try testing.expectEqual(@as(u32, 0), Codec.prefix_length(null, null));
        try testing.expect(Codec.window(
            std.math.minInt(Int),
            0,
        ) < Codec.window(std.math.maxInt(Int), 0));
    }
}

test "integer prefix lengths count shared encoding bits" {
    const Codec = Integer(u64);
    try testing.expectEqual(@as(u32, 40), Codec.prefix_length(null, 0x0000_0000_00ff_0000));
    try testing.expectEqual(@as(u32, 16), Codec.prefix_length(0xffff_0000_0000_0000, null));
    try testing.expectEqual(@as(u32, 63), Codec.prefix_length(0x1234, 0x1235));
    try testing.expectEqual(@as(u32, 61), Codec.prefix_length(0x1230, 0x1234));
    try testing.expectEqual(@as(u32, 64), Codec.prefix_length(9, 9));
    try testing.expectEqual(
        @as(u64, 0x3456_0000_0000_0000),
        Codec.window(0x1234_5600_0000_0000, 8) & 0xffff_0000_0000_0000,
    );
    try testing.expectEqual(@as(u64, 0x8000_0000_0000_0000), Codec.window(1, 63));

    const Signed = Integer(i16);
    try testing.expectEqual(@as(u64, 0x7fff) << 48, Signed.window(-1, 0));
    try testing.expectEqual(@as(u64, 0x8000) << 48, Signed.window(0, 0));
}

test "byte codecs order lexicographically" {
    var prng = std.Random.DefaultPrng.init(0xb17e5);
    const random = prng.random();
    const Array = Auto([6]u8);
    for (0..2_000) |_| {
        var keys: [3][6]u8 = undefined;
        for (&keys) |*key| {
            for (key) |*byte| byte.* = random.intRangeAtMost(u8, 0, 3);
        }
        std.mem.sort([6]u8, &keys, {}, struct {
            fn less(_: void, a: [6]u8, b: [6]u8) bool {
                return std.mem.order(u8, &a, &b) == .lt;
            }
        }.less);
        try expect_codec_contract(Array, keys[0], keys[1], keys[2]);
    }

    var storage: [3][20]u8 = undefined;
    for (0..2_000) |_| {
        var keys: [3][]const u8 = undefined;
        for (&storage, &keys) |*bytes, *key| {
            const length = random.uintAtMost(usize, bytes.len);
            for (bytes[0..length]) |*byte| byte.* = random.intRangeAtMost(u8, 0, 2) * 0x7f;
            key.* = bytes[0..length];
        }
        std.mem.sort([]const u8, &keys, {}, struct {
            fn less(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.order(u8, a, b) == .lt;
            }
        }.less);
        try expect_codec_contract(ByteSlice, keys[0], keys[1], keys[2]);
        try testing.expectEqual(
            std.mem.order(u8, keys[0], keys[1]),
            ByteSlice.order(keys[0], keys[1]),
        );
    }
}

test "byte prefixes count the zeros after a shorter key" {
    try expect_codec_contract(ByteSlice, "", "\x00", "\x00\x00");
    try testing.expectEqual(@as(u32, 17), ByteSlice.prefix_length("ab", "abc"));
    const without_end: u32 = std.math.maxInt(u32);
    try testing.expectEqual(without_end, ByteSlice.prefix_length("ab", "ab\x00"));
    try testing.expectEqual(@as(u32, 9), ByteSlice.prefix_length(null, "\x00\x40"));
    try testing.expectEqual(@as(u32, 16), ByteSlice.prefix_length("\xff\xff\x3f", null));
    try testing.expectEqual(@as(u32, 0), ByteSlice.prefix_length(null, null));
    try testing.expectEqual(@as(u64, 0x0102_0300_0000_0000), ByteSlice.window("\x01\x02\x03", 0));
    try testing.expectEqual(@as(u64, 0x0180_0000_0000_0000), ByteSlice.window("\x01\x02\x03", 15));
    // The window's last bit is the first of the ninth byte.
    const nine = "\x00\x00\x00\x00\x00\x00\x00\x00\x80";
    try testing.expectEqual(@as(u64, 1), ByteSlice.window(nine, 1));
}

test "byte array fences without bounds use the extreme encodings" {
    const Codec = ByteArray(4);
    try testing.expectEqual(@as(u32, 21), Codec.prefix_length(null, .{ 0, 0, 7, 1 }));
    try testing.expectEqual(@as(u32, 8), Codec.prefix_length(.{ 0xff, 3, 0, 0 }, null));
    try testing.expectEqual(@as(u32, 0), Codec.prefix_length(null, null));
    try testing.expectEqual(@as(u64, 0x0701_0000_0000_0000), Codec.window(.{ 0, 0, 7, 1 }, 16));
}
