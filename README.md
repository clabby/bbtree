# `bbtree`

[![CI](https://img.shields.io/github/actions/workflow/status/clabby/bbtree/ci.yaml?style=for-the-badge&label=CI)](https://github.com/clabby/bbtree/actions/workflows/ci.yaml)
[![Zig](https://img.shields.io/badge/zig-0.17.1-f7a41d?style=for-the-badge)](https://ziglang.org/download/)

A wicked fast ordered map, inspired by the work of Xiangpeng Hao in [`bbtree`](https://github.com/XiangpengHao/bbtree).

```zig
const bbtree = @import("bbtree");

const Map = bbtree.Map(u64, u64, .{});

var map: Map = undefined;
try map.init(gpa, try Map.options_for(1_000_000));
defer map.deinit(gpa);

_ = try map.put(1, 10);
std.debug.assert(map.get(1).? == 10);

var iterator = map.iterator(.{ .start = .{ .included = 1 } });
while (iterator.next()) |entry| std.debug.print("{d}: {d}\n", .{ entry.key, entry.value });
```

## Benchmarks

Apple M4 Max, Zig 0.17.1, rustc 1.99.0, against [bbtree](https://crates.io/crates/bbtree) 0.2.1,
std's `BTreeMap` and [crossbeam-skiplist](https://crates.io/crates/crossbeam-skiplist) 0.1.3.
`just bench` runs every map on the same keys five times, alternating which side goes first,
and reports each cell's median. Upstream builds the way its own benchmarks build it, with one
codegen unit, fat LTO and mimalloc, and both sides spin for 300 ms before each workload.
Compare `bbtree-zig` with `bbtree (concurrent)`, and the single-threaded map with
`bbtree (unsync)`.

`u64` keys, 1M entries, ns per operation:

| map | insert | get hit | get miss | batched get | remove | scan 100 | ascending insert |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bbtree-zig | 80.7 | 34.9 | 25.8 | 21.3 | 86.7 | 193.5 | 8.2 |
| bbtree-zig, single-threaded | 81.9 | 28.6 | 21.7 | 15.2 | 88.3 | 177.9 | 6.8 |
| bbtree (concurrent) | 83.5 | 39.9 | 34.2 | 27.5 | 96.0 | 180.7 | 11.0 |
| bbtree (unsync) | 83.6 | 34.2 | 31.2 | 23.2 | 92.5 | 225.3 | 7.5 |
| std `BTreeMap` | 85.3 | 117.0 | 120.4 | | 121.5 | 523.7 | 41.1 |

`u64` keys, 10M entries, ns per operation:

| map | insert | get hit | get miss | batched get | remove | scan 100 | ascending insert |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bbtree-zig | 146.3 | 88.6 | 76.6 | 28.3 | 186.8 | 354.4 | 8.6 |
| bbtree-zig, single-threaded | 144.7 | 74.4 | 61.2 | 21.7 | 182.5 | 339.2 | 7.1 |
| bbtree (concurrent) | 151.7 | 112.1 | 85.7 | 41.3 | 182.6 | 530.3 | 10.9 |
| bbtree (unsync) | 154.0 | 102.0 | 76.4 | 33.9 | 186.6 | 505.7 | 7.6 |
| std `BTreeMap` | 117.0 | 246.5 | 261.7 | | 250.6 | 1030.4 | 44.8 |

Keys of 26 to 40 bytes behind a shared `user/` prefix, 1M entries, ns per operation:

| map | insert | get hit | get miss |
| --- | ---: | ---: | ---: |
| bbtree-zig | 156.7 | 275.7 | 82.9 |
| bbtree-zig, single-threaded | 153.6 | 253.8 | 71.2 |
| bbtree (concurrent) | 220.7 | 328.2 | 168.4 |
| bbtree (unsync) | 203.0 | 289.6 | 155.5 |
| std `BTreeMap` | 170.1 | 411.5 | 188.1 |

Random lookups on 10M entries, 2M per thread, millions per second:

| map | 1 thread | 4 | 8 | 16 | 16, 10% inserts |
| --- | ---: | ---: | ---: | ---: | ---: |
| bbtree-zig | 11.5 | 44.9 | 89.7 | 129.6 | 112.3 |
| bbtree (concurrent) | 8.8 | 34.4 | 67.7 | 102.6 | 91.7 |
| crossbeam `SkipMap` | 1.1 | 4.5 | 8.8 | 14.8 | 13.6 |
| `RwLock<BTreeMap>` | 4.0 | 13.0 | 5.5 | 2.0 | 2.0 |

Cells move by about 5% between runs, so smaller gaps are ties, as the 10M removals are.
Upstream's concurrent map scans 1M entries faster, and std's `BTreeMap` inserts random keys
faster at 10M entries. The Zig program writes its pool memory before it times anything, so its
numbers leave out page faults on first touch; the Rust maps run in one process, where later
runs reuse memory their allocator recycled.

## Differences from upstream

| | bbtree (Rust) | bbtree-zig |
| --- | --- | --- |
| Memory | Pages allocated as the map grows | Every page reserved at `init`; a full map returns `error.MapFull` |
| Pages | 1 KiB, aligned to 1 KiB | 1 KiB, each followed by a spare cache line |
| Child links | 8-byte pointers | 4-byte page indexes |
| Internal nodes | Key windows only; a tie reads a leaf | Key windows and full separator keys |
| Windows | Start at the first byte where a node's keys differ | Start at the first differing bit |
| Scans | Latch each leaf and visit entries in place | Copy each leaf out and check its version |
| Large maps | | Lookups prefetch the leaf they enter |

## Development

```sh
just test          # tests, Debug
just test-stress   # tests, ReleaseSafe, 10x iterations
just lint          # zig fmt, 100-column lines, rustfmt and clippy on the bench harness
just bench         # the tables above; needs cargo
```
