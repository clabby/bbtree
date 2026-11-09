//! Compares bbtree's Zig map with Rust's ordered maps: upstream bbtree, std's `BTreeMap`,
//! crossbeam's `SkipMap` and an `RwLock<BTreeMap>`. Prints one markdown table per workload, a
//! row per map and a column per operation.
//!
//! The Rust maps run in this process. The Zig map runs natively in its own program,
//! `bench/bench.zig`, which this one builds with `zig build install -Doptimize=ReleaseFast` and
//! runs once per workload: no call crosses a language boundary. Both sides run the same
//! workloads key for key: keys are scrambled ordinals, and random orders come from SplitMix64
//! with the same seeds. Constants marked as shared must match `bench/bench.zig`.
//!
//! Upstream bbtree builds here as its own benchmarks build it: one codegen unit, fat LTO and
//! mimalloc. Each side spins for 300 ms before a workload, so that both start on a busy core.
//!
//! `cargo run --release -- --runs <n>` repeats every workload n times, alternating which side
//! goes first, and prints each cell's median. `ZIG` names the zig binary.

use std::collections::BTreeMap;
use std::env;
use std::hint::black_box;
use std::ops::{ControlFlow, Range};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Barrier, RwLock};
use std::thread;
use std::time::Instant;

use crossbeam_skiplist::SkipMap;

#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Shared with bench/bench.zig: odd, so multiplying by it permutes u64.
const SCRAMBLE: u64 = 0x9e37_79b9_7f4a_7c15;
/// Shared with bench/bench.zig.
const BATCH: usize = 64;
/// Shared with bench/bench.zig.
const SCAN_LEN: usize = 100;
/// Shared with bench/bench.zig.
const THREAD_COUNTS: [usize; 4] = [1, 4, 8, 16];
/// Shared with bench/bench.zig.
const MIX_THREADS: usize = 16;
const THREAD_OPS: usize = 2_000_000;

fn key(ordinal: u64) -> u64 {
    ordinal.wrapping_mul(SCRAMBLE)
}

/// SplitMix64, as bench/bench.zig draws it.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(SCRAMBLE);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `0..bound`, without a division.
    fn below(&mut self, bound: u64) -> u64 {
        ((u128::from(self.next()) * u128::from(bound)) >> 64) as u64
    }
}

/// The ordinals `0..n` in a random order: Fisher-Yates, as bench/bench.zig shuffles.
fn permutation(n: usize, seed: u64) -> Vec<u64> {
    let mut ordinals: Vec<u64> = (0..n as u64).collect();
    let mut rng = Rng(seed);
    for i in (1..n).rev() {
        ordinals.swap(i, rng.below(i as u64 + 1) as usize);
    }
    ordinals
}

/// Runs `f` once and returns its nanoseconds per operation, for `ops` operations, with its result.
fn timed<R>(ops: usize, f: impl FnOnce() -> R) -> (f64, R) {
    let start = Instant::now();
    let result = f();
    (start.elapsed().as_nanos() as f64 / ops as f64, result)
}

/// A markdown table of maps (rows) by operations (columns), keeping every run's value of each
/// cell and printing their median.
struct Table {
    title: String,
    columns: &'static [&'static str],
    rows: Vec<(String, Vec<Vec<f64>>)>,
}

impl Table {
    fn new(title: String, columns: &'static [&'static str]) -> Self {
        Self {
            title,
            columns,
            rows: Vec::new(),
        }
    }

    fn record(&mut self, map: &str, column: &str, value: f64) {
        let index = self
            .columns
            .iter()
            .position(|known| *known == column)
            .unwrap_or_else(|| panic!("unknown column {column:?} in {:?}", self.title));
        let row = match self.rows.iter().position(|(name, _)| name == map) {
            Some(row) => row,
            None => {
                self.rows
                    .push((map.to_owned(), vec![Vec::new(); self.columns.len()]));
                self.rows.len() - 1
            }
        };
        self.rows[row].1[index].push(value);
    }

    fn print(&self) {
        println!("### {}\n", self.title);
        println!("| map | {} |", self.columns.join(" | "));
        println!("| --- |{}", " ---: |".repeat(self.columns.len()));
        for (map, cells) in &self.rows {
            let cells: Vec<String> = cells
                .iter()
                .map(|values| median(values).map_or(String::new(), |value| format!("{value:.1}")))
                .collect();
            println!("| {map} | {} |", cells.join(" | "));
        }
        println!();
    }
}

fn median(values: &[f64]) -> Option<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    match sorted.len() {
        0 => None,
        length if length % 2 == 1 => Some(sorted[middle]),
        _ => Some((sorted[middle - 1] + sorted[middle]) / 2.0),
    }
}

/// Spins for 300 ms before a workload, as bench/bench.zig does: a process that has just
/// started or woken may run on an efficiency core, or slowly, until the scheduler sees it busy.
fn warm_up() {
    let start = Instant::now();
    let mut state = 1u64;
    while start.elapsed().as_millis() < 300 {
        for _ in 0..10_000 {
            state = black_box(state.wrapping_mul(SCRAMBLE).wrapping_add(1));
        }
    }
}

/// The Zig benchmark program, built for the native CPU with every safety check off.
struct ZigBench(PathBuf);

impl ZigBench {
    fn build() -> Self {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let zig = env::var("ZIG").unwrap_or_else(|_| "zig".to_owned());
        let status = Command::new(&zig)
            .args(["build", "install", "-Doptimize=ReleaseFast"])
            .current_dir(&root)
            .status()
            .unwrap_or_else(|error| panic!("failed to run {zig}: {error}"));
        assert!(status.success(), "zig build failed");
        Self(root.join("zig-out/bin/bench"))
    }

    /// Runs one workload and records its cells, each a line of map, column and value.
    fn run(&self, table: &mut Table, arguments: &[String]) {
        let output = Command::new(&self.0)
            .args(arguments)
            .output()
            .expect("run the Zig benchmark");
        assert!(
            output.status.success(),
            "the Zig benchmark failed: {output:?}"
        );
        for line in String::from_utf8(output.stdout).expect("UTF-8").lines() {
            let fields: Vec<&str> = line.split('\t').collect();
            let [map, column, value] = fields[..] else {
                panic!("unexpected line from the Zig benchmark: {line:?}");
            };
            table.record(map, column, value.parse().expect("a number"));
        }
    }
}

/// A single-threaded `u64 -> u64` ordered map, through its fastest API for each operation.
trait U64Ordered {
    const NAME: &'static str;
    fn new() -> Self;
    fn put(&mut self, key: u64, value: u64);
    fn lookup(&self, key: u64) -> Option<u64>;
    /// How many of `keys` are present, looked up in one batched call; `None` without a
    /// batched lookup.
    fn lookup_batch(&self, _keys: &[u64; BATCH]) -> Option<usize> {
        None
    }
    fn delete(&mut self, key: u64) -> Option<u64>;
    /// The sum of the values of the first `SCAN_LEN` entries from `start` on.
    fn scan_sum(&self, start: u64) -> u64;
}

impl U64Ordered for bbtree::BBTreeMap<u64, u64> {
    const NAME: &'static str = "bbtree (concurrent)";
    fn new() -> Self {
        Self::new()
    }
    fn put(&mut self, key: u64, value: u64) {
        black_box(self.insert(key, value).expect("allocate"));
    }
    fn lookup(&self, key: u64) -> Option<u64> {
        self.get_copied(&key)
    }
    fn lookup_batch(&self, keys: &[u64; BATCH]) -> Option<usize> {
        Some(
            self.get_many_copied(keys.each_ref())
                .iter()
                .flatten()
                .count(),
        )
    }
    fn delete(&mut self, key: u64) -> Option<u64> {
        self.remove(&key)
    }
    fn scan_sum(&self, start: u64) -> u64 {
        let (mut sum, mut left) = (0, SCAN_LEN);
        self.scan(start.., |_, value| {
            sum += value;
            left -= 1;
            if left == 0 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        });
        sum
    }
}

impl U64Ordered for bbtree::unsync::BBTreeMap<u64, u64> {
    const NAME: &'static str = "bbtree (unsync)";
    fn new() -> Self {
        Self::new()
    }
    fn put(&mut self, key: u64, value: u64) {
        black_box(self.insert(key, value));
    }
    fn lookup(&self, key: u64) -> Option<u64> {
        self.get(&key).copied()
    }
    fn lookup_batch(&self, keys: &[u64; BATCH]) -> Option<usize> {
        Some(self.get_many(keys.each_ref()).iter().flatten().count())
    }
    fn delete(&mut self, key: u64) -> Option<u64> {
        self.remove(&key)
    }
    fn scan_sum(&self, start: u64) -> u64 {
        self.range(start..)
            .take(SCAN_LEN)
            .map(|(_, value)| value)
            .sum()
    }
}

impl U64Ordered for BTreeMap<u64, u64> {
    const NAME: &'static str = "std BTreeMap";
    fn new() -> Self {
        Self::new()
    }
    fn put(&mut self, key: u64, value: u64) {
        black_box(self.insert(key, value));
    }
    fn lookup(&self, key: u64) -> Option<u64> {
        self.get(&key).copied()
    }
    fn delete(&mut self, key: u64) -> Option<u64> {
        self.remove(&key)
    }
    fn scan_sum(&self, start: u64) -> u64 {
        self.range(start..)
            .take(SCAN_LEN)
            .map(|(_, value)| value)
            .sum()
    }
}

const SINGLE_THREAD_COLUMNS: [&str; 7] = [
    "insert random",
    "get hit",
    "get miss",
    "batched get hit (per key)",
    "remove random",
    "scan 100 (per scan)",
    "insert ascending",
];

/// Records nanoseconds per operation for each of `SINGLE_THREAD_COLUMNS`, on a map of
/// `order.len()` keys; the same sequence as bench/bench.zig's `single_thread`.
fn single_thread<M: U64Ordered>(table: &mut Table, order: &[u64]) {
    let n = order.len();
    let mut map = M::new();
    let (insert, ()) = timed(n, || {
        for ordinal in 0..n as u64 {
            map.put(key(ordinal), ordinal);
        }
    });
    let (get_hit, hits) = timed(n, || {
        order
            .iter()
            .filter(|&&ordinal| black_box(map.lookup(key(ordinal))).is_some())
            .count()
    });
    assert_eq!(hits, n);
    let (get_miss, hits) = timed(n, || {
        order
            .iter()
            .filter(|&&ordinal| black_box(map.lookup(key(ordinal + n as u64))).is_some())
            .count()
    });
    assert_eq!(hits, 0);
    let batched_keys = n / BATCH * BATCH;
    let (batched, hits) = timed(batched_keys, || {
        let mut hits = 0;
        for chunk in order.as_chunks::<BATCH>().0 {
            hits += map.lookup_batch(&chunk.map(key))?;
        }
        Some(hits)
    });
    let scans = &order[..n / 10];
    let (scan, sum) = timed(scans.len(), || {
        scans.iter().fold(0u64, |sum, &ordinal| {
            sum.wrapping_add(map.scan_sum(key(ordinal)))
        })
    });
    black_box(sum);
    let (remove, removed) = timed(n, || {
        order
            .iter()
            .filter(|&&ordinal| map.delete(key(ordinal)).is_some())
            .count()
    });
    assert_eq!(removed, n);
    drop(map);

    let mut map = M::new();
    let (ascending, ()) = timed(n, || {
        for ordinal in 0..n as u64 {
            map.put(ordinal, ordinal);
        }
    });
    for (column, value) in [
        ("insert random", insert),
        ("get hit", get_hit),
        ("get miss", get_miss),
        ("remove random", remove),
        ("scan 100 (per scan)", scan),
        ("insert ascending", ascending),
    ] {
        table.record(M::NAME, column, value);
    }
    if let Some(hits) = hits {
        assert_eq!(hits, batched_keys);
        table.record(M::NAME, "batched get hit (per key)", batched);
    }
}

fn single_thread_rust(table: &mut Table, n: usize) {
    let order = permutation(n, 1);
    single_thread::<bbtree::BBTreeMap<u64, u64>>(table, &order);
    single_thread::<bbtree::unsync::BBTreeMap<u64, u64>>(table, &order);
    single_thread::<BTreeMap<u64, u64>>(table, &order);
}

/// A single-threaded map from borrowed byte strings to `u64`.
trait BytesOrdered<'k> {
    const NAME: &'static str;
    fn new() -> Self;
    fn put(&mut self, key: &'k [u8], value: u64);
    fn lookup(&self, key: &[u8]) -> Option<u64>;
}

impl<'k> BytesOrdered<'k> for bbtree::BBTreeMap<&'k [u8], u64> {
    const NAME: &'static str = "bbtree (concurrent)";
    fn new() -> Self {
        Self::new()
    }
    fn put(&mut self, key: &'k [u8], value: u64) {
        black_box(self.insert(key, value).expect("allocate"));
    }
    fn lookup(&self, key: &[u8]) -> Option<u64> {
        self.get_copied(key)
    }
}

impl<'k> BytesOrdered<'k> for bbtree::unsync::BBTreeMap<&'k [u8], u64> {
    const NAME: &'static str = "bbtree (unsync)";
    fn new() -> Self {
        Self::new()
    }
    fn put(&mut self, key: &'k [u8], value: u64) {
        black_box(self.insert(key, value));
    }
    fn lookup(&self, key: &[u8]) -> Option<u64> {
        self.get(key).copied()
    }
}

impl<'k> BytesOrdered<'k> for BTreeMap<&'k [u8], u64> {
    const NAME: &'static str = "std BTreeMap";
    fn new() -> Self {
        Self::new()
    }
    fn put(&mut self, key: &'k [u8], value: u64) {
        black_box(self.insert(key, value));
    }
    fn lookup(&self, key: &[u8]) -> Option<u64> {
        self.get(key).copied()
    }
}

/// A key like `user/0517/9e3779b97f4a7c15012`, as bench/bench.zig makes it: 26 to 40 bytes
/// behind a shared `user/` prefix, distinct by the 16 hex digits of the scrambled ordinal.
fn byte_key(ordinal: u64) -> Vec<u8> {
    let scrambled = key(ordinal);
    let shard = scrambled >> 54;
    let padding = &"0123456789abcd"[..(scrambled % 15) as usize];
    format!("user/{shard:04}/{scrambled:016x}{padding}").into_bytes()
}

/// The keys of `ordinals`, back to back in one buffer as bench/bench.zig's arena lays them out,
/// and where each lies in it.
fn byte_keys(ordinals: Range<u64>) -> (Vec<u8>, Vec<Range<usize>>) {
    let mut buffer = Vec::new();
    let mut spans = Vec::new();
    for ordinal in ordinals {
        let start = buffer.len();
        buffer.extend_from_slice(&byte_key(ordinal));
        spans.push(start..buffer.len());
    }
    (buffer, spans)
}

/// Records nanoseconds per insert, hit and miss, on a map of `keys` looked up in `order`.
fn bytes<'k, M: BytesOrdered<'k>>(
    table: &mut Table,
    keys: &[&'k [u8]],
    misses: &[&[u8]],
    order: &[u64],
) {
    let n = keys.len();
    let mut map = M::new();
    let (insert, ()) = timed(n, || {
        for (value, &key) in keys.iter().enumerate() {
            map.put(key, value as u64);
        }
    });
    let (get_hit, hits) = timed(n, || {
        order
            .iter()
            .filter(|&&ordinal| black_box(map.lookup(keys[ordinal as usize])).is_some())
            .count()
    });
    assert_eq!(hits, n);
    let (get_miss, hits) = timed(n, || {
        misses
            .iter()
            .filter(|&&key| black_box(map.lookup(key)).is_some())
            .count()
    });
    assert_eq!(hits, 0);
    table.record(M::NAME, "insert random", insert);
    table.record(M::NAME, "get hit", get_hit);
    table.record(M::NAME, "get miss", get_miss);
}

fn bytes_rust(table: &mut Table, n: usize) {
    let (hit_bytes, hit_spans) = byte_keys(0..n as u64);
    let (miss_bytes, miss_spans) = byte_keys(n as u64..2 * n as u64);
    let keys: Vec<&[u8]> = hit_spans
        .iter()
        .map(|span| &hit_bytes[span.clone()])
        .collect();
    let misses: Vec<&[u8]> = miss_spans
        .iter()
        .map(|span| &miss_bytes[span.clone()])
        .collect();
    let order = permutation(n, 2);
    bytes::<bbtree::BBTreeMap<&[u8], u64>>(table, &keys, &misses, &order);
    bytes::<bbtree::unsync::BBTreeMap<&[u8], u64>>(table, &keys, &misses, &order);
    bytes::<BTreeMap<&[u8], u64>>(table, &keys, &misses, &order);
}

/// A `u64 -> u64` map that threads share.
trait Shared: Sync {
    const NAME: &'static str;
    fn new() -> Self;
    fn lookup(&self, key: u64) -> Option<u64>;
    fn put(&self, key: u64, value: u64);
}

impl Shared for bbtree::BBTreeMap<u64, u64> {
    const NAME: &'static str = "bbtree (concurrent)";
    fn new() -> Self {
        Self::new()
    }
    fn lookup(&self, key: u64) -> Option<u64> {
        self.get_copied(&key)
    }
    fn put(&self, key: u64, value: u64) {
        black_box(self.insert(key, value).expect("allocate"));
    }
}

impl Shared for SkipMap<u64, u64> {
    const NAME: &'static str = "crossbeam SkipMap";
    fn new() -> Self {
        Self::new()
    }
    fn lookup(&self, key: u64) -> Option<u64> {
        self.get(&key).map(|entry| *entry.value())
    }
    fn put(&self, key: u64, value: u64) {
        self.insert(key, value);
    }
}

impl Shared for RwLock<BTreeMap<u64, u64>> {
    const NAME: &'static str = "RwLock<BTreeMap>";
    fn new() -> Self {
        Self::default()
    }
    fn lookup(&self, key: u64) -> Option<u64> {
        self.read().unwrap().get(&key).copied()
    }
    fn put(&self, key: u64, value: u64) {
        black_box(self.write().unwrap().insert(key, value));
    }
}

/// Millions of operations per second with `threads` threads each running `THREAD_OPS` random
/// lookups of the `n` prebuilt keys, or with `mixed`, one in ten of them inserts of new keys.
fn throughput<M: Shared>(map: &M, n: u64, threads: usize, mixed: bool) -> f64 {
    let barrier = Barrier::new(threads + 1);
    let start = thread::scope(|scope| {
        for thread in 0..threads {
            let barrier = &barrier;
            scope.spawn(move || {
                let mut rng = Rng(thread as u64 + 1);
                let mut next_insert = n + (thread * THREAD_OPS) as u64;
                let (mut gets, mut hits) = (0, 0);
                barrier.wait();
                for _ in 0..THREAD_OPS {
                    if mixed && rng.next().is_multiple_of(10) {
                        map.put(key(next_insert), next_insert);
                        next_insert += 1;
                    } else {
                        gets += 1;
                        hits += usize::from(map.lookup(key(rng.below(n))).is_some());
                    }
                }
                assert_eq!(hits, gets);
            });
        }
        barrier.wait();
        Instant::now()
    });
    (threads * THREAD_OPS) as f64 / start.elapsed().as_secs_f64() / 1e6
}

const THREAD_COLUMNS: [&str; 5] = [
    "get ×1",
    "get ×4",
    "get ×8",
    "get ×16",
    "90% get / 10% insert ×16",
];

/// Builds a map of `n` keys and records each thread cell.
fn threaded<M: Shared>(table: &mut Table, n: usize) {
    let map = M::new();
    for ordinal in 0..n as u64 {
        map.put(key(ordinal), ordinal);
    }
    for (column, threads) in THREAD_COLUMNS.iter().zip(THREAD_COUNTS) {
        table.record(M::NAME, column, throughput(&map, n as u64, threads, false));
    }
    let mixed = throughput(&map, n as u64, MIX_THREADS, true);
    table.record(M::NAME, THREAD_COLUMNS[4], mixed);
}

fn threaded_rust(table: &mut Table, n: usize) {
    threaded::<bbtree::BBTreeMap<u64, u64>>(table, n);
    threaded::<SkipMap<u64, u64>>(table, n);
    threaded::<RwLock<BTreeMap<u64, u64>>>(table, n);
}

/// One table: the Zig program's arguments for it, and the Rust side's workload.
struct Workload {
    table: Table,
    zig_arguments: Vec<String>,
    rust: fn(&mut Table, usize),
    n: usize,
}

fn workloads() -> Vec<Workload> {
    let u64_table = |n: usize, title: &str| Workload {
        table: Table::new(title.to_owned(), &SINGLE_THREAD_COLUMNS),
        zig_arguments: vec!["u64".to_owned(), n.to_string()],
        rust: single_thread_rust,
        n,
    };
    vec![
        u64_table(1_000_000, "u64 keys, 1M entries, ns per operation"),
        u64_table(10_000_000, "u64 keys, 10M entries, ns per operation"),
        Workload {
            table: Table::new(
                "26 to 40 byte keys with a shared prefix, 1M entries, ns per operation".to_owned(),
                &["insert random", "get hit", "get miss"],
            ),
            zig_arguments: vec!["bytes".to_owned(), "1000000".to_owned()],
            rust: bytes_rust,
            n: 1_000_000,
        },
        Workload {
            table: Table::new(
                format!(
                    "Lookups by threads, 10M entries, {THREAD_OPS} operations a thread, \
                     millions of operations per second"
                ),
                &THREAD_COLUMNS,
            ),
            zig_arguments: vec![
                "threads".to_owned(),
                "10000000".to_owned(),
                THREAD_OPS.to_string(),
            ],
            rust: threaded_rust,
            n: 10_000_000,
        },
    ]
}

fn main() {
    let arguments: Vec<String> = env::args().collect();
    let runs: usize = match arguments.get(1..) {
        Some([flag, value]) if flag == "--runs" => value.parse().expect("--runs <n>"),
        Some([]) | None => 1,
        _ => panic!("usage: bbtree-bench [--runs <n>]"),
    };
    let zig = ZigBench::build();
    let started = Instant::now();
    let mut workloads = workloads();
    for run in 0..runs {
        for workload in &mut workloads {
            // Alternate which side goes first, so neither always meets a cold machine.
            if run % 2 == 0 {
                zig.run(&mut workload.table, &workload.zig_arguments);
                warm_up();
                (workload.rust)(&mut workload.table, workload.n);
            } else {
                warm_up();
                (workload.rust)(&mut workload.table, workload.n);
                zig.run(&mut workload.table, &workload.zig_arguments);
            }
        }
    }
    for workload in &workloads {
        workload.table.print();
    }
    println!(
        "Median of {runs} run(s) a cell; {:.0} s in all.",
        started.elapsed().as_secs_f64()
    );
}
