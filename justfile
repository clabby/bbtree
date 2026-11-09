build *args='':
    zig build {{args}}

test *args='':
    zig build test --summary all {{args}}

# Randomized and concurrency tests with assertions on and optimizations too, scaled up.
test-stress stress='10':
    zig build test -Doptimize=ReleaseSafe -Dstress={{stress}} --summary all

# One table per workload, the Zig map beside Rust's ordered maps; the median of runs per cell.
bench runs='5':
    cd bench/rust && cargo run --release -- --runs {{runs}}

fmt:
    zig fmt src bench build.zig
    cd bench/rust && cargo fmt

check-fmt:
    zig fmt --check src bench build.zig
    cd bench/rust && cargo fmt --check

# TigerStyle: no line over 100 columns.
check-lines:
    #!/usr/bin/env bash
    set -euo pipefail
    long=$(awk 'length > 100 { print FILENAME ":" FNR }' src/*.zig bench/*.zig build.zig)
    if [ -n "$long" ]; then
        echo "lines over 100 columns:"
        echo "$long"
        exit 1
    fi

clippy:
    cd bench/rust && cargo clippy --release -- -D warnings

lint: check-fmt check-lines clippy

# Everything CI runs, on this machine.
ci: lint test test-stress
