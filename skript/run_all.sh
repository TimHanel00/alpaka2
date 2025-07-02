#!/bin/bash

set +e  # Do not stop on errors

BINARY_DIR="$HOME/workspace/alpaka2_fork/build"

echo "=== Running babelstream benchmarks ==="
for exp in 17 19 21 23 25 27 29; do
    size=$((2**exp))
    echo "Running size=$size"
    "$BINARY_DIR/benchmark/babelstream/babelstream" --array-size=$size --number-runs=40000> "babel_${size}_timing.csv"|| echo "babelstream failed for size=$size"
done

echo "=== Running heatEquation2D_Frame benchmarks ==="
for factor in 1 4 8 12 16; do
    size=$((1024 * factor))
    echo "Running size=$size"
    "$BINARY_DIR/example/heatEqu2DWithFrame/heatEquation2D_Frame" --numNodes=$size > "heat_${size}_timing.csv" || echo "heatEquation2D_Frame failed for size=$size"
done

echo "=== All done (some runs may have failed, which is expected) ==="
