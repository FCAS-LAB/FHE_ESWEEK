#!/bin/bash
TIME_FILE="./src/Resnet50/Resnet50_time.txt"
BENCH_FILE="./src/Resnet50/Resnet50_bench.txt"
TRACE_FILE="./src/Resnet50/Resnet50_trace.txt"
TOPO_FILE="./src/Resnet50/Resnet50_gv.txt"
SHOW_FILE="./src/Resnet50/Resnet50_show.txt"
echo "Running Bilevel Algorithm"
python main.py \
--time_file="$TIME_FILE" \
--bench_file="$BENCH_FILE" \
--trace_file="$TRACE_FILE" \
--topo_file="$TOPO_FILE" \
--show_file="$SHOW_FILE"
echo "All experiments completed."