#!/usr/bin/env python3
#
# usage: $0 KERNEL_STATS_JSON [BENCMARK_SUMMARY_JSON]
#
# Parses data (KERNEL_STATS_JSON) produced by
# launcher-capture-pocl-kernel-stats.py. Optional
# BENCMARK_SUMMARY_JSON file filters out benchmarks that didn't
# succeed due to errors.

#import re, time, datetime, sys, subprocess, multiprocessing, os, shutil
#from collections import namedtuple
import os, sys
from statistics import median
import json

def load_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def read_rejects(summary_json_path):
    rejects = set()
    try:
        benchmarks = load_json(summary_json_path)
        for k, v in benchmarks.items():
            if not 'run' in v or v['run'] != 'success':
                rejects.add(k)

    except Exception as e:
        print(e, file=sys.stderr)
        print("warning: failed to process benchmark summary file. "
              "Proceeding without it.", file=sys.stderr)
        return set()

    return rejects

KEY_BINSIZES = 'kernel-binary-bytes'
KEY_WARMUP = 'process-warmup-time'
KEY_RUNTIME = 'process-run-time'

def main():

    data = load_json(sys.argv[1])

    rejects = set()
    if (len(sys.argv) > 2):
        rejects = read_rejects(sys.argv[2])

    first = True
    for k, v in data.items():
        benchmark_name = k
        if benchmark_name in rejects:
            print(f"skipping rejected benchmark: {benchmark_name}",
                  file=sys.stderr)
            continue

        kernel_binary_size = v.get(KEY_BINSIZES, "N/A")
        kernel_compilation_time = "N/A"

        try:
            warmup_time = float(v[KEY_WARMUP])
            run_time = median([float(x) for x in v[KEY_RUNTIME]])
            kernel_compilation_time = warmup_time - run_time
        except:
            print("warning: error in obtaining kernel compilation time "
                  f"for {k}", file=sys.stderr)

        if first:
            first = False
            print(f"benchmark,kernel binary size,kernel compilation time")

        print(f"{k},{kernel_binary_size},{kernel_compilation_time}")


if __name__ == "__main__":
    main()
