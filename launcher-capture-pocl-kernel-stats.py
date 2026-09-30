#!/usr/bin/env python3
#
# A benchmark launcher to collect PoCL kernel statistics. Collects:
#
# * CPU kernel binary sizes by setting up a cold kcache and counting bytes
#   of kernel binaries after benchmark run.
#
# * kernel compilation time by measuring benchmark process
#   times. After collection the time is obtained by substracting process time
#   on warmp-up with average time of the actual runs.

import re, time, datetime, sys, subprocess, multiprocessing, os, shutil
from collections import namedtuple
import json

def setup_kcache(args):
    kcache_dir = os.environ['POCL_CACHE_DIR']

    if len(kcache_dir) == 0:
        raise Exception("POCL_CACHE_DIR is an empty string!");

    kcache_dir += "/" + args.name
    os.environ['POCL_CACHE_DIR'] = kcache_dir

    if not args.warmup and os.path.exists(kcache_dir):
        return kcache_dir

    if os.path.exists(kcache_dir):
        if not os.path.isdir(kcache_dir):
            raise Exception("POCL_CACHE_DIR points to non-directory!")
        shutil.rmtree(kcache_dir)

    os.makedirs(kcache_dir)

    return kcache_dir

def setup_output_dir(args):
    output_dir = "./test_dir/" + args.name # placeholder
    os.makedirs(kcache_dir)
    return output_dir

def load_json(json_path):
    dirs, filename = os.path.split(json_path)
    os.makedirs(dirs, exist_ok=True)

    if not os.path.isfile(json_path):
        with open(json_path, 'w') as f:
            json.dump({}, f)

    with open(json_path, 'r') as f:
        return json.load(f)

def save_json(json_path, data):
    dirs, filename = os.path.split(json_path)
    os.makedirs(dirs, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(data, f)

def count_kernel_binary_bytes(kcache_dir):
    n_bytes = 0
    for root, dirs, files in os.walk(kcache_dir):
        for file in files:
            if not (file.endswith(".so") or
                    file.endswith(".so.o")):
                continue
            fullpath = os.path.join(root, file)
            n_bytes += os.path.getsize(fullpath)

    return n_bytes

def get_result_json_path():
    key = 'KERNEL_STATS_RESULT_PATH'
    if key in os.environ:
        return os.environ[key]
    result_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(result_dir, "kernel-stats.json")

def main():
    args = namedtuple('Args', ['name', 'warmup', 'cmd'])
    args.name = sys.argv[1]
    args.warmup = int(sys.argv[2])
    args.cmd = sys.argv[3:]

    # This launcher is meant for benchmarks targeting PoCL. Skip stats
    # collection if not targeting PoCL.
    if not 'POCL_CACHE_DIR' in os.environ:
        proc = subprocess.run(args.cmd, encoding="utf-8")
        sys.exit(proc.returncode)

    # Do setups up-front for failing fast on any trouble.
    kcache_dir = setup_kcache(args)
    json_path = get_result_json_path()
    data = load_json(json_path)

    t0 = time.time()
    proc = subprocess.run(args.cmd, encoding="utf-8")
    proc_time = time.time() - t0

    if not args.name in data:
        data[args.name] = {}
    entry = data[args.name]

    KEY_BINSIZES = 'kernel-binary-bytes'
    KEY_WARMUP = 'process-warmup-time'
    KEY_RUNTIME = 'process-run-time'

    if args.warmup:
        entry[KEY_BINSIZES] = count_kernel_binary_bytes(kcache_dir)
        entry[KEY_WARMUP] = proc_time
        if KEY_RUNTIME in entry:
            del entry[KEY_RUNTIME]
    else:
        if not KEY_RUNTIME in entry:
            entry[KEY_RUNTIME] = []
        entry[KEY_RUNTIME].append(proc_time)

    save_json(json_path, data)
    sys.exit(proc.returncode)

if __name__ == "__main__":
    main()
