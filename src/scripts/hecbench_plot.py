#!/usr/bin/python3

# Usage: ./despmd_plot.py BASELINE_CSV CONTENDER_CSV...

from optparse import OptionParser
import matplotlib.pyplot as plt
from matplotlib import ticker
import csv
import statistics
import math
import sys
import numpy as np

def geomean(xs):
    return math.exp(math.fsum(math.log(x) for x in xs) / len(xs))

def is_anomaly(number):
    return not math.isfinite(number) or number == 0

def parse_numbers(strlist):
    """Converts list of strings to floating point values.

    Throws an exception if parsing fails or a converted number is an
    anomaly: zero, infinity or NaN.
    """

    numbers = []
    for elt in strlist:
        number = float(elt)
        if is_anomaly(number):
            raise RuntimeError(f"anomalous number: {number}")
        numbers.append(number)
    return numbers

def removesuffixes(string, suffix_list):
    for suffix in suffix_list:
        string = string.removesuffix(suffix)
    return string

def process_samples(csv_file):
    """
    Returns dict[benchmark_name:str, value:float]
    """

    f = open(csv_file, 'r')
    reader = csv.reader(f, delimiter = ',')
    bench_stats = {}
    for row in reader:
        name = removesuffixes(row[0], ['-hip', '-sycl', '-cuda', '-omp'])
        assert name != None and name != ""
        try:
            numbers = parse_numbers(row[1:])
        except Exception as e:
            print(f"warning: failed parsing {name} row: {e}")
            continue
        bench_stats[name] = min(numbers)

    return bench_stats

# Drop benchmarks that are not present in all runs.
def filter_benchmarks(benchmark_stat_list):

    if len(benchmark_stat_list) <= 1:
        return

    common_keys = set(benchmark_stat_list[0].keys())
    common_keys = common_keys.intersection(*map(set, benchmark_stat_list[1:]))

    missing_keys = set()
    for i, bench_stats in enumerate(benchmark_stat_list):
        bench_keys = set(bench_stats.keys())
        missing_keys = missing_keys.union(bench_keys.difference(common_keys))
        benchmark_stat_list[i] = {k: v for k, v in bench_stats.items()
                                  if k in common_keys}

    for k in missing_keys:
        print(f"warning: some inputs are missing a benchmark: {k}")

def times_to_speed(number_list):
    for i in reversed(range(len(number_list))):
        number_list[i] = number_list[0] / number_list[i]

def filter_outliers(benchmarks, threshold):

    filtered = []
    for name, numbers in benchmarks:
        for i, n in enumerate(numbers):
            if n > threshold or n < (1 / threshold):
                print(f"warning: filtered benchmark as outlier: {name} (n={n:.2f})")
                break
        else:
            filtered.append((name, numbers))
    return filtered

def main():
    parser = OptionParser(description="TODO")

    parser.add_option("-o", "--output-file", dest="output", default=None,
                      metavar="PATH",
                      help="if specified, write output to this file (SVG,PDF,..) otherwise show chart on screen")
    parser.add_option("-g", "--geomean", dest="geomean", default=False, action="store_true",
                      help="draw geometric mean, default = don't draw")
    parser.add_option("-r", "--refline", dest="refline", default=True,
                      action="store_true",
                      help="draw dotted line @ y=1.0, default = draw")
    parser.add_option("-y", "--ylabel", dest="ylabel", default=None,
                      help="Y axis label (optional)", metavar="YLABEL")
    parser.add_option("-t", "--title", dest="title", default=None,
                      help="chart title (optional)", metavar="TITLE")
    parser.add_option("--legends", default=None,
                      help="Set legends for contender data sets.")

    (options, args) = parser.parse_args()

    if (len(args) < 2):
        raise RuntimeError("Need two or more CSV files.")

    n_contenders = len(args) - 1

    if options.legends:
        categories = options.legends.split(",")
        if len(categories) < n_contenders:
            print(f"warning: --legends needs {n_contenders} values but "
                  + f"only {len(categories)} were given.")
            categories += ["MISSING LEGEND"] * (n_contenders - len(categories))
    else:
        # Problems if more than 6 contender data sets are given.
        categories = "123456"[:n_contenders]

    benchmark_stat_list = [process_samples(arg) for arg in args]

    # TODO: option to include benchmarks with missing data in some runs by
    #       displaying missing times as infinite in the chart. Do filter out
    #       benchmarks out when data is missing on the baseline.
    filter_benchmarks(benchmark_stat_list)

    benchmark_names = [k for k, v in benchmark_stat_list[0].items()]
    benchmarks = []
    for b in benchmark_names:
        numbers = []
        for stat_list in benchmark_stat_list:
            numbers.append(stat_list[b])
        benchmarks.append((b, numbers))

    # Sort by times of the baseline numbers
    #benchmarks = sorted(benchmarks, key = lambda x: x[1][0])

    for name, numbers in benchmarks:
        times_to_speed(numbers)

    # Sort by speed by the first contender.
    benchmarks = sorted(benchmarks, key = lambda x: x[1][1])

    benchmarks = filter_outliers(benchmarks, 10)

    if False:
        for name, numbers in benchmarks:
            print(f"{name}", end='')
            for n in numbers[1:]:
                print(f" | {n:.2f}", end='')
            print()

    n_categories = n_contenders
    numbers_by_category = {}
    for cat in categories:
        numbers_by_category[cat] = []

    for name, numbers in benchmarks:
        for i, n in enumerate(numbers[1:]):
            numbers_by_category[categories[i]].append(n)

    # Set font size globally.
    plt.rc('font', **{'size': 6})

    locs = np.arange(len(benchmarks))
    width = 0.8
    baseline = 1

    fig, ax = plt.subplots()
    for i, (cat, numbers) in enumerate(numbers_by_category.items()):
        subwidth = width / n_categories
        offset = subwidth * i
        bars = ax.bar(locs + offset, [n - baseline for n in numbers], subwidth,
                      bottom=baseline, label=cat)

        # Add numbers on the bars.
        # ax.bar_label(bars, padding=3, fmt='%.2f', rotation=90)

        ax.tick_params(axis='x', which='major', rotation=90)

    xtick_locs = locs - 1 / n_categories / 2 + 0.5 * width
    ax.set_xticks(xtick_locs, [b[0] for b in benchmarks])

    ax.margins(x=None, y=0.1)

    if True:
        # Logarithmic Y-axis scale with dashed horizontal lines.
        ax.set_yscale('log')
        ax.set_axisbelow(True)
        ax.grid(True, 'both', 'y', linestyle='dashed', lw=0.5)
        Formatter = ticker.FormatStrFormatter('%.1f')
        ax.yaxis.set_major_formatter(Formatter)
        ax.yaxis.set_minor_formatter(Formatter)

    if options.refline:
        ax.axhline(1.0, ls='dotted')

    if options.title:
        plt.title(options.title)

    if options.ylabel:
        plt.ylabel(options.ylabel)

    if options.geomean and n_contenders > 1:
        print("warning: --geomean with >1 contenders is not implemented.")
    elif options.geomean:
        g = geomean([x[1][1] for x in benchmarks])
        s = f"Geomean = {g:.2f}"
        print(s)
        ax.axhline(g, ls='dashed', label=s)

    ax.legend(loc='upper left', ncols=n_categories)

    plt.tight_layout()

    if options.output:
        fig.savefig(options.output)
    else:
        plt.show()

if __name__=="__main__":
    main()
