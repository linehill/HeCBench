#!/usr/bin/python3

# example for GPU vs CPU comparison:
# ./plot.py -g -s seaborn-v0_8-bright -t "HeCBench, GPU vs CPU speedup" -c hecbench_GPU.csv -b hecbench_CPU.csv

#### CONFIG USED BY CHIPSTAR PAPER:


# ./plot.py -v -r --color '#b7cce9' -m 0.8 -s seaborn-v0_8-pastel -t "HeCBench, Intel Arc750, HIP vs SYCL speedup" -c test_20_strict_hip_oclBE_arc_after.csv -b test_20_strict_sycl_oclBE_arc_after.csv -o hecbench_intel_arc750_hip_vs_sycl.pdf

# ./plot.py -v -r --color '#b7cce9' -s seaborn-v0_8-pastel -t "PowerVR BXE-4-32 GPU vs StarFive JH7110 CPU" -b CPU_3_x_with_abort_strictmath.csv -c GPU_3_x_with_abort_strictmath.csv -o hecbench_visionfive2_gpu_vs_cpu.pdf


from optparse import OptionParser
import matplotlib.pyplot as plt
import numpy as np
import csv

import math

def geomean(xs):
        return math.exp(math.fsum(math.log(x) for x in xs) / len(xs))

parser = OptionParser(description="Takes as input two CSV files produced by HeCBench's runner script (autohecbench.py), and produces a matplotlib chart with data from one CSV normalized to other CSV. Note that since hecbench.py produces CSV with timings, the input files are internally swapped to produce relative speedups (A = 3x faster B) rather than relative timing (A = 0.33 of B's time).")
parser.add_option("-b", "--input-file-baseline", dest="input_base",
                  help="CSV file with baseline data", metavar="PATH")
parser.add_option("-c", "--input-file-compared", dest="input_comps",
                  action='append',
                  help="CSV file with compared data", metavar="PATH")
parser.add_option("-o", "--output-file", dest="output", default=None, metavar="PATH",
                  help="if specified, write output to this file (SVG,PDF,..) otherwise show chart on screen")

parser.add_option("-e", "--errorbars", dest="errbars", default=False, action="store_true",
                  help="draw error bars, default = don't draw")

parser.add_option("-g", "--geomean", dest="geomean", default=False, action="store_true",
                  help="draw geometric mean, default = don't draw")

parser.add_option("--color", dest="color", default=None,
                  help="Adjust chart bar color (optional, default = None)", metavar="STRING")

parser.add_option("--ecolor", dest="ecolor", default=None,
                  help="Adjust chart bar error color (optional, default = None)", metavar="STRING")


parser.add_option("-m", "--bottom", dest="bottom", default=0.0,  type="float",
                  help="Adjust chart bottom (optional, default = 0.0)", metavar="FLOAT")

parser.add_option("-r", "--refline", dest="refline", default=False, action="store_true",
                  help="draw dotted line @ y=1.0, default = don't draw")

parser.add_option("-s", "--style", dest="style", default=None,
                  help="chart style (optional, default = None)", metavar="STRING")

parser.add_option("-t", "--title", dest="title", default=None,
                  help="chart title (optional)", metavar="TITLE")

parser.add_option("-x", "--xlabel", dest="xlabel", default=None,
                  help="X axis label (optional)", metavar="XLABEL")
parser.add_option("-y", "--ylabel", dest="ylabel", default=None,
                  help="Y axis label (optional)", metavar="YLABEL")
parser.add_option("-z", "--zlabel", dest="zlabel", default=None,
                  help="Label embedded into chart (optional)", metavar="ZLABEL")

parser.add_option("-v", "--bar-values", dest="bar_values", default=True, action="store_false",
                  help="do not draw values of each bar, default = draw")

(options, args) = parser.parse_args()
if (not options.input_comps) or (not options.input_base):
	parser.error("both input files must be specified")

Baseline = {}
Compared = []

for ci in options.input_comps:
        f = open(ci,'r')
        reader = csv.reader(f, delimiter = ',')
        Compared += [{}]
        for row in reader:
	        K = row[0]
	        if K.endswith('-hip'):
		        K = K[:-4]
	        if K.endswith('-cuda'):
		        K = K[:-5]
	        if K.endswith('-sycl'):
		        K = K[:-5]
	        Compared[-1][K] = {
	        'min': float(row[1]),
	        'mean': float(row[2]),
	        'stddev': float(row[3]),
	        'var': float(row[4]) }
        f.close()

f = open(options.input_base,'r')
reader = csv.reader(f, delimiter = ',')
for row in reader:
	K = row[0]
	if K.endswith('-hip'):
		K = K[:-4]
	if K.endswith('-cuda'):
		K = K[:-5]
	if K.endswith('-sycl'):
		K = K[:-5]
	Baseline[K] = {
	'min': float(row[1]),
	'mean': float(row[2]),
	'stddev': float(row[3]),
	'var': float(row[4]) }
f.close()

# Find benchmarks present in all inputs, erase others.
bm_intersection = set(Baseline.keys())
bm_union = set(Baseline.keys())
for c in Compared:
        bm_intersection &= set(c.keys())
        bm_union |= set(c.keys())

benchmarks_to_drop = bm_union - bm_intersection;
if len(benchmarks_to_drop):
        print("Dropped benchmarks due to missing results in some of the inputs:")
        print(benchmarks_to_drop)
for bm in Compared + [Baseline]:
        for to_drop in benchmarks_to_drop:
                bm.pop(to_drop, None)

zipped_data = [] # Compared x benchmarks x data-points
outliers = set()
for C in Compared:
        zipped_data += [[]]
        for K in Baseline.keys():
                # Recorded values are intervals, we want to plot speed-ups.
                Comp = Baseline[K]['min'] / C[K]['min']
                if Comp < 0.1 or 10 < Comp:
                        outliers.add(K)
                mins = Baseline[K]['min'] / C[K]['min']
                means = Baseline[K]['mean'] / C[K]['mean']
                stddevs = Baseline[K]['stddev'] / C[K]['mean']
                zipped_data[-1] += [(K, mins, means, stddevs)]

# Drop benches in every group if any of them which have extreme results.
for data_group in range(0, len(zipped_data)):
        zipped_data[data_group] = [x for x in zipped_data[data_group] if x[0]
                                 not in outliers]

# Sort data using 'mins' in the first compare group as the sorting key.
sort_idxs, zipped_data[0] = zip(*sorted(enumerate(zipped_data[0]),
                                        key = lambda x: x[1][1]))
for data_group in range(1, len(zipped_data)):
        temp = [()] * len(sort_idxs)
        for i in range(len(temp)):
                temp[i] = zipped_data[data_group][sort_idxs[i]]
        zipped_data[data_group] = temp

sorted_bench_names, sorted_mins, sorted_means, sorted_stddevs = zip(*zipped_data[0])

label_pos = np.arange(len(sorted_bench_names))  # the label locations
fig, ax = plt.subplots(layout='constrained')
num_groups = len(zipped_data)
bar_width = 1.0 / (num_groups + 1)

set_labels = "ABCDEFD"
for i in range(num_groups):
        _, sorted_mins, *_ = zip(*zipped_data[i])
        offset = bar_width * i
        offseted_sorted_mins = [x - float(options.bottom) for x in sorted_mins]
        rects = ax.bar(label_pos + offset, offseted_sorted_mins, bar_width,
                       align='edge', bottom=float(options.bottom), label=set_labels[i])
        if options.bar_values and num_groups == 1:
                ax.bar_label(rects, padding=3, fmt='%.2f', label_type='edge',
                             rotation='vertical', fontsize='small')

ax.set_xticks(label_pos + (bar_width * 0.5) * num_groups,
              sorted_bench_names, rotation='vertical', fontsize='small')

if num_groups > 1:
        ax.legend(loc='upper left', ncols=len(zipped_data))

# errs = sorted_stddevs
# if True or not options.errbars: # TODO
# 	errs = None

if options.style:
	plt.style.use(options.style)

if options.xlabel:
        plt.xlabel(options.xlabel)
if options.ylabel:
        plt.ylabel(options.ylabel)
if options.title:
        plt.title(options.title)

plt.xticks(rotation='vertical')
if options.refline:
	plt.axhline(1.0, ls='dotted')
if options.geomean and num_groups == 1:
	g = geomean(sorted_mins)
	s = "Geomean = %.2f" % g
	plt.axhline(g, ls='dashed', label=s)

plt.tight_layout()

if options.output:
	plt.savefig(options.output)
else:
	plt.show()
