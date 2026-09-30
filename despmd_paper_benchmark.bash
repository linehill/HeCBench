#!/bin/bash
# USAGE $0 [-o OUTPUT_DIR] ENVIRONMENT [RUNCONF]
#
# Run benchmarks specified by RUNCONF on the given ENVIRONMENT.
#
# ENVIRONMENT: A sourceable bash script that setups benchmark
# environment (e.g. to run benchmarks on chipStar->PoCL). The script must
# export following things:
#
# * RUNTIME: variable: choose whether to run hip, sycl or cuda variants
#            of the benchmarks. Valid options are 'cpu', 'hip', 'sycl'
#            and 'cuda'.
#
# * COMPILER: variable: path to corresponding compiler of the
#             ${RUNTIME} to used to build benchmarks.
#
# * SYCL_TYPE: variable: Backend to use for SYCL runtime. Required
#              when RUNTIME=sycl.
#
# * print_env_details: a function which can be empty. Use it if you want to
#   record details about the benchmark run into benchmark log.
#
# Variables needed in specific cases:
#
# * SYCL_VENDOR=AdaptiveCpp when the SYCL runtime is AdaptiveCPP.
#
# RUNCONF: Optional. Sourceable bash script that configures the
# benchmark run. If not defined <HeCBench>/default.runconf will be
# used. Following things must be exported by the script:
#
# * REPEATS variable: The number of times each benchmark is run.
#
# * WARMUP variable: whether each benchmark is run once without measurements.
#
# * TIMEOUT variable: Terminate single benchmark if its time exceeds
#           TIMEOUT seconds.
#
# * BENCHMARK_LIST variable: Path to JSON file which lists benchmarks to be
#                  run. See 'subset.json' for an example.
#
# -o OUTPUT_DIR: Destination dir to store benchmark results. Defaults to
#                <HeCBench>/benchmark-results/<ENVIRONMENT>.
set -u -o pipefail
ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

OUTPUT_DIR=""
if [[ "$1" == "-o" ]]; then
    shift
    OUTPUT_DIR=${1:?"error: missing -o option value!"}
    shift
fi

ENVIRONMENT=${1:?"Need environment setup file!"}
RUNCONF=${2:-"${ROOT_DIR}/default.runconf"}

# Avoid mistake of mixing up the order of ENVIRONMENT and RUNCONF arguments.
if ! [[ "${ENVIRONMENT}" =~ .*[.]env ]]; then
   >&2 echo "error: ENVIRONMENT argument is missing .env suffix: '${ENVIRONMENT}'"
    exit 2
fi

if ! [[ "${RUNCONF}" =~ .*[.]runconf ]]; then
   >&2 echo "error: RUNCONF argument is missing .runconf suffix: '${RUNCONF}'"
    exit 2
fi

if [ ! -e "${ENVIRONMENT}" ]; then
    >&2 echo "error: missing environment setup script: '${ENVIRONMENT}'"
    exit 2
fi

if [ ! -e "${RUNCONF}" ]; then
    >&2 echo "error: missing run config script: '${RUNCONF}'"
    exit 2
fi

set -e

if [[ "${OUTPUT_DIR}" == "" ]]; then
    OUTPUT_DIR="${ROOT_DIR}/benchmark-results/${ENVIRONMENT}"
    mkdir -p "${OUTPUT_DIR}"
else
    mkdir -p "${OUTPUT_DIR}"
    OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"
fi

# Place to save logs and benchmark results.

mkdir -p "${OUTPUT_DIR}"

echo "saving benchmark results in '${OUTPUT_DIR}'"


# Echo command before executing it and log its outputs.
function simon_says() {
    echo
    echo "> $@"
    "$@" |& tee -a "${OUTPUT_DIR}/benchmarks.log"
}

source "${ENVIRONMENT}"
source "${RUNCONF}"

# print_details is defined in ${ENVIRONMENT}.env.
simon_says print_env_details

simon_says "$COMPILER" --version

# Record benchmark environment.
(
    cd "${OUTPUT_DIR}"
    env | sort > benchmark.env
)

# Smoketest hipcc and HIP runtime and stop early. Lots of benchmarks
# are unaware of kernel launch failures and thus produce bogus kernel
# times.
# TODO: make one for 'sycl'.
if [[ "${RUNTIME}" == "hip" ]]; then
    (
	simon_says logsave -a ${OUTPUT_DIR}/smoketest.log \
		   $COMPILER -O2 ${ROOT_DIR}/tools/smoketest.hip -o hip-smoketest \
		   >/dev/null 2>&1
	simon_says logsave -a ${OUTPUT_DIR}/smoketest.log \
		   ./hip-smoketest
    ) || {
	cat ${OUTPUT_DIR}/smoketest.log
	echo "ABORT: smoketest failed!"
	exit 2
    }
fi



SYCL_OPTS=""
SYCL_VENDOR=${SYCL_VENDOR:-unknown}
if [[ "${RUNTIME}" == "sycl" ]]; then
    SYCL_OPTS="--extra-compile-flags=-ffp-model=precise --sycl-type=${SYCL_TYPE}"
fi

# Record what benchmarks was / were attemped to be run.
cp "${BENCHMARK_LIST}" ${OUTPUT_DIR}/

(
    export KERNEL_STATS_RESULT_PATH=${OUTPUT_DIR}/pocl-kernel-stats.json

    simon_says "${ROOT_DIR}/src/scripts/autohecbench.py" \
               --compiler-name="$COMPILER" $SYCL_OPTS \
               --sycl-vendor=${SYCL_VENDOR} -r "$REPEATS" \
               -w "$WARMUP" --yes-prompt --timeout="$TIMEOUT" --clean \
               --bench-data "${BENCHMARK_LIST}" --overwrite \
               --summary "${OUTPUT_DIR}"/summary.json \
               -o ${OUTPUT_DIR}/benchmarks.csv \
               --launcher "${ROOT_DIR}/launcher-capture-pocl-kernel-stats.py" \
               ${BENCHMARK_PREFIX:-}${RUNTIME}
)
