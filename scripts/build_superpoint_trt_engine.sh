#!/usr/bin/env bash

set -euo pipefail

TRTEXEC_BIN="${TRTEXEC_BIN:-/usr/bin/trtexec}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

onnx_path="${ROOT_DIR}/onnx_files/superpoint.onnx"
engine_dir="${ROOT_DIR}/trt_engine_files"
batch=1
height=480
width=640
mode="fp32"
tag="noDLA"
use_cuda_graph=1
warmup=200
iterations=50
duration=30
verbose=0
use_shapes=0

remove_path_entry() {
  local target="$1"
  local current_value="${LD_LIBRARY_PATH:-}"
  local updated_value=""
  local old_ifs="$IFS"

  IFS=':'
  for entry in $current_value; do
    if [[ -z "$entry" || "$entry" == "$target" ]]; then
      continue
    fi
    if [[ -n "$updated_value" ]]; then
      updated_value+="":"$entry"
    else
      updated_value="$entry"
    fi
  done
  IFS="$old_ifs"

  export LD_LIBRARY_PATH="$updated_value"
}

usage() {
  cat <<'EOF'
Usage: build_superpoint_trt_engine.sh [options]

Options:
  --onnx PATH           Path to superpoint ONNX.
  --engine-dir PATH     Output directory for engine and log.
  --batch N             Batch size, usually 1 for mono or 2 for stereo.
  --height N            Input image height.
  --width N             Input image width.
  --mode MODE           fp32 | fp16 | best.
  --tag TAG             Engine filename tag suffix.
  --warmup N            trtexec warmUp iterations.
  --iterations N        trtexec iterations.
  --duration N          trtexec duration in seconds.
  --with-shapes         Pass min/opt/max shape profiles to trtexec.
  --verbose             Enable trtexec verbose log.
  --no-cuda-graph       Disable --useCudaGraph.
  -h, --help            Show this help.

Notes:
  fp16 mode intentionally uses only --fp16.
  Do not add --precisionConstraints=obey or --layerPrecisions=*:fp16 for SuperPoint.
  Static ONNX models should typically omit shape profiles; use --with-shapes only for dynamic-shape exports.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --onnx)
      onnx_path="$2"
      shift 2
      ;;
    --engine-dir)
      engine_dir="$2"
      shift 2
      ;;
    --batch)
      batch="$2"
      shift 2
      ;;
    --height)
      height="$2"
      shift 2
      ;;
    --width)
      width="$2"
      shift 2
      ;;
    --mode)
      mode="$2"
      shift 2
      ;;
    --tag)
      tag="$2"
      shift 2
      ;;
    --warmup)
      warmup="$2"
      shift 2
      ;;
    --iterations)
      iterations="$2"
      shift 2
      ;;
    --duration)
      duration="$2"
      shift 2
      ;;
    --with-shapes)
      use_shapes=1
      shift
      ;;
    --verbose)
      verbose=1
      shift
      ;;
    --no-cuda-graph)
      use_cuda_graph=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "$onnx_path" ]]; then
  echo "ONNX file not found: $onnx_path" >&2
  exit 1
fi

if [[ ! -x "$TRTEXEC_BIN" ]]; then
  echo "trtexec not found or not executable: $TRTEXEC_BIN" >&2
  exit 1
fi

case "$mode" in
  fp32|fp16|best)
    ;;
  *)
    echo "Unsupported mode: $mode" >&2
    exit 1
    ;;
esac

mkdir -p "$engine_dir"

if [[ "$batch" -eq 1 ]]; then
  layout="mono"
elif [[ "$batch" -eq 2 ]]; then
  layout="stereo"
else
  layout="batch${batch}"
fi

engine_path="${engine_dir}/superpoint.${layout}.${height}x${width}.${mode}.${tag}.engine"
log_path="${engine_path}.txt"
shape="image:${batch}x1x${height}x${width}"

# Avoid stale local TensorRT runtimes under /usr/local/cuda overshadowing the apt-managed 10.16.1 libraries.
remove_path_entry "/usr/local/cuda/lib64"
remove_path_entry "/usr/local/cuda/targets/x86_64-linux/lib"

cmd=(
  "$TRTEXEC_BIN"
  "--onnx=${onnx_path}"
  "--saveEngine=${engine_path}"
  "--warmUp=${warmup}"
  "--iterations=${iterations}"
  "--duration=${duration}"
)

if [[ "$use_shapes" -eq 1 ]]; then
  cmd+=(
    "--minShapes=${shape}"
    "--optShapes=${shape}"
    "--maxShapes=${shape}"
  )
fi

if [[ "$use_cuda_graph" -eq 1 ]]; then
  cmd+=("--useCudaGraph")
fi

if [[ "$verbose" -eq 1 ]]; then
  cmd+=("--verbose")
fi

case "$mode" in
  fp16)
    cmd+=("--fp16")
    ;;
  best)
    cmd+=("--best")
    ;;
esac

echo "Building ${engine_path}"
echo "Using trtexec: $(readlink -f "$TRTEXEC_BIN")"
printf '%q ' "${cmd[@]}"
echo

"${cmd[@]}" |& tee "$log_path"
