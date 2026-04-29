#!/usr/bin/env bash

# Use the moonshine env.
source /home/cjh9fw/.bashrc
conda activate moonshine

MODEL=${MODEL:-/sfs/weka/scratch/cjh9fw/moonshine/results/moonshine-en-accents-phase1/final}
OUT=${OUT:-/sfs/weka/scratch/cjh9fw/moonshine/deployment/moonshine-en-accents-phase1-base-en}
MOONSHINE_REPO=${MOONSHINE_REPO:-/home/cjh9fw/Desktop/2026/repos/moonshine}

ONNX_MERGED="$OUT/onnx_merged"
ONNX_QUANTIZED="$OUT/onnx_quantized"
ORT_DIR="$OUT/ort"

# Prefer the same style of integer-activation quantization used by Moonshine
# assets when onnx-shrink-ray is installed. Fall back to ORT dynamic quantization
# so the script can still produce smaller Android assets in the current env.
QUANTIZATION_BACKEND=${QUANTIZATION_BACKEND:-auto}  # auto, shrink_ray, ort_dynamic
SHRINK_RAY_METHOD=${SHRINK_RAY_METHOD:-integer_activations}
ORT_OPTIMIZATION_STYLE=${ORT_OPTIMIZATION_STYLE:-Fixed}
TARGET_PLATFORM=${TARGET_PLATFORM:-arm}

have_python_module() {
  python - "$1" <<'PY'
import importlib.util
import sys

try:
    found = importlib.util.find_spec(sys.argv[1]) is not None
except ModuleNotFoundError:
    found = False

sys.exit(0 if found else 1)
PY
}

choose_quantization_backend() {
  case "$QUANTIZATION_BACKEND" in
    auto)
      if have_python_module onnx_shrink_ray.shrink; then
        echo "shrink_ray"
      else
        echo "ort_dynamic"
      fi
      ;;
    shrink_ray|ort_dynamic)
      echo "$QUANTIZATION_BACKEND"
      ;;
    *)
      echo "Unsupported QUANTIZATION_BACKEND=$QUANTIZATION_BACKEND" >&2
      echo "Use auto, shrink_ray, or ort_dynamic." >&2
      exit 2
      ;;
  esac
}

quantized_suffix_for_shrink_ray() {
  case "$SHRINK_RAY_METHOD" in
    integer_activations)
      echo "quantized_activations"
      ;;
    integer_weights)
      echo "quantized_weights"
      ;;
    *)
      echo "Unsupported SHRINK_RAY_METHOD=$SHRINK_RAY_METHOD" >&2
      echo "Use integer_activations or integer_weights." >&2
      exit 2
      ;;
  esac
}

quantize_with_shrink_ray() {
  local src="$1"
  local dst="$2"
  local base stem suffix tmp produced

  if ! have_python_module onnx_shrink_ray.shrink; then
    echo "onnx-shrink-ray is not installed in the moonshine env." >&2
    echo "Install it or use QUANTIZATION_BACKEND=ort_dynamic." >&2
    exit 2
  fi

  base=$(basename "$src")
  stem=${base%.onnx}
  suffix=$(quantized_suffix_for_shrink_ray)
  tmp="$ONNX_QUANTIZED/_tmp_${base}"
  produced="$ONNX_QUANTIZED/_tmp_${stem}_${suffix}.onnx"

  cp "$src" "$tmp"
  python -m onnx_shrink_ray.shrink \
    --ir-version 10 \
    --method "$SHRINK_RAY_METHOD" \
    "$tmp"
  mv "$produced" "$dst"
  rm -f "$tmp"
}

quantize_with_ort_dynamic() {
  local src="$1"
  local dst="$2"
  local base tmp

  base=$(basename "$src")
  tmp="$ONNX_QUANTIZED/_tmp_${base}"
  cp "$src" "$tmp"

  python - "$tmp" "$dst" <<'PY'
from pathlib import Path
import sys

from onnxruntime.quantization import QuantType, quantize_dynamic

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

quantize_dynamic(
    model_input=src,
    model_output=dst,
    weight_type=QuantType.QUInt8,
    per_channel=False,
    reduce_range=False,
)
PY

  rm -f "$tmp" "${tmp%.onnx}-inferred.onnx"
}

quantize_model() {
  local backend="$1"
  local src="$2"
  local dst="$3"

  echo "Quantizing $(basename "$src") with $backend -> $dst"
  case "$backend" in
    shrink_ray)
      quantize_with_shrink_ray "$src" "$dst"
      ;;
    ort_dynamic)
      quantize_with_ort_dynamic "$src" "$dst"
      ;;
  esac
}

normalize_ort_names() {
  # Runtime optimization style emits *.with_runtime_opt.ort; the Android runtime
  # expects the stable Moonshine filenames.
  if [ -f "$ORT_DIR/encoder_model.with_runtime_opt.ort" ]; then
    mv "$ORT_DIR/encoder_model.with_runtime_opt.ort" "$ORT_DIR/encoder_model.ort"
  fi
  if [ -f "$ORT_DIR/decoder_model_merged.with_runtime_opt.ort" ]; then
    mv "$ORT_DIR/decoder_model_merged.with_runtime_opt.ort" "$ORT_DIR/decoder_model_merged.ort"
  fi
  if [ -f "$ORT_DIR/required_operators.with_runtime_opt.config" ]; then
    mv "$ORT_DIR/required_operators.with_runtime_opt.config" "$ORT_DIR/required_operators.config"
  fi
}

echo "Model: $MODEL"
echo "Output: $OUT"

# Convert HF safetensors checkpoint -> ONNX merged decoder. ORT conversion is
# skipped here because full-float ORT files are much larger than the quantized
# Android assets shipped for base-en.
python -u scripts/convert_for_deployment.py \
  --model "$MODEL" \
  --output "$OUT" \
  --skip-tokenizer-extension \
  --skip-embedding-resize \
  --skip-ort-conversion

# Optimum's ONNX export may omit tokenizer.json; the deployment copy still has it.
if [ ! -f "$ONNX_MERGED/tokenizer.json" ]; then
  cp "$OUT/model_resized/tokenizer.json" "$ONNX_MERGED/tokenizer.json"
fi

python "$MOONSHINE_REPO/scripts/convert_tokenizer.py" \
  "$ONNX_MERGED/tokenizer.json" \
  "$ONNX_MERGED/tokenizer.bin"

mkdir -p "$ONNX_QUANTIZED" "$ORT_DIR"
rm -f \
  "$ONNX_QUANTIZED/encoder_model.onnx" \
  "$ONNX_QUANTIZED/decoder_model_merged.onnx" \
  "$ONNX_QUANTIZED"/_tmp_*.onnx \
  "$ORT_DIR/encoder_model.ort" \
  "$ORT_DIR/decoder_model_merged.ort" \
  "$ORT_DIR/encoder_model.with_runtime_opt.ort" \
  "$ORT_DIR/decoder_model_merged.with_runtime_opt.ort" \
  "$ORT_DIR/required_operators.config" \
  "$ORT_DIR/required_operators.with_runtime_opt.config" \
  "$ORT_DIR/required_operators_and_types.config"

BACKEND=$(choose_quantization_backend)
echo "Quantization backend: $BACKEND"
if [ "$BACKEND" = "ort_dynamic" ]; then
  echo "Note: onnx-shrink-ray was not found; using ORT dynamic quantization fallback."
  echo "Install onnx-shrink-ray and set QUANTIZATION_BACKEND=shrink_ray for the closest match to shipped Moonshine assets."
fi

quantize_model "$BACKEND" "$ONNX_MERGED/encoder_model.onnx" "$ONNX_QUANTIZED/encoder_model.onnx"
quantize_model "$BACKEND" "$ONNX_MERGED/decoder_model_merged.onnx" "$ONNX_QUANTIZED/decoder_model_merged.onnx"

python -m onnxruntime.tools.convert_onnx_models_to_ort \
  "$ONNX_QUANTIZED" \
  --output_dir "$ORT_DIR" \
  --optimization_style "$ORT_OPTIMIZATION_STYLE" \
  --target_platform "$TARGET_PLATFORM"

normalize_ort_names

cp "$ONNX_MERGED/tokenizer.bin" "$ORT_DIR/tokenizer.bin"

echo
echo "Android-ready ORT bundle:"
ls -lh \
  "$ORT_DIR/encoder_model.ort" \
  "$ORT_DIR/decoder_model_merged.ort" \
  "$ORT_DIR/tokenizer.bin" \
  "$ORT_DIR/required_operators.config"
