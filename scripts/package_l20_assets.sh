#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: package_l20_assets.sh --model-dir DIR --repo-dir DIR --commit COMMIT --output-dir DIR [--split-size 2G]

Create an exact-commit ColaCare archive, a validated model archive, 2GB-style parts,
and SHA256 manifests. This script does not download anything.
EOF
}

MODEL_DIR=""
REPO_DIR=""
COMMIT=""
OUTPUT_DIR=""
SPLIT_SIZE="2G"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir) MODEL_DIR="${2:-}"; shift 2 ;;
    --repo-dir) REPO_DIR="${2:-}"; shift 2 ;;
    --commit) COMMIT="${2:-}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:-}"; shift 2 ;;
    --split-size) SPLIT_SIZE="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for value_name in MODEL_DIR REPO_DIR COMMIT OUTPUT_DIR; do
  if [[ -z "${!value_name}" ]]; then
    echo "Missing required argument for ${value_name}" >&2
    usage >&2
    exit 2
  fi
done

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Model directory not found: $MODEL_DIR" >&2
  exit 2
fi
if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "Git repository not found: $REPO_DIR" >&2
  exit 2
fi

MODEL_DIR="$(cd "$MODEL_DIR" && pwd -P)"
REPO_DIR="$(cd "$REPO_DIR" && pwd -P)"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd -P)"

case "$OUTPUT_DIR/" in
  "$MODEL_DIR/"*) echo "Output directory must not be inside the model directory" >&2; exit 2 ;;
esac

for command_name in git tar split sha256sum find awk; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command is unavailable: $command_name" >&2
    exit 2
  fi
done

RESOLVED_COMMIT="$(git -C "$REPO_DIR" rev-parse --verify "${COMMIT}^{commit}")" || {
  echo "Git commit does not exist: $COMMIT" >&2
  exit 2
}
SHORT_COMMIT="${RESOLVED_COMMIT:0:12}"

TRACKED_FILES="$(git -C "$REPO_DIR" ls-tree -r --name-only "$RESOLVED_COMMIT")"
FORBIDDEN_TRACKED="$(printf '%s\n' "$TRACKED_FILES" | grep -Ei '^(\.env$|models?/|data/|datasets/|artifacts/|patient_outputs/|logs?/|response/|output/|wheels?/)|\.(safetensors|gguf|ckpt|pth|pt|onnx|h5|hdf5|tar|tar\.gz|zip)$' || true)"
if [[ -n "$FORBIDDEN_TRACKED" ]]; then
  echo "Refusing to package commit with forbidden tracked assets:" >&2
  printf '%s\n' "$FORBIDDEN_TRACKED" >&2
  exit 2
fi
LARGE_TRACKED="$(git -C "$REPO_DIR" ls-tree -r -l "$RESOLVED_COMMIT" | awk '$4 ~ /^[0-9]+$/ && $4 > 52428800 {print $0}')"
if [[ -n "$LARGE_TRACKED" ]]; then
  echo "Refusing to package tracked files larger than 50 MiB:" >&2
  printf '%s\n' "$LARGE_TRACKED" >&2
  exit 2
fi

for required in config.json tokenizer_config.json MODEL_MANIFEST.json SHA256SUMS; do
  if [[ ! -s "$MODEL_DIR/$required" ]]; then
    echo "Model snapshot is missing required file: $required" >&2
    exit 2
  fi
done
if ! find "$MODEL_DIR" -maxdepth 1 -type f -name '*.safetensors' -size +0c -print -quit | grep -q .; then
  echo "Model snapshot contains no non-empty safetensors weights" >&2
  exit 2
fi
(cd "$MODEL_DIR" && sha256sum -c SHA256SUMS)

MODEL_NAME="$(basename "$MODEL_DIR")"
CODE_ARCHIVE="colacare-${SHORT_COMMIT}.tar.gz"
MODEL_ARCHIVE="${MODEL_NAME}.tar.gz"
MANIFEST="OFFLINE_ASSET_MANIFEST.txt"
CHECKSUMS="OFFLINE_ASSET_SHA256SUMS"

for target in "$OUTPUT_DIR/$CODE_ARCHIVE" "$OUTPUT_DIR/$MODEL_ARCHIVE" "$OUTPUT_DIR/$MANIFEST" "$OUTPUT_DIR/$CHECKSUMS"; do
  if [[ -e "$target" ]]; then
    echo "Refusing to overwrite existing output: $target" >&2
    exit 2
  fi
done
if compgen -G "$OUTPUT_DIR/${MODEL_ARCHIVE}.part-*" >/dev/null; then
  echo "Refusing to overwrite existing model parts in $OUTPUT_DIR" >&2
  exit 2
fi

git -C "$REPO_DIR" archive \
  --format=tar.gz \
  --prefix=ColaCare/ \
  --output="$OUTPUT_DIR/$CODE_ARCHIVE" \
  "$RESOLVED_COMMIT"

tar \
  --exclude="$MODEL_NAME/.cache" \
  --exclude="$MODEL_NAME/.cache/*" \
  -C "$(dirname "$MODEL_DIR")" \
  -czf "$OUTPUT_DIR/$MODEL_ARCHIVE" \
  "$MODEL_NAME"

split -b "$SPLIT_SIZE" -d -a 3 \
  "$OUTPUT_DIR/$MODEL_ARCHIVE" \
  "$OUTPUT_DIR/${MODEL_ARCHIVE}.part-"

PART_FILES=("$OUTPUT_DIR/${MODEL_ARCHIVE}.part-"*)
if [[ ! -e "${PART_FILES[0]}" ]]; then
  echo "Model archive splitting produced no parts" >&2
  exit 2
fi

cat >"$OUTPUT_DIR/$MANIFEST" <<EOF
FORMAT_VERSION=1
CREATED_AT_UTC=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
COLACARE_COMMIT=$RESOLVED_COMMIT
CODE_ARCHIVE=$CODE_ARCHIVE
MODEL_DIRECTORY=$MODEL_NAME
MODEL_ARCHIVE=$MODEL_ARCHIVE
MODEL_PART_PREFIX=${MODEL_ARCHIVE}.part-
MODEL_PART_COUNT=${#PART_FILES[@]}
SPLIT_SIZE=$SPLIT_SIZE
CHECKSUM_FILE=$CHECKSUMS
EOF

(
  cd "$OUTPUT_DIR"
  RELATIVE_PARTS=()
  for part in "${PART_FILES[@]}"; do
    RELATIVE_PARTS+=("$(basename "$part")")
  done
  sha256sum "$CODE_ARCHIVE" "$MODEL_ARCHIVE" "${RELATIVE_PARTS[@]}" "$MANIFEST" >"$CHECKSUMS"
)

echo "Offline assets created successfully"
echo "COLACARE_COMMIT=$RESOLVED_COMMIT"
echo "CODE_ARCHIVE=$OUTPUT_DIR/$CODE_ARCHIVE"
echo "MODEL_ARCHIVE=$OUTPUT_DIR/$MODEL_ARCHIVE"
echo "MODEL_PART_COUNT=${#PART_FILES[@]}"
echo "MANIFEST=$OUTPUT_DIR/$MANIFEST"
echo "CHECKSUMS=$OUTPUT_DIR/$CHECKSUMS"

