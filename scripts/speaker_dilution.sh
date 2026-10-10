#!/usr/bin/env bash
# Early fusion: the guided spectrogram is concatenated onto the mixture.
# One process. Each GPU holds 16 clips. Three free cards train at batch 48.
# One free card trains at batch 16.
#
# Usage:
#   ./scripts/speaker_dilution.sh
#   ./scripts/speaker_dilution.sh gpus
#   EPOCHS=20 ./scripts/speaker_dilution.sh
#   BATCH=8 ./scripts/speaker_dilution.sh
#
# gpus prints which cards are free. A card is free when it has no compute
# process and is under GPU_FREE_MIB MiB (default 4096). The job uses the first
# 3 free cards, or the one free card when that is all there is. LOSS defaults
# to si-sdr. EPOCHS defaults to 120. BATCH is clips per GPU and defaults to 16,
# so one card trains at 16 and three cards train at 48.
set -euo pipefail
cd "$(dirname "$0")/.."

LOSS="${LOSS:-si-sdr}"
EPOCHS="${EPOCHS:-120}"
GPU_FREE_MIB="${GPU_FREE_MIB:-4096}"
MAX_GPUS=3
PER_GPU_BATCH="${BATCH:-16}"
if ! [[ "${PER_GPU_BATCH}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH must be a positive integer (clips per GPU)." >&2
  exit 2
fi

# GeForce rejects the newer CUDA compat library. Point libcuda.so.1 at the
# driver build (libcuda.so.535.274.02, not libcuda.so.535).
align_libcuda() {
  local major="" lib="" dir="" filtered="" part
  major="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | awk -F. 'NR==1{print $1}')"
  [[ -n "${major}" ]] || {
    echo "nvidia-smi did not report a driver, so TensorFlow cannot see the GPUs." >&2
    return 1
  }
  IFS=':' read -ra parts <<< "${LD_LIBRARY_PATH:-}"
  for part in "${parts[@]}"; do
    [[ -z "${part}" || "${part}" == *"/cuda/compat"* ]] && continue
    filtered="${filtered:+${filtered}:}${part}"
  done
  local -a matches=()
  shopt -s nullglob
  matches=(
    /usr/lib/x86_64-linux-gnu/libcuda.so."${major}"*
    /usr/lib64/libcuda.so."${major}"*
    /usr/local/nvidia/lib64/libcuda.so."${major}"*
    /usr/lib/wsl/lib/libcuda.so."${major}"*
  )
  shopt -u nullglob
  if ((${#matches[@]} == 0)); then
    mapfile -t matches < <(find /usr/lib /usr/lib64 /usr/local /opt -name "libcuda.so.${major}*" -type f 2>/dev/null)
  fi
  if ((${#matches[@]} == 0)); then
    echo "libcuda: no libcuda.so.${major}* found. TensorFlow would stay on the CPU." >&2
    return 1
  fi
  lib="${matches[0]}"
  dir="$(mktemp -d)"
  ln -s "${lib}" "${dir}/libcuda.so.1"
  export LD_LIBRARY_PATH="${dir}${filtered:+:${filtered}}"
  echo "libcuda: ${lib}"
}

gpu_allowed() {
  local want="$1" id
  [[ -z "${LIMIT_GPUS:-}" ]] && return 0
  IFS=',' read -ra ids <<< "${LIMIT_GPUS}"
  for id in "${ids[@]}"; do
    id="${id// /}"
    [[ "${id}" == "${want}" ]] && return 0
  done
  return 1
}

# Status on stderr, free indices on stdout.
free_gpus() {
  local line index uuid used name
  declare -A busy=()
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is not on PATH." >&2
    return 1
  fi
  while IFS= read -r uuid; do
    uuid="${uuid// /}"
    [[ -z "${uuid}" || "${uuid}" == "gpu_uuid" ]] && continue
    busy["${uuid}"]=1
  done < <(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null || true)

  while IFS=',' read -r index uuid used name; do
    index="${index// /}"
    uuid="${uuid// /}"
    used="${used// /}"
    name="${name#"${name%%[![:space:]]*}"}"
    name="${name%"${name##*[![:space:]]}"}"
    gpu_allowed "${index}" || continue
    if [[ -n "${busy[${uuid}]:-}" || "${used}" -ge "${GPU_FREE_MIB}" ]]; then
      echo "GPU ${index} (${name}): busy, ${used} MiB" >&2
      continue
    fi
    echo "GPU ${index} (${name}): free, ${used} MiB" >&2
    printf '%s\n' "${index}"
  done < <(nvidia-smi --query-gpu=index,uuid,memory.used,name --format=csv,noheader,nounits)
}

run_early() {
  local -a gpus=()
  local devices="" n batch
  mapfile -t gpus < <(free_gpus)
  if ((${#gpus[@]} == 0)); then
    echo "No free GPU under ${GPU_FREE_MIB} MiB." >&2
    exit 1
  fi
  n=${#gpus[@]}
  if ((n > MAX_GPUS)); then
    n=$MAX_GPUS
  fi
  devices="$(printf '%s,' "${gpus[@]:0:n}")"
  devices="${devices%,}"
  batch=$((PER_GPU_BATCH * n))
  echo "=== early fusion loss=${LOSS} epochs=${EPOCHS} batch=${batch} (${PER_GPU_BATCH} per GPU x ${n}) gpus=${devices} ==="
  CUDA_VISIBLE_DEVICES="${devices}" PYTHONUNBUFFERED=1 \
    python train.py --l "${LOSS}" --epochs "${EPOCHS}" --batch "${PER_GPU_BATCH}"
}

LIMIT_GPUS="${CUDA_VISIBLE_DEVICES:-}"
align_libcuda

target="${1:-train}"
case "${target}" in
  gpus)
    free_gpus >/dev/null
    ;;
  train)
    run_early
    ;;
  *)
    echo "Usage: $0 [train|gpus]" >&2
    echo "Early fusion only. BATCH is clips per GPU (default 16)." >&2
    exit 2
    ;;
esac
