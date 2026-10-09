#!/usr/bin/env bash
# Where the enrollment cue enters the U-Net.
#
# Claim: concatenating the guided spectrogram onto the mixture at the input
# lets the encoder pool that cue away. Injecting the same cue at the bottleneck
# and in the decoder should beat that early concatenate.
#
# Every run uses the same guided STFT, loss, data, and full-utterance trainer.
# The only change is --inject.
#
#   none         enrollment is ignored
#   early        input concatenate only                 (the diluted case)
#   bottleneck   ASFF after the xLSTM only
#   decoder      concatenate at each decoder stage only
#   late         bottleneck + decoder                   (the proposed injection)
#   all          early + bottleneck + decoder           (current --stft-interact graph)
#
# Dev SI-SDR that supports the claim:
#   none < early < late, and late >= all.
# decoder above early isolates depth, because both use concatenation.
# bottleneck above early mixes a better fusion operator into the comparison.
#
# Usage:
#   ./scripts/speaker_dilution.sh gpus
#   ./scripts/speaker_dilution.sh early
#   EPOCHS=20 ./scripts/speaker_dilution.sh late
#   ./scripts/speaker_dilution.sh series
#
# series places one condition on each free GPU and queues the rest.
# A card is free when it has no compute process and is under GPU_FREE_MIB
# MiB (default 4096). Logs are logs/inject_<name>.log.
set -euo pipefail
cd "$(dirname "$0")/.."

LOSS="${LOSS:-si-sdr}"
EPOCHS="${EPOCHS:-200}"
GPU_FREE_MIB="${GPU_FREE_MIB:-4096}"
LOG_DIR="${LOG_DIR:-logs}"
CONDITIONS=(none early bottleneck decoder late all)

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

run_foreground() {
  echo "=== inject=$1 loss=${LOSS} epochs=${EPOCHS} ==="
  python train.py --l "${LOSS}" --full-utterance --inject "$1" --epochs "${EPOCHS}"
}

launch() {
  local gpu="$1" name="$2" log
  mkdir -p "${LOG_DIR}"
  log="${LOG_DIR}/inject_${name}.log"
  echo "GPU ${gpu}: inject=${name} loss=${LOSS} epochs=${EPOCHS} -> ${log}" >&2
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
    python train.py --l "${LOSS}" --full-utterance --inject "${name}" --epochs "${EPOCHS}" \
    >"${log}" 2>&1 &
  LAUNCH_PID=$!
}

run_series() {
  local -a gpus=() queue=("${CONDITIONS[@]}")
  local -A pid_gpu=() pid_name=()
  local failed=0 next=0 name gpu pid
  mapfile -t gpus < <(free_gpus)
  if ((${#gpus[@]} == 0)); then
    echo "No free GPU under ${GPU_FREE_MIB} MiB." >&2
    exit 1
  fi
  echo "Scheduling ${#queue[@]} conditions on ${#gpus[@]} free GPU(s)."

  while (( next < ${#queue[@]} )) || ((${#pid_gpu[@]})); do
    if ((${#pid_gpu[@]})); then
      local -a finished=()
      for pid in "${!pid_gpu[@]}"; do
        if ! kill -0 "${pid}" 2>/dev/null; then
          finished+=("${pid}")
        fi
      done
      for pid in "${finished[@]+"${finished[@]}"}"; do
        gpu="${pid_gpu[${pid}]}"
        name="${pid_name[${pid}]}"
        if wait "${pid}"; then
          echo "done inject=${name} on GPU ${gpu}"
        else
          echo "FAILED inject=${name} on GPU ${gpu}. See ${LOG_DIR}/inject_${name}.log" >&2
          failed=$((failed + 1))
        fi
        unset "pid_gpu[${pid}]" "pid_name[${pid}]"
        gpus+=("${gpu}")
      done
    fi
    while (( next < ${#queue[@]} && ${#gpus[@]} )); do
      name="${queue[${next}]}"
      next=$((next + 1))
      gpu="${gpus[0]}"
      if ((${#gpus[@]} > 1)); then
        gpus=("${gpus[@]:1}")
      else
        gpus=()
      fi
      launch "${gpu}" "${name}"
      pid_gpu["${LAUNCH_PID}"]="${gpu}"
      pid_name["${LAUNCH_PID}"]="${name}"
    done
    if ((${#pid_gpu[@]})); then
      sleep 15
    fi
  done
  if ((failed)); then
    echo "${failed} condition(s) failed." >&2
    exit 1
  fi
  echo "All ${#CONDITIONS[@]} conditions finished."
}

LIMIT_GPUS="${CUDA_VISIBLE_DEVICES:-}"
align_libcuda

target="${1:-}"
case "${target}" in
  gpus)
    free_gpus >/dev/null
    ;;
  none|early|bottleneck|decoder|late|all)
    run_foreground "${target}"
    ;;
  series)
    run_series
    ;;
  *)
    echo "Usage: $0 {gpus|none|early|bottleneck|decoder|late|all|series}" >&2
    exit 2
    ;;
esac
