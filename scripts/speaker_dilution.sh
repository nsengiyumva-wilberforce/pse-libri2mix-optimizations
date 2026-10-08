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
# series puts one condition on each free GPU and starts the next when a card
# finishes. A GPU is free when it has no compute process and memory.used is
# below GPU_FREE_MIB (default 2048). Desktop processes sit under that.
#
# Usage:
#   ./scripts/speaker_dilution.sh gpus
#   ./scripts/speaker_dilution.sh early
#   EPOCHS=20 ./scripts/speaker_dilution.sh late
#   ./scripts/speaker_dilution.sh series
set -euo pipefail
cd "$(dirname "$0")/.."

LOSS="${LOSS:-si-sdr}"
EPOCHS="${EPOCHS:-200}"
GPU_FREE_MIB="${GPU_FREE_MIB:-2048}"
LOG_DIR="${LOG_DIR:-logs}"
CONDITIONS=(none early bottleneck decoder late all)

# Print one free GPU index per line. Status lines go to stderr.
free_gpus() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is not on PATH." >&2
    return 1
  fi
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi cannot see a GPU." >&2
    return 1
  fi

  local busy idx uuid used name allowed p
  busy="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits 2>/dev/null || true)"

  local -a limit=()
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra limit <<< "${CUDA_VISIBLE_DEVICES}"
  fi

  while IFS=, read -r idx uuid used; do
    idx="${idx//[[:space:]]/}"
    uuid="${uuid//[[:space:]]/}"
    used="${used//[[:space:]]/}"
    [[ -n "${idx}" ]] || continue
    if ((${#limit[@]})); then
      allowed=0
      for p in "${limit[@]}"; do
        [[ "${p//[[:space:]]/}" == "${idx}" ]] && allowed=1
      done
      ((allowed)) || continue
    fi
    name="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${idx}" | head -1)"
    if printf '%s\n' "${busy}" | grep -Fxq "${uuid}"; then
      echo "GPU ${idx} (${name}): busy, compute process, ${used} MiB" >&2
      continue
    fi
    if ((used >= GPU_FREE_MIB)); then
      echo "GPU ${idx} (${name}): busy, ${used} MiB used" >&2
      continue
    fi
    echo "GPU ${idx} (${name}): free, ${used} MiB" >&2
    printf '%s\n' "${idx}"
  done < <(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits)
}

launch() {
  local name="$1"
  local gpu="$2"
  local log="${LOG_DIR}/inject_${name}.log"
  mkdir -p "${LOG_DIR}"
  echo "GPU ${gpu}: inject=${name} loss=${LOSS} epochs=${EPOCHS} -> ${log}" >&2
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
    python train.py --l "${LOSS}" --full-utterance --inject "${name}" --epochs "${EPOCHS}" \
    > "${log}" 2>&1 &
  echo $!
}

run_one() {
  local name="$1"
  local -a gpus=()
  mapfile -t gpus < <(free_gpus)
  if ((${#gpus[@]} == 0)); then
    echo "No free GPU. Raise GPU_FREE_MIB or wait until a card has no compute process." >&2
    exit 1
  fi
  local pid
  pid="$(launch "${name}" "${gpus[0]}")"
  local status=0
  wait "${pid}" || status=$?
  if ((status != 0)); then
    echo "inject=${name} failed. See ${LOG_DIR}/inject_${name}.log" >&2
    exit "${status}"
  fi
}

run_series() {
  local -a gpus=()
  mapfile -t gpus < <(free_gpus)
  if ((${#gpus[@]} == 0)); then
    echo "No free GPU. Raise GPU_FREE_MIB or wait until a card has no compute process." >&2
    exit 1
  fi
  echo "Scheduling ${#CONDITIONS[@]} conditions on ${#gpus[@]} free GPU(s)."

  local -A pid_gpu=() pid_name=()
  local -a idle=("${gpus[@]}")
  local next=0 failures=0

  while ((next < ${#CONDITIONS[@]})) || ((${#pid_gpu[@]} > 0)); do
    while ((next < ${#CONDITIONS[@]} && ${#idle[@]} > 0)); do
      local gpu="${idle[0]}"
      if ((${#idle[@]} > 1)); then
        idle=("${idle[@]:1}")
      else
        idle=()
      fi
      local name="${CONDITIONS[$next]}"
      next=$((next + 1))
      local pid
      pid="$(launch "${name}" "${gpu}")"
      pid_gpu["${pid}"]="${gpu}"
      pid_name["${pid}"]="${name}"
    done

    ((${#pid_gpu[@]} == 0)) && break

    local pid status
    local -a pids=()
    for pid in "${!pid_gpu[@]}"; do
      pids+=("${pid}")
    done
    local still=0
    for pid in "${pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        still=1
        continue
      fi
      status=0
      wait "${pid}" || status=$?
      if ((status == 0)); then
        echo "finished inject=${pid_name[$pid]} on GPU ${pid_gpu[$pid]}"
      else
        echo "FAILED inject=${pid_name[$pid]} on GPU ${pid_gpu[$pid]}. See ${LOG_DIR}/inject_${pid_name[$pid]}.log" >&2
        failures=$((failures + 1))
      fi
      idle+=("${pid_gpu[$pid]}")
      unset "pid_gpu[${pid}]" "pid_name[${pid}]"
    done
    if ((still && (${#idle[@]} == 0 || next >= ${#CONDITIONS[@]}))); then
      sleep 15
    fi
  done

  if ((failures > 0)); then
    echo "${failures} condition(s) failed." >&2
    exit 1
  fi
}

target="${1:-}"
case "${target}" in
  gpus)
    mapfile -t _free < <(free_gpus)
    echo "Free: ${#_free[@]}"
    ;;
  none|early|bottleneck|decoder|late|all)
    run_one "${target}"
    ;;
  series)
    run_series
    ;;
  *)
    echo "Usage: $0 {gpus|none|early|bottleneck|decoder|late|all|series}" >&2
    exit 2
    ;;
esac
