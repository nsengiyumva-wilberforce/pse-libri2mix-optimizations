#!/usr/bin/env bash
# Run every experiment in experiements/, one at a time, on the first 3 free GPUs.
# --batch 16 is clips per GPU, so three cards train at batch 48.
# Each train.py saves its best dev checkpoint beside itself. After training,
# the same script evaluates that checkpoint on the full test set.
# When the queue finishes, best weights are copied to experiements/best_checkpoints/
# and the means are written to experiements/results_table.csv.
#
# Usage:
#   ./scripts/run_experiments.sh
#   ./scripts/run_experiments.sh table
#   EPOCHS=20 ./scripts/run_experiments.sh
#
# A finished experiment is one that already has evaluation_results_full.csv.
# A checkpoint with no table is evaluated and not trained again.
set -euo pipefail
cd "$(dirname "$0")/.."

LOSS="${LOSS:-si-sdr}"
EPOCHS="${EPOCHS:-120}"
PER_GPU_BATCH="${BATCH:-16}"
GPU_FREE_MIB="${GPU_FREE_MIB:-4096}"
MAX_GPUS=3
if ! [[ "${PER_GPU_BATCH}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH must be a positive integer (clips per GPU)." >&2
  exit 2
fi

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

write_table() {
  python - <<'PY'
import csv
import glob
import os
import shutil

root = "experiements"
dest = os.path.join(root, "best_checkpoints")
os.makedirs(dest, exist_ok=True)
rows = []
for train_py in sorted(glob.glob(os.path.join(root, "*", "train.py"))):
    folder = os.path.dirname(train_py)
    name = os.path.basename(folder)
    scores = os.path.join(folder, "evaluation_results_full.csv")
    weights = sorted(glob.glob(os.path.join(folder, "model_weights_*.keras")))
    checkpoint = weights[0] if weights else ""
    copied = ""
    if checkpoint:
        copied = os.path.join(dest, os.path.basename(checkpoint))
        shutil.copy2(checkpoint, copied)
    record = {
        "experiment": name,
        "n": "",
        "si_sdr": "",
        "pesq": "",
        "stoi": "",
        "si_sdri": "",
        "pesqi": "",
        "stoii": "",
        "checkpoint": copied,
    }
    if os.path.isfile(scores):
        with open(scores, newline="") as handle:
            data = list(csv.DictReader(handle))
        def mean(key):
            vals = []
            for item in data:
                text = item[key]
                if text == "" or text.lower() == "nan":
                    continue
                vals.append(float(text))
            if not vals:
                return ""
            return f"{sum(vals) / len(vals):.6f}"
        record["n"] = str(len(data))
        record["si_sdr"] = mean("enh_sisdr")
        record["pesq"] = mean("enh_pesq")
        record["stoi"] = mean("enh_stoi")
        record["si_sdri"] = mean("sisdr_i")
        record["pesqi"] = mean("pesq_i")
        record["stoii"] = mean("stoi_i")
    rows.append(record)

out_path = os.path.join(root, "results_table.csv")
fields = ["experiment", "n", "si_sdr", "pesq", "stoi", "si_sdri", "pesqi", "stoii", "checkpoint"]
with open(out_path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"{'experiment':<32} {'n':>5} {'SI-SDR':>8} {'PESQ':>7} {'STOI':>7} {'SI-SDRi':>8} {'PESQi':>7} {'STOIi':>7}")
for row in rows:
    def cell(key, width, digits):
        text = row[key]
        if text == "":
            return f"{'':>{width}}"
        return f"{float(text):{width}.{digits}f}"
    print(
        f"{row['experiment']:<32} {row['n']:>5} "
        f"{cell('si_sdr', 8, 2)} {cell('pesq', 7, 3)} {cell('stoi', 7, 3)} "
        f"{cell('si_sdri', 8, 2)} {cell('pesqi', 7, 3)} {cell('stoii', 7, 3)}"
    )
print(f"\nTable: {out_path}")
print(f"Checkpoints: {dest}")
PY
}

LIMIT_GPUS="${CUDA_VISIBLE_DEVICES:-}"
target="${1:-train}"
if [[ "${target}" == "table" ]]; then
  write_table
  exit 0
fi
if [[ "${target}" != "train" ]]; then
  echo "Usage: $0 [train|table]" >&2
  exit 2
fi

align_libcuda
mapfile -t gpus < <(free_gpus)
if ((${#gpus[@]} < MAX_GPUS)); then
  echo "Need ${MAX_GPUS} free GPUs under ${GPU_FREE_MIB} MiB. Found ${#gpus[@]}." >&2
  exit 1
fi
devices="$(printf '%s,' "${gpus[@]:0:MAX_GPUS}")"
devices="${devices%,}"
global_batch=$((PER_GPU_BATCH * MAX_GPUS))
echo "Queue on GPUs ${devices}: batch ${global_batch} (${PER_GPU_BATCH} per GPU x ${MAX_GPUS}), epochs ${EPOCHS}, loss ${LOSS}"

mapfile -t experiments < <(find experiements -mindepth 2 -maxdepth 2 -name train.py | sort)
if ((${#experiments[@]} == 0)); then
  echo "No experiements/*/train.py found." >&2
  exit 1
fi

for train_py in "${experiments[@]}"; do
  folder="$(dirname "${train_py}")"
  name="$(basename "${folder}")"
  scores="${folder}/evaluation_results_full.csv"
  if [[ -f "${scores}" ]]; then
    echo "=== ${name}: scores already exist, skipping ==="
    continue
  fi
  shopt -s nullglob
  weights=("${folder}"/model_weights_*.keras)
  shopt -u nullglob
  if ((${#weights[@]} == 0)); then
    echo "=== ${name}: train loss=${LOSS} epochs=${EPOCHS} batch=${global_batch} gpus=${devices} ==="
    set +e
    CUDA_VISIBLE_DEVICES="${devices}" PYTHONUNBUFFERED=1 \
      python "${train_py}" --l "${LOSS}" --epochs "${EPOCHS}" --batch "${PER_GPU_BATCH}" \
      > "${folder}/train.log" 2>&1
    train_status=$?
    set -e
    echo "${train_status}" > "${folder}/train_exit"
    if ((train_status != 0)); then
      echo "=== ${name}: training failed (exit ${train_status}). See ${folder}/train.log ===" >&2
      continue
    fi
  else
    echo "=== ${name}: checkpoint present, evaluating only ==="
  fi
  echo "=== ${name}: evaluate best checkpoint ==="
  set +e
  CUDA_VISIBLE_DEVICES="${devices}" PYTHONUNBUFFERED=1 \
    python "${train_py}" > "${folder}/eval.log" 2>&1
  eval_status=$?
  set -e
  echo "${eval_status}" > "${folder}/eval_exit"
  if ((eval_status != 0)); then
    echo "=== ${name}: evaluation failed (exit ${eval_status}). See ${folder}/eval.log ===" >&2
    continue
  fi
  echo "=== ${name}: done ==="
done

write_table
