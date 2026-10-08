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
#   ./scripts/speaker_dilution.sh early
#   EPOCHS=20 ./scripts/speaker_dilution.sh late
#   ./scripts/speaker_dilution.sh series
set -euo pipefail
cd "$(dirname "$0")/.."

LOSS="${LOSS:-si-sdr}"
EPOCHS="${EPOCHS:-200}"

run() {
  echo "=== inject=$1 loss=${LOSS} epochs=${EPOCHS} ==="
  python train.py --l "${LOSS}" --full-utterance --inject "$1" --epochs "${EPOCHS}"
}

target="${1:-}"
case "${target}" in
  none|early|bottleneck|decoder|late|all)
    run "${target}"
    ;;
  series)
    for name in none early bottleneck decoder late all; do
      run "${name}"
    done
    ;;
  *)
    echo "Usage: $0 {none|early|bottleneck|decoder|late|all|series}" >&2
    exit 2
    ;;
esac
