# Libri2Mix personalized enhancement

A U-Net mask estimator for target-speaker extraction on Libri2Mix. The frontend matches SEF-PNet: 8 kHz, 256-point square-root Hann, 32 ms window, 8 ms hop, power-law compression on the complex STFT. The loss is scored on the waveform.

Channel gates are [ECA](https://github.com/BangguWu/ECANet) (global pool and a small 1D convolution). Two aligned maps are mixed with [ASFF](https://github.com/GOATmessi8/ASFF): a per-bin softmax over a pair of 1×1 scores. That mix starts equal because the score convolution is zero.

## Data

Each split is three Kaldi-style scripts:

```text
data/{train,dev,test}/mix_clean.scp   # mixture
data/{train,dev,test}/auxs1.scp       # enrollment
data/{train,dev,test}/ref.scp         # target
```

## Train and evaluate

Training uses the full enrollment utterance. The cue is early fusion: frame similarity mixed with the enrollment mean, concatenated onto the mixture. The default schedule is 200 epochs. Adam starts at `5e-4`. Each GPU holds 16 clips, so one GPU trains at batch 16 and three GPUs train at batch 48.

```bash
./scripts/speaker_dilution.sh
EPOCHS=20 ./scripts/speaker_dilution.sh
BATCH=8 ./scripts/speaker_dilution.sh
```

`BATCH` is clips per GPU and defaults to 16. One GPU then trains at 8. Three GPUs train at 24.

`--l` is one of `time-mse`, `stsa-mse`, `stoi`, `estoi`, `si-sdr`, `pmsqe`. Omit it to evaluate the saved `si-sdr` checkpoint. Scores are SI-SDR, PESQ, and STOI, written to `evaluation_results_full.csv`. Paste a summary into `RESULTS.MD`.

```bash
python train.py
```

`LOSS` defaults to `si-sdr`. `EPOCHS` defaults to 200. `gpus` only prints which cards are free. Training takes the first 3 free cards, or the single free card on a one-GPU machine. A card counts as free when it has no compute process and is using under `GPU_FREE_MIB` MiB (default 4096). The script also prefers the host `libcuda` over the CUDA compat library, which these 4090s reject when it is newer than driver 535.

## Checkpoints

The file name ends in `_inject_early.keras`. Evaluation loads the `si-sdr` checkpoint with that suffix.
