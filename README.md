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

Training uses the full enrollment utterance. The default schedule is 200 epochs. Adam starts at `5e-4`.

```bash
python train.py --l si-sdr --full-utterance
```

`--l` is one of `time-mse`, `stsa-mse`, `stoi`, `estoi`, `si-sdr`, `pmsqe`. Omit it to evaluate the saved `si-sdr` checkpoint. Scores are SI-SDR, PESQ, and STOI, written to `evaluation_results_full.csv`. Paste a summary into `RESULTS.MD`.

```bash
python train.py --full-utterance
```

Pass the same `--inject` or `--stft-interact` flag you trained with, or the loader looks for a different file.

## Where the enrollment cue goes

The default model injects a speaker embedding with FiLM and enrollment cross-attention at the bottleneck and in the decoder.

The dilution series uses one guided STFT (frame similarity mixed with the enrollment mean) and changes only the insertion point. Concatenating that map onto the mixture at the input lets the encoder pool the cue away. Putting the same map back at the bottleneck and in the decoder is the alternative.

| `--inject` | Cue |
|---|---|
| `none` | Ignored. The enrollment input stays in the graph and adds nothing. |
| `early` | Concatenated onto the mixture. This is the diluted case. |
| `bottleneck` | ASFF into the bottleneck map, then the xLSTM. |
| `decoder` | Concatenated at each decoder stage. |
| `late` | Bottleneck and decoder. This is the proposed injection. |
| `all` | Input, bottleneck, and decoder. |

`--stft-interact` is the `all` graph with the older checkpoint name.

`decoder` against `early` holds the operator fixed at concatenation, so a gain there is depth. `bottleneck` against `early` also swaps in ASFF. On dev SI-SDR, the claim holds when `none < early < late` and `late` is at least as high as `all`. The six graphs are about 6.61M to 6.64M parameters.

```bash
./scripts/speaker_dilution.sh early
EPOCHS=20 ./scripts/speaker_dilution.sh late
./scripts/speaker_dilution.sh series
```

`LOSS` defaults to `si-sdr`. `EPOCHS` defaults to 200. `gpus` only prints which cards are free. A single condition stays in this terminal. `series` runs `none`, `early`, `bottleneck`, `decoder`, `late`, then `all`, one condition per free GPU, and queues the rest. A card counts as free when it has no compute process and is using under `GPU_FREE_MIB` MiB (default 4096). Epoch lines for a series are in `logs/inject_<name>.log`. The script also prefers the host `libcuda` over the CUDA compat library, which these 4090s reject when it is newer than driver 535.

## Checkpoints

Files are named from the loss, then the run suffixes. An injection run ends in `_inject_<name>.keras`, for example `_inject_late.keras`. `--stft-interact` without `--inject` keeps `_stft` in the name and does not add `_inject_all`. Each condition writes its own file.
