# Experiment 1: implementation notes

How [`exp1-preanalysis.md`](exp1-preanalysis.md) maps to code, which unstated details were fixed and how, and what is still open. Everything that departs from the plan's text is in [`exp1-deviations.md`](exp1-deviations.md).

## Still open (blocks Stage 1)

| # | What | Status |
|---|---|---|
| 1 | **reBEN data.** | Loader done (`src/data/reben.py`); tiers not yet built. It reads the [LMDB published by the reBEN authors](https://huggingface.co/datasets/hackelle/BigEarthNetV2-LMDB) (S1 and S2 already paired, keyed by patch name; same content as [Zenodo record 10891137](https://zenodo.org/records/10891137), v2.0.0). Uses the official `metadata.parquet`: 480,038 patches (237,871 train / 122,342 val / 119,825 test), snow/cloud/shadow already excluded. Preparation runs **only on an HF Job** (`scripts/prepare_hf.sh`), never locally; see "Building the data" below. |
| 2 | **Backbone.** | Decided: the `ConvEncoder` CNN for Experiment 1, random init shared across arms. |
| 3 | **`temporal` as a spatial offset.** | Confirmed: the description says one S1 patch "for each Sentinel-2 patch", taken at the closest timestamp. The §7.6 fallback is the only option. Patch names end `_<H-order>_<V-order>` within the S2 tile, which `grid_position` parses. |
| 4 | **BIFOLD reference numbers** for pre-flight check 4. | ResNet-18, v0.2.0 model cards, test macro-F1: [S2-only](https://huggingface.co/BIFOLD-BigEarthNetv2-0/resnet18-s2-v0.2.0) 0.644, [S1-only](https://huggingface.co/BIFOLD-BigEarthNetv2-0/resnet18-s1-v0.2.0) 0.566, [S1+S2](https://huggingface.co/BIFOLD-BigEarthNetv2-0/resnet18-all-v0.2.0) 0.659. Pass `--ref opt=0.644 --ref sar=0.566`. |
| 5 | **Training schedule.** `max_epochs` (15 pilot / 30 full), lr 1e-3 and batch size 64 are placeholders. Set once, identical for every arm. | Open. For comparison, BIFOLD used AdamW, lr 1e-3, warmup + cosine, early stopping; best epochs 23–28. |

## Arms (§2.1)

All six are one class, `src/models/fusion_net.py:FusionNet`; only `modalities` and the `fusion` config differ (`configs/model/<arm>.yaml`).

| Arm | Fusion | Structure |
|---|---|---|
| `opt`, `sar` | `EarlyFusion` over one modality | one encoder → GAP → linear head |
| `early` | `EarlyFusion` | 14-band input → one encoder → head |
| `late` | `LateFusion` | encoder + head per modality; logits mixed by `softmax(w)`, one learned weight per modality shared by all samples |
| `gate_global` | `GlobalGateFusion` | encoder per modality → GAP → gate MLP → one weight per modality per sample |
| `gate_spatial` | `SpatialGateFusion` | same gate as 1×1 convs → one weight per modality per token → GAP |

- **Identical initialisation (§6).** `ConvEncoder` is always built over all 14 bands, initialised from the training seed with a private generator (or loaded from `pretrained`), and only then is its stem sliced to its bands. For one seed, every arm's encoders start from identical weights; tests assert this. Fusion layers and heads are initialised separately.
- **Gate arms are parameter-matched.** The global and spatial gates are the same MLP, applied to pooled features or per token, so their parameter counts are equal. A granularity effect cannot be a capacity effect, which already controls for the §5 "spatial wins on capacity" row.
- **Dropped modality.** `late` and the gates give a missing (None) modality zero weight and renormalise. `early` and the unimodal arms see zeros, i.e. the training mean after standardisation. An arm whose only input is gone predicts from a zero image.
- **Diagnostics.** `late` and the gates record their modality weights. Every result row stores the mean weight per modality in `modality_weights`. Use it to check that the gate does what H3 claims, e.g. the optical weight falls as cloud rises.

## Degradations (§2.2)

All run on standardised test batches, on CPU, from a generator seeded only by `degradation_seed`.

| Name | Severity | Method |
|---|---|---|
| `cloud` | exact fraction of the patch occluded | Value-noise field (Gaussian grids of 2/4/8 cells, bilinear, amplitude halving per octave), top-k pixels per sample, so the fraction is exact and the mask is coherent blobs. All optical bands inside the mask are set to 0, the training mean: total information loss, not a bright cloud signature. |
| `speckle` | s → L = 1/s looks (grid → L = 4, 2, 1.33, 1) | `n ~ Gamma(L, 1/L)` (mean 1, variance 1/L = s), multiplied into SAR intensity. Data are dB, so it is applied as `+10·log10(n)` scaled by each band's std. Per-pixel independent; it adds to the product's existing speckle. |
| `drop_optical`, `drop_sar` | {0, 1} | the modality becomes None |
| `temporal` | grid steps {1, 2, 4, 8} | SAR replaced by the SAR of the patch k cells away in the same tile (k × 1.2 km). Precomputed into test shards as `sar_shift_<k>` from the full source. The manifest records how many neighbours were exact vs nearest available. |

## Data

- `prepare_data` writes `band_stats` (per-band mean/std of the tier's train split) and `temporal` metadata into each tier's `manifest.json`. Test shards carry `sar_shift_<k>`.
- `StreamingDataModule` standardises every band with those stats (`normalize: true`) and applies random flips to the train split only (`augment`, on in the experiment configs).

## Building the data (remote only)

```bash
git push                                   # the job runs the pushed commit
./scripts/prepare_hf.sh <user>/reben --dry-run
./scripts/prepare_hf.sh <user>/reben       # -> private datasets <user>/reben-{dev,sweep,full}
```

On a `cpu-xl` job (16 vCPU, 124 GB RAM, 1 TB disk, about $1/h), the job:
1. Downloads the LMDB (155 GB) to the job's disk.
2. Samples each tier within the official splits, keeping the per-class floor.
3. Reads patches lazily and resamples the 20 m and 60 m S2 bands bilinearly to 120 × 120. S2 is stored as uint16 and S1 as float32 dB (non-finite dB clamped to -50/+30).
4. Writes litdata shards with 16 workers and pushes each tier with a resumable upload.

Estimated output: dev about 2.5 GB, sweep about 37 GB, full about 250 GB, all private Hub datasets (check your storage quota). To build only the pilot tier first, add `--fraction 0.15`.

## Running it

```bash
# Stage 1: train 6 arms x 1 seed, then sweep each run under every degradation
./scripts/sweep_hf.sh +experiment=exp1_pilot data.data_uri=<user>/reben-sweep \
    model=opt,sar,early,late,gate_global,gate_spatial seed=0
./scripts/submit_hf.sh --entry sweep_eval --data <user>/reben-sweep run_dir=/ckpt/<run_id> stage=pilot
# (locally: uv run python -m src.sweep_eval run_dir=outputs/<run_id> stage=pilot)

uv run python -m src.preflight <runs dir> --ref opt=<BIFOLD S2> --ref sar=<BIFOLD S1>
uv run python -m src.analysis <runs dir>      # tables, verdicts, figures -> <dir>/analysis

# Stage 2, only after preflight prints READY
./scripts/sweep_hf.sh +experiment=exp1 data.data_uri=<user>/reben-full \
    model=opt,sar,early,late,gate_global,gate_spatial seed=0,1,2
./scripts/submit_hf.sh --entry sweep_eval --data <user>/reben-full run_dir=/ckpt/<run_id> stage=full
```

`src.analysis` writes:

| File | Plan item |
|---|---|
| `table1.csv` | Table 1: clean, AUDC, AUDC_rel (seed mean/min/max), params, FLOPs/sample, latency, ranks |
| `table2_hypotheses.csv` | Table 2: H1–H6 verdict, the deciding numbers, the §5 action |
| `orderings.csv` | absolute vs relative ordering per degradation, flagged where they disagree (§3) |
| `fig1_absolute.png`, `fig2_retention.png`, `fig3_h4_interaction.png` | Figures 1–3 |
| `summary.txt` | verdicts plus the §5 hard-stop check (no pair of configs separable anywhere) |
| `per_run.csv`, `curves.csv` | everything above is computed from these |
