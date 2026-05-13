# Multimodal Survival Public

Public code for multimodal survival prediction using precomputed 2D MII image embeddings and EHR feature vectors.

Supported tasks:

- `ascvd`: ASCVD/MACE-or-death prediction.
- `pe`: pulmonary embolism outcome prediction.

Private patient data, metadata files, embeddings, checkpoints, notebooks, and generated outputs are not included.

## Patient-Level Data Split

| Task | Internal Train | Internal Validation | Internal Test | External Test |
|---|---:|---:|---:|---:|
| PE Mortality Prediction | 2,439 (64.78%) | 268 (7.12%) | 1,058 (28.10%) | 396 |
| MACE Prediction | 2,835 (71.3%) | 330 (8.3%) | 809 (20.4%) | 665 |

## Repository Layout

```text
.
├── config.py
├── train.py
├── evaluate.py
├── datasets/
│   ├── ascvd.py
│   ├── pe.py
│   ├── common.py
│   └── build_dataset.py
├── network/
├── util/
└── requirements.txt
```

## Installation

```bash
conda create -n multimodal-survival-public python=3.11
conda activate multimodal-survival-public
pip install -r requirements.txt
```

## Data Requirements

The data are not distributed with this repository. To run the code, prepare local metadata files and 2D embedding files that match the schemas below.

EHR feature dimensions are inferred from the metadata at runtime. This allows ASCVD and PE to use different `ONEHOT` vector lengths.

### ASCVD Metadata

ASCVD metadata should be a CSV file with these columns:

- `split_name`: one of `train`, `val`, or `test`.
- `PatientID`
- `AccessionNumber`
- `heart_largest_cross_section_slice_index`
- `heart_highest_slice_index`
- `heart_lowest_slice_index`
- `MACE_or_DEATH_month`
- `MACE_or_DEATH_event`
- `features` for `--ehr_type CLMBR`, or `1hot_features` for `--ehr_type ONEHOT`.

ASCVD embeddings are expected under:

```text
<embedding_root>/<slice_folder>/<split>/<PatientID>_<AccessionNumber>_embeddings.npy
```

For `--slice_type LCS_slice`, files are expected as:

```text
<embedding_root>/<slice_folder>/<split>/<PatientID>_<AccessionNumber>_<slice_index>_embedding.npy
```

The ASCVD `slice_folder` is determined from `--slice_type` and `--window`:

```text
--window soft --slice_type heart     -> heart_soft
--window soft --slice_type LCS_slice -> LCS_slice_soft
--window full --slice_type heart     -> heart
--window full --slice_type LCS_slice -> LCS_slice
```

### PE Metadata

PE metadata should be a JSONL file with these columns:

- `split_name`: one of `train`, `val`, or `test`.
- `patient_id`
- `AccessionNumber`
- `heart_low_index`
- `heart_high_index`
- `time`
- `event`
- `image`
- `features` for `--ehr_type CLMBR`, or `1hot_feature` for `--ehr_type ONEHOT`.

PE embeddings are expected under:

```text
<embedding_root>/<window>/<patient_id>_<AccessionNumber>_emb.npy
```

If split subfolders exist, the loader will use:

```text
<embedding_root>/<window>/<split>/<patient_id>_<AccessionNumber>_emb.npy
```

Use `--external_embedding_root` when the external cohort embeddings live under a different root.

## Fusion Strategies

Supported `--fusion_type` values:

```text
ImageOnly
EHROnly
Concat
ConcatCLIP
CrossAttn
CoAttn
```

For `CoAttn`, specify the guide modality:

```bash
--co_attn_guide image
```

or:

```bash
--co_attn_guide ehr
```

## Training

### ASCVD Survival Prediction

```bash
python train.py \
  --task ascvd \
  --metadata_file /path/to/ascvd_metadata.csv \
  --embedding_root /path/to/ascvd_embeddings \
  --fusion_type Concat \
  --ehr_type CLMBR \
  --slice_type heart \
  --window soft
```

### PE Survival Prediction

```bash
python train.py \
  --task pe \
  --metadata_file /path/to/pe_metadata.jsonl \
  --embedding_root /path/to/pe_embeddings \
  --fusion_type Concat \
  --ehr_type CLMBR \
  --window Lung
```

For `CoAttn`, add the same guide modality during training and evaluation:

```bash
--co_attn_guide image
```

or:

```bash
--co_attn_guide ehr
```

## Contrastive Pretraining

The framework supports CLIP-style contrastive pretraining for aligning image and EHR embeddings before downstream survival prediction.

### Step 1: Train the CLIP Alignment Model

```bash
python train.py \
  --task ascvd \
  --metadata_file /path/to/metadata.csv \
  --embedding_root /path/to/embeddings \
  --fusion_type CLIP \
  --ehr_type CLMBR
```

### Step 2: Train a Survival Model Using the Aligned Embedding Space

```bash
python train.py \
  --task ascvd \
  --metadata_file /path/to/metadata.csv \
  --embedding_root /path/to/embeddings \
  --fusion_type ConcatCLIP \
  --ehr_type CLMBR \
  --clip_ckpt_path outputs/checkpoints/ascvd/ckpt_MII_CLMBR/CLIP.pth
```

## Evaluation

Use `--ckpt_path` if the checkpoint is not in the default location:

```text
outputs/checkpoints/<task>/ckpt_MII_<ehr_type>/<fusion_type>.pth
```

### ASCVD Internal Evaluation

```bash
python evaluate.py \
  --task ascvd \
  --metadata_file /path/to/ascvd_metadata.csv \
  --embedding_root /path/to/ascvd_embeddings \
  --fusion_type Concat \
  --ehr_type CLMBR \
  --slice_type heart \
  --window soft \
  --evaluation_set test
```

### ASCVD External Evaluation

```bash
python evaluate.py \
  --task ascvd \
  --metadata_file /path/to/ascvd_metadata.csv \
  --embedding_root /path/to/ascvd_embeddings \
  --external_metadata_file /path/to/external_ascvd_metadata.csv \
  --external_embedding_root /path/to/external_ascvd_embeddings \
  --fusion_type Concat \
  --ehr_type CLMBR \
  --slice_type heart \
  --window soft \
  --evaluation_set external
```

### PE Internal Evaluation

```bash
python evaluate.py \
  --task pe \
  --metadata_file /path/to/pe_metadata.jsonl \
  --embedding_root /path/to/pe_embeddings \
  --fusion_type Concat \
  --ehr_type CLMBR \
  --window Lung \
  --evaluation_set test
```

### PE External Evaluation

```bash
python evaluate.py \
  --task pe \
  --metadata_file /path/to/pe_metadata.jsonl \
  --embedding_root /path/to/pe_embeddings \
  --external_metadata_file /path/to/external_pe_metadata.jsonl \
  --external_embedding_root /path/to/external_pe_embeddings \
  --fusion_type Concat \
  --ehr_type CLMBR \
  --window Lung \
  --evaluation_set external
```

For `ConcatCLIP`, include the CLIP checkpoint during evaluation:

```bash
--clip_ckpt_path outputs/checkpoints/ascvd/ckpt_MII_CLMBR/CLIP.pth
```

## Outputs

Checkpoints are saved to:

```text
outputs/checkpoints/<task>/ckpt_MII_<ehr_type>/
```

Evaluation CSVs are saved to:

```text
outputs/results/
```

## Public Release Notes

- This repository intentionally excludes all private data and model weights.
- Only the 2D MII embedding workflow is included.
- Raw CT processing, private cohort construction scripts, and exploratory notebooks are excluded from the public release.
