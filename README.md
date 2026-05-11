# Multimodal Survival Public

This repository contains the public, reproducible code for multimodal survival modeling on two tasks:

- `ascvd`: ASCVD/MACE-or-death prediction.
- `pe`: pulmonary embolism outcome prediction.

Private patient data, metadata files, embeddings, checkpoints, notebooks, and generated outputs are not included. The public workflow uses precomputed 2D MII image embeddings plus EHR feature vectors only.

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

### ASCVD Metadata

ASCVD metadata should be a CSV file with these columns:

- `split_name`: one of `train`, `val`, `test`.
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

- `--window soft`, `--slice_type heart`: `heart_soft`
- `--window soft`, `--slice_type LCS_slice`: `LCS_slice_soft`
- `--window full`, `--slice_type heart`: `heart`
- `--window full`, `--slice_type LCS_slice`: `LCS_slice`

### PE Metadata

PE metadata should be a JSONL file with these columns:

- `split_name`: one of `train`, `val`, `test`.
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

## Training

Train an ASCVD survival model:

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

Train a PE survival model:

```bash
python train.py \
  --task pe \
  --metadata_file /path/to/pe_metadata.jsonl \
  --embedding_root /path/to/pe_embeddings \
  --fusion_type Concat \
  --ehr_type CLMBR \
  --window PE
```

Supported survival fusion strategies are:

```text
ImageOnly, EHROnly, Concat, ConcatCLIP, CrossAttn, CrossAttnCLIP, CoAttn
```

### CLIP Pretraining

Train the CLIP alignment model first:

```bash
python train.py \
  --task ascvd \
  --metadata_file /path/to/metadata.csv \
  --embedding_root /path/to/embeddings \
  --fusion_type CLIP \
  --ehr_type CLMBR
```

Then train a CLIP-initialized survival model:

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

```bash
python evaluate.py \
  --task ascvd \
  --metadata_file /path/to/ascvd_metadata.csv \
  --embedding_root /path/to/ascvd_embeddings \
  --fusion_type Concat \
  --ehr_type CLMBR \
  --evaluation_set test
```

For external evaluation:

```bash
python evaluate.py \
  --task pe \
  --metadata_file /path/to/pe_metadata.jsonl \
  --embedding_root /path/to/pe_embeddings \
  --external_metadata_file /path/to/external_metadata.jsonl \
  --external_embedding_root /path/to/external_embeddings \
  --fusion_type Concat \
  --ehr_type CLMBR \
  --evaluation_set external
```

## Public Release Notes

- This repository intentionally excludes all private data and model weights.
- Only the 2D MII embedding workflow is included.
- Raw CT processing, private cohort construction scripts, and exploratory notebooks are excluded from the public release.
