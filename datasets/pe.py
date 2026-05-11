from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from datasets.common import (
    parse_features,
    replace_frac_with_uniform_noise,
    replace_frac_with_uniform_noise_torch,
)


FEATURE_COLUMNS = {
    "CLMBR": "features",
    "ONEHOT": "1hot_feature",
}


def build_embedding_path(embedding_root, window, split, rec):
    uid = f"{rec['patient_id']}_{rec['accession_number']}"
    root = Path(embedding_root) / window
    if split:
        split_root = root / split
        if split_root.exists():
            root = split_root
    return root / f"{uid}_emb.npy"


def iter_embeddings(
    df,
    embedding_root,
    window,
    split,
    tabular_col,
    add_img_noise=False,
    img_noise_frac=0.25,
    seed=None,
):
    rng = np.random.default_rng(seed)
    for _, row in df.iterrows():
        features = parse_features(row[tabular_col])
        rec = {
            "accession_number": str(row["AccessionNumber"]),
            "patient_id": str(row["patient_id"]),
            "time": float(row["time"]),
            "event": int(float(row["event"])),
            "path": str(row.get("image", "")),
            "features": features,
        }

        emb_path = build_embedding_path(embedding_root, window, split, rec)
        arr = np.squeeze(np.load(emb_path)).mean(axis=0)
        if add_img_noise:
            arr = replace_frac_with_uniform_noise(arr, frac=img_noise_frac, rng=rng)
        yield rec, arr


class PEMIIDataset(Dataset):
    def __init__(self, df, embedding_root, split, tabular_col, window="PE"):
        self.samples = []
        for rec, arr_mean in iter_embeddings(df, embedding_root, window, split, tabular_col):
            self.samples.append(make_sample(rec, arr_mean))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class PEMIIDatasetNoise(PEMIIDataset):
    def __init__(
        self,
        df,
        embedding_root,
        split,
        tabular_col,
        window="PE",
        demo_noise_frac=0.25,
        add_demo_noise=False,
        img_noise_frac=0.25,
        add_img_noise=False,
        seed=0,
    ):
        self.add_demo_noise = add_demo_noise
        self.demo_noise_frac = demo_noise_frac
        self.seed = seed
        self.samples = []

        for rec, arr_mean in iter_embeddings(
            df,
            embedding_root,
            window,
            split,
            tabular_col,
            add_img_noise=add_img_noise,
            img_noise_frac=img_noise_frac,
            seed=seed,
        ):
            self.samples.append(make_sample(rec, arr_mean))

    def __getitem__(self, idx):
        sample = dict(self.samples[idx])
        if self.add_demo_noise:
            generator = torch.Generator()
            generator.manual_seed(self.seed + idx)
            sample["demo_features"] = replace_frac_with_uniform_noise_torch(
                sample["demo_features"], frac=self.demo_noise_frac, generator=generator
            )
        return sample


def make_sample(rec, arr_mean):
    return {
        "image": torch.tensor(arr_mean, dtype=torch.float32),
        "demo_features": torch.tensor(rec["features"], dtype=torch.float32),
        "time": torch.tensor(rec["time"], dtype=torch.float32),
        "event": torch.tensor(rec["event"], dtype=torch.int),
        "AccessionNumber": rec["accession_number"],
        "patient_id": rec["patient_id"],
        "path": rec["path"],
    }


def read_metadata(path):
    return pd.read_json(Path(path), orient="records", lines=True)


def _clean_pe_df(df):
    df = df.dropna(subset=["event", "heart_low_index"])
    df = df.copy()
    df["event"] = df["event"].astype(float).astype(int)
    return df


def build_datasets(
    metadata_file,
    embedding_root,
    ehr_feature_name,
    external_metadata_file=None,
    external_embedding_root=None,
    window="PE",
    **_,
):
    if ehr_feature_name not in FEATURE_COLUMNS:
        raise ValueError(f"ehr_feature_name must be one of {sorted(FEATURE_COLUMNS)}")

    tabular_col = FEATURE_COLUMNS[ehr_feature_name]
    df = _clean_pe_df(read_metadata(metadata_file))
    datasets = {}

    for split in ("train", "val", "test"):
        split_df = df[df["split_name"] == split] if "split_name" in df.columns else df
        datasets[split] = PEMIIDataset(split_df, embedding_root, split, tabular_col, window=window)

    test_df = df[df["split_name"] == "test"] if "split_name" in df.columns else df
    datasets["test_noise"] = PEMIIDatasetNoise(
        test_df,
        embedding_root,
        "test",
        tabular_col,
        window=window,
        add_demo_noise=True,
        demo_noise_frac=0.5,
        seed=123,
    )

    if external_metadata_file:
        df_external = _clean_pe_df(read_metadata(external_metadata_file))
        external_root = external_embedding_root or embedding_root
        datasets["external"] = PEMIIDataset(
            df_external, external_root, "external", tabular_col, window=window
        )
        datasets["external_noise"] = PEMIIDatasetNoise(
            df_external,
            external_root,
            "external",
            tabular_col,
            window=window,
            add_demo_noise=True,
            demo_noise_frac=0.5,
            seed=123,
        )

    return datasets
