from torch.utils.data import DataLoader

from datasets import ascvd, pe


TASK_BUILDERS = {
    "ascvd": ascvd.build_datasets,
    "pe": pe.build_datasets,
}


def build_dataset(
    task,
    metadata_file,
    embedding_root,
    ehr_feature_name,
    batch_size,
    num_workers,
    external_metadata_file=None,
    external_embedding_root=None,
    slice_type="heart",
    window=None,
):
    if task not in TASK_BUILDERS:
        raise ValueError(f"task must be one of {sorted(TASK_BUILDERS)}")

    if window is None:
        window = "soft" if task == "ascvd" else "PE"

    datasets = TASK_BUILDERS[task](
        metadata_file=metadata_file,
        external_metadata_file=external_metadata_file,
        embedding_root=embedding_root,
        external_embedding_root=external_embedding_root,
        ehr_feature_name=ehr_feature_name,
        slice_type=slice_type,
        window=window,
    )

    dataloaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )
        for split, dataset in datasets.items()
    }
    sizes = {split: len(dataset) for split, dataset in datasets.items()}
    return dataloaders, sizes
