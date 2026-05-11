import argparse
from pathlib import Path

import pandas as pd
import torch
from torchsurv.loss import cox
from torchsurv.metrics.cindex import ConcordanceIndex

from config import Config
from datasets.build_dataset import build_dataset
from train import SURVIVAL_FUSION_CHOICES, build_optional_clip_model
from util.helpers import bootstrap_ci, load_model_with_flex, prepare_batch_inputs
from util.uniform_models import ModelFactory


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate multimodal survival models on 2D MII embeddings.")
    parser.add_argument("--task", required=True, choices=["ascvd", "pe"])
    parser.add_argument("--metadata_file", required=True, help="Task metadata file with train/val/test rows.")
    parser.add_argument("--embedding_root", required=True, help="Root directory containing 2D MII .npy embeddings.")
    parser.add_argument("--external_metadata_file", default="", help="Optional external evaluation metadata file.")
    parser.add_argument(
        "--external_embedding_root",
        default="",
        help="Optional root for external 2D MII embeddings. Defaults to --embedding_root.",
    )
    parser.add_argument(
        "--evaluation_set",
        default="test",
        choices=["test", "test_noise", "external", "external_noise"],
    )
    parser.add_argument("--fusion_type", default="ImageOnly", choices=SURVIVAL_FUSION_CHOICES)
    parser.add_argument("--ehr_type", default="CLMBR", choices=["CLMBR", "ONEHOT"])
    parser.add_argument("--slice_type", default="heart", choices=["heart", "LCS_slice"])
    parser.add_argument("--window", default="", help="ASCVD: soft/full. PE: Lung/PE/Soft tissue.")
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ckpt_path", default="", help="Model checkpoint path.")
    parser.add_argument("--clip_ckpt_path", default="", help="Required for ConcatCLIP or CrossAttnCLIP.")
    parser.add_argument("--results_dir", default="outputs/results")
    parser.add_argument("--n_boot", type=int, default=100)
    parser.add_argument("--co_attn_guide", default="image", choices=["image", "ehr"])
    parser.add_argument("--co_attn_branch", default="fuse", choices=["fuse", "single"])
    return parser.parse_args()


def find_checkpoint(args):
    if args.ckpt_path:
        path = Path(args.ckpt_path)
    else:
        ckpt_root = Config.get_checkpoint_dir(args.task, "MII", args.ehr_type)
        if args.fusion_type == "CoAttn":
            path = ckpt_root / f"{args.fusion_type}_G{args.co_attn_guide}_B{args.co_attn_branch}.pth"
        else:
            path = ckpt_root / f"{args.fusion_type}.pth"

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    data_loaders, data_sizes = build_dataset(
        task=args.task,
        metadata_file=args.metadata_file,
        external_metadata_file=args.external_metadata_file or None,
        embedding_root=args.embedding_root,
        external_embedding_root=args.external_embedding_root or None,
        ehr_feature_name=args.ehr_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        slice_type=args.slice_type,
        window=args.window or None,
    )
    if args.evaluation_set not in data_loaders:
        raise ValueError(f"{args.evaluation_set} was not built. Provide --external_metadata_file if needed.")

    test_loader = data_loaders[args.evaluation_set]
    sample_batch = next(iter(test_loader))
    ehr_dim = sample_batch["demo_features"].shape[1]
    print(f"Evaluating {args.task}_{args.fusion_type}_MII_{args.ehr_type} on {args.evaluation_set}")
    print(f"Samples: {data_sizes[args.evaluation_set]} | Device: {device}")

    clip_model = build_optional_clip_model(args, ehr_dim, device)
    model = ModelFactory.build(
        strategy=args.fusion_type,
        img_feature_type="MII",
        ehr_dim=ehr_dim,
        co_attn_guide=args.co_attn_guide,
        co_attn_branch=args.co_attn_branch,
        device=device,
        clip_model=clip_model,
        dropout=args.dropout,
    )
    model = load_model_with_flex(
        model=model,
        checkpoint_path=str(find_checkpoint(args)),
        checkpoint_key="state_dict",
        map_location=str(device),
    )
    model.eval()

    cindex_metric = ConcordanceIndex()
    estimates, times_all, events_all = [], [], []
    accessions, patient_ids, image_paths = [], [], []
    test_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
            img_feats, ehr_feats = prepare_batch_inputs(batch, "MII", None, device)
            times = batch["time"].to(device).float()
            events = batch["event"].to(device).to(torch.int)
            preds = model(img_feats, ehr_feats).view(-1)

            estimates.extend(preds.cpu().tolist())
            times_all.extend(times.cpu().tolist())
            events_all.extend(events.cpu().tolist())
            accessions.extend([str(x) for x in batch.get("AccessionNumber", [])])
            patient_ids.extend([str(x) for x in batch.get("patient_id", [])])
            image_paths.extend([str(x) for x in batch.get("path", [])])

            if events.sum() > 0:
                test_loss += cox.neg_partial_log_likelihood(preds, events, times).item()

    estimates_tensor = torch.tensor(estimates, dtype=torch.float32)
    events_tensor = torch.tensor(events_all, dtype=torch.bool)
    times_tensor = torch.tensor(times_all, dtype=torch.float32)

    if events_tensor.sum() > 0:
        c_index = cindex_metric(estimates_tensor, events_tensor, times_tensor)
        print(f"C-index: {float(c_index):.4f}")
        try:
            c_mean, ci = bootstrap_ci(
                cindex_metric,
                estimates_tensor,
                events_tensor,
                times_tensor,
                num_bootstrap=args.n_boot,
            )
            print(f"C-index bootstrap: {c_mean:.4f} (95% CI: {ci[0]:.4f}, {ci[1]:.4f})")
        except Exception as exc:
            print(f"Bootstrap CI failed: {exc}")
    else:
        print("C-index: NaN (no observed events)")

    print(f"Average Cox loss: {test_loss / max(len(test_loader), 1):.4f}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / f"{args.task}_{args.evaluation_set}_{args.ehr_type}_{args.fusion_type}.csv"
    pd.DataFrame(
        {
            "estimate_risk": estimates,
            "event": events_all,
            "time": times_all,
            "accession_num": accessions,
            "patient_id": patient_ids,
            "image_path": image_paths,
        }
    ).to_csv(out_csv, index=False)
    print(f"Saved results to: {out_csv}")


if __name__ == "__main__":
    main()
