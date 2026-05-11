import argparse
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from torchsurv.loss import cox
from torchsurv.metrics.cindex import ConcordanceIndex

from config import Config
from datasets.build_dataset import build_dataset
from util.helpers import load_model_with_flex, prepare_batch_inputs, save_checkpoint
from util.uniform_models import ModelFactory


SURVIVAL_FUSION_CHOICES = [
    "ImageOnly",
    "EHROnly",
    "Concat",
    "ConcatCLIP",
    "CrossAttn",
    #"CrossAttnCLIP",
    "CoAttn",
]
FUSION_CHOICES = ["CLIP", *SURVIVAL_FUSION_CHOICES]


def infer_ehr_dim(sample_batch):
    """Infer EHR feature dimension from the dataloader output."""
    if "demo_features" not in sample_batch:
        raise KeyError("Expected dataloader batch to include 'demo_features'.")

    features = sample_batch["demo_features"]
    if features.ndim != 2:
        raise ValueError(
            "'demo_features' must be a 2D tensor shaped [batch_size, ehr_dim]. "
            f"Got shape: {tuple(features.shape)}"
        )
    return int(features.shape[1])


def parse_args():
    parser = argparse.ArgumentParser(description="Train multimodal survival models on 2D MII embeddings.")
    parser.add_argument("--task", required=True, choices=["ascvd", "pe"])
    parser.add_argument("--metadata_file", required=True, help="Task metadata file with train/val/test rows.")
    parser.add_argument("--embedding_root", required=True, help="Root directory containing 2D MII .npy embeddings.")
    parser.add_argument("--external_metadata_file", default="", help="Optional external evaluation metadata file.")
    parser.add_argument(
        "--external_embedding_root",
        default="",
        help="Optional root for external 2D MII embeddings. Defaults to --embedding_root.",
    )
    parser.add_argument("--fusion_type", default="ImageOnly", choices=FUSION_CHOICES)
    parser.add_argument("--ehr_type", default="CLMBR", choices=["CLMBR", "ONEHOT"])
    parser.add_argument("--slice_type", default="heart", choices=["heart", "LCS_slice"])
    parser.add_argument("--window", default="", help="ASCVD: soft/full. PE: Lung/PE/Soft tissue.")
    parser.add_argument("--clip_ckpt_path", default="", help="Required for ConcatCLIP or CrossAttnCLIP.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--co_attn_guide", default="image", choices=["image", "ehr"])
    parser.add_argument("--co_attn_branch", default="fuse", choices=["fuse", "single"])
    return parser.parse_args()


def build_optional_clip_model(args, ehr_dim, device):
    if args.fusion_type not in {"ConcatCLIP", "CrossAttnCLIP"}:
        return None
    if not args.clip_ckpt_path:
        raise ValueError("--clip_ckpt_path is required for CLIP-initialized fusion models.")

    clip_model = ModelFactory.build(
        strategy="CLIP",
        img_feature_type="MII",
        ehr_dim=ehr_dim,
        co_attn_guide=args.co_attn_guide,
        co_attn_branch=args.co_attn_branch,
        device=device,
        dropout=args.dropout,
    )
    return load_model_with_flex(
        model=clip_model,
        checkpoint_path=args.clip_ckpt_path,
        checkpoint_key="state_dict",
        map_location=str(device),
    )


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

    train_loader = data_loaders["train"]
    val_loader = data_loaders["val"]
    sample_batch = next(iter(train_loader))
    ehr_dim = infer_ehr_dim(sample_batch)

    run_name = f"{args.task}_{args.fusion_type}_MII_{args.ehr_type}"
    writer = SummaryWriter(log_dir=Config.LOG_DIR / run_name)
    ckpt_dir = Config.get_checkpoint_dir(args.task, "MII", args.ehr_type)

    print(f"Training {run_name} on {device}")
    print(f"Detected EHR dimension: {ehr_dim}")
    print(f"Train samples: {data_sizes['train']} | Val samples: {data_sizes['val']}")
    print(f"Saving checkpoints to: {ckpt_dir}")

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    cindex = ConcordanceIndex()
    best_val_loss = float("inf")
    train_clip = args.fusion_type == "CLIP"

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            img_feats, ehr_feats = prepare_batch_inputs(batch, "MII", None, device)
            if train_clip:
                loss = model(img_feats, ehr_feats)
            else:
                times = batch["time"].to(device).float()
                events = batch["event"].to(device).to(torch.int)
                preds = model(img_feats, ehr_feats).squeeze()
                loss = cox.neg_partial_log_likelihood(preds, events, times)
            (loss / args.grad_accum_steps).backward()

            if (batch_idx + 1) % args.grad_accum_steps == 0 or batch_idx + 1 == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item()

        avg_train_loss = train_loss / max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        all_preds, all_times, all_events = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                img_feats, ehr_feats = prepare_batch_inputs(batch, "MII", None, device)
                if train_clip:
                    val_loss += model(img_feats, ehr_feats).item()
                else:
                    times = batch["time"].to(device).float()
                    events = batch["event"].to(device).to(torch.int)
                    preds = model(img_feats, ehr_feats).squeeze()

                    if events.sum() > 0:
                        val_loss += cox.neg_partial_log_likelihood(preds, events, times).item()
                    all_preds.extend(preds.cpu().tolist())
                    all_times.extend(times.cpu().tolist())
                    all_events.extend(events.cpu().tolist())

        avg_val_loss = val_loss / max(len(val_loader), 1)
        if train_clip:
            val_cindex = None
        elif sum(all_events) > 0:
            val_cindex = cindex(
                torch.tensor(all_preds),
                torch.tensor(all_events, dtype=torch.bool),
                torch.tensor(all_times),
            ).item()
        else:
            val_cindex = 0.0

        if train_clip:
            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Train CLIP Loss: {avg_train_loss:.4f} | "
                f"Val CLIP Loss: {avg_val_loss:.4f}"
            )
        else:
            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"C-index: {val_cindex:.4f}"
            )
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        if val_cindex is not None:
            writer.add_scalar("Metric/CIndex", val_cindex, epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if args.fusion_type == "CoAttn":
                filename = f"{args.fusion_type}_G{args.co_attn_guide}_B{args.co_attn_branch}.pth"
            else:
                filename = f"{args.fusion_type}.pth"
            save_checkpoint(model, optimizer, epoch, avg_val_loss, ckpt_dir, filename)
            print(f"Saved checkpoint: {Path(ckpt_dir) / filename}")

    writer.close()


if __name__ == "__main__":
    main()
