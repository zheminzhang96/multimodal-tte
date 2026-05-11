import torch.nn as nn

from network.CLIP import CLIPModel
from network.CoAttention import CoAttention
from network.CrossAttention import SurvivalNetCrossAttention
from network.FusionModel import (
    SurvivalNetConcat,
    SurvivalNetEHROnly,
    SurvivalNetImageOnly,
)


class SurvivalModelWrapper(nn.Module):
    """Optional backbone wrapper kept for compatibility with saved checkpoints."""

    def __init__(self, backbone, fusion_head):
        super().__init__()
        self.backbone = backbone
        self.fusion_head = fusion_head

    def forward(self, img, ehr):
        if self.backbone is not None:
            img = self.backbone(img)
        return self.fusion_head(img, ehr)


class ModelFactory:
    @staticmethod
    def build(
        strategy,
        img_feature_type,
        ehr_dim,
        co_attn_guide,
        co_attn_branch,
        device,
        clip_model=None,
        dropout=0.3,
    ):
        if img_feature_type != "MII":
            raise ValueError("The public release supports only precomputed 2D MII embeddings.")

        img_dim = 1024
        if clip_model is not None and isinstance(clip_model, SurvivalModelWrapper):
            clip_model = clip_model.fusion_head

        if strategy == "ImageOnly":
            fusion_head = SurvivalNetImageOnly(img_dim, 1, dropout)
        elif strategy == "EHROnly":
            fusion_head = SurvivalNetEHROnly(ehr_dim, 1, dropout)
        elif strategy in {"Concat", "ConcatCLIP"}:
            fusion_head = SurvivalNetConcat(img_dim, ehr_dim, 256, 1, clip_model, dropout)
        elif strategy in {"CrossAttn", "CrossAttnCLIP"}:
            fusion_head = SurvivalNetCrossAttention(
                image_embedding=img_dim,
                tabular_embedding=ehr_dim,
                project_embedding=256,
                sequenth_dim=64,
                num_heads=4,
                dropout_rate=dropout,
                clip_model=clip_model,
                device=device,
            )
        elif strategy == "CoAttn":
            fusion_head = CoAttention(
                fusion="concat",
                guide=co_attn_guide,
                out_branch=co_attn_branch,
                image_embed_size=img_dim,
                ehr_embed_size=ehr_dim,
                project_embed_size=256,
                num_heads=2,
                dropout=dropout,
                device=device,
            )
        elif strategy == "CLIP":
            fusion_head = CLIPModel(
                temperature=0.07,
                image_embedding=img_dim,
                tabular_embedding=ehr_dim,
                project_embedding=256,
                dropout_rate=dropout,
                device=device,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return SurvivalModelWrapper(None, fusion_head).to(device)
