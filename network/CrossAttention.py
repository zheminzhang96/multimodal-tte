import torch
import torch.nn as nn
import torch.nn.functional as F

from network.Cox_MLP import SurvivalNet512
from network.projection_heads import ProjectionHead


class SurvivalNetCrossAttention(nn.Module):
    def __init__(
        self,
        image_embedding,
        tabular_embedding,
        project_embedding,
        sequenth_dim,
        num_heads,
        dropout_rate,
        clip_model,
        device,
    ):
        super().__init__()
        self.embed_dim = project_embedding
        self.num_heads = num_heads
        self.sequence_dim = sequenth_dim
        self.sequence_n = project_embedding // sequenth_dim
        self.head_dim = sequenth_dim // num_heads
        self.clip_model = clip_model
        assert project_embedding % num_heads == 0, "embed_dim must be divisible by num_heads"

        if self.clip_model is None:
            self.image_projection = ProjectionHead(
                embedding_dim=image_embedding,
                projection_dim=project_embedding,
                dropout_rate=dropout_rate,
            )
            self.tab_projection = ProjectionHead(
                embedding_dim=tabular_embedding,
                projection_dim=project_embedding,
                dropout_rate=dropout_rate,
            )
        else:
            self.image_projection = clip_model.image_projection
            self.tab_projection = clip_model.tab_projection

        self.q_proj = nn.Linear(sequenth_dim, sequenth_dim)
        self.k_proj = nn.Linear(sequenth_dim, sequenth_dim)
        self.v_proj = nn.Linear(sequenth_dim, sequenth_dim)
        self.out_proj = nn.Linear(sequenth_dim, sequenth_dim)
        self.scale = self.head_dim**-0.5
        self.surv_model = SurvivalNet512(
            input_size=512,
            hidden_size=256,
            output_size=1,
            dropout_prob=dropout_rate,
        )
        self.device = device

    def forward(self, image_input, tabular_input):
        image_input = image_input.to(self.device)
        tabular_input = tabular_input.to(self.device)

        image_features = self.image_projection(image_input).unsqueeze(1).to(self.device)
        tabular_features = self.tab_projection(tabular_input).unsqueeze(1).to(self.device)
        image_features = image_features.reshape(-1, self.sequence_n, self.sequence_dim)
        tabular_features = tabular_features.reshape(-1, self.sequence_n, self.sequence_dim)

        image_to_tabular = self._cross_attention(image_features, tabular_features)
        tabular_to_image = self._cross_attention(tabular_features, image_features)
        fused = torch.cat([image_to_tabular, tabular_to_image], dim=-1)
        return self.surv_model(fused)

    def _cross_attention(self, query_features, context_features):
        batch_size, query_len, _ = query_features.shape
        _, context_len, _ = context_features.shape

        query = self.q_proj(query_features)
        key = self.k_proj(context_features)
        value = self.v_proj(context_features)

        query = query.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, context_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, context_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attended = torch.matmul(attn_weights, value)
        attended = attended.transpose(1, 2).reshape(batch_size, query_len, self.sequence_dim)
        attended = self.out_proj(attended)
        return attended.reshape(-1, 1, self.embed_dim)
