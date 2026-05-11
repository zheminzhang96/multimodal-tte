import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from network.projection_heads import *

class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature,
        image_embedding,
        tabular_embedding,
        project_embedding,
        dropout_rate,
        device
    ):
        super().__init__()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding, projection_dim=project_embedding, dropout_rate=dropout_rate)
        self.tab_projection = ProjectionHead(embedding_dim=tabular_embedding, projection_dim=project_embedding, dropout_rate=dropout_rate)
        self.temperature = temperature
        self.device = device
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.temperature))

    def forward(self, image_input, tabular_input):
        #print("image input", image_input.shape)
        #print("tabular input", tabular_input.shape)
        image_input = image_input.to(self.device)
        tabular_input = tabular_input.to(self.device)
        #image_features = self.image_encoder(image_input)
        image_features = self.image_projection(image_input)
        #print("image features", image_features.shape)
        tabular_features = self.tab_projection(tabular_input)

        feat_img = image_features / image_features.norm(dim=1, keepdim=True)
        feat_tab = tabular_features / tabular_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * feat_img @ feat_tab.t()
        logits_per_tab = logits_per_image.t()

        n = logits_per_image.size(0)
        labels = torch.arange(n).to(logits_per_image.device)  # Ensure labels are on the same device as logits

        # # Compute cross-entropy loss for image-to-text (axis=0 in the pseudocode)
        loss_i = F.cross_entropy(logits_per_image, labels)

        # # Compute cross-entropy loss for text-to-image (axis=1 in the pseudocode)
        loss_t = F.cross_entropy(logits_per_tab, labels)

        # # Compute the final loss as the average of loss_i and loss_t
        loss = (loss_i + loss_t) / 2
        #print("loss:", loss)

        return loss
