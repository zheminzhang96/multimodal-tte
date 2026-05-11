import torch
from torch import nn

class ImageFeaturesPool(nn.Module):
    def __init__(self, emb_size=1024, dropout_prob=0.3):
        super(ImageFeaturesPool, self).__init__()
        
        self.in_linear = 512*2*2*2
        self.conv3d = nn.Sequential(
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),               # [B, 512, 2, 2, 2]
        )
        self.fc = nn.Sequential(
            nn.Linear(self.in_linear, emb_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_prob),  # Dropout for regularization
        )
    
    def forward(self, x):
        x = self.conv3d(x)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.fc(x)  # [b, 1024]   
        return x
    
class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout_rate):
        super(ProjectionHead, self).__init__()

        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x