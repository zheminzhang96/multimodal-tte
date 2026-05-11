import torch
import torch.nn as nn
from network.projection_heads import *

# ==========================================
# Single Modality Models
# ==========================================

class SurvivalNetImageOnly(nn.Module):
    def __init__(self, img_dim, output_dim=1, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(img_dim, 256),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Dropout(dropout),  
            nn.Linear(64, output_dim),
        )

    def forward(self, img, ehr=None):
        return self.head(img)

class SurvivalNetEHROnly(nn.Module):
    def __init__(self, ehr_dim, output_dim=1, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(ehr_dim, 256),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Dropout(dropout),  
            nn.Linear(64, output_dim),
        )

    def forward(self, img=None, ehr=None):
        return self.head(ehr)

# ==========================================
# Fusion Models
# ==========================================

class SurvivalNetConcat(nn.Module):
    def __init__(self, img_dim, ehr_dim, project_size=256, output_size=1, clip_model=None, dropout=0.3):
        super().__init__()
        if clip_model == None:
            self.img_proj = ProjectionHead(img_dim, project_size, dropout)
            self.ehr_proj = ProjectionHead(ehr_dim, project_size, dropout)
        elif clip_model != None:
            self.img_proj = clip_model.image_projection
            self.ehr_proj = clip_model.tab_projection
        self.fc = nn.Sequential(
            nn.Linear(project_size * 2, project_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(project_size, project_size // 2),
            nn.LeakyReLU(),
            nn.Linear(project_size // 2, project_size // 8),
            nn.LeakyReLU(),
            nn.Dropout(dropout),  
            nn.Linear(project_size // 8, output_size)
        )
    def forward(self, img, ehr):
        i = self.img_proj(img)
        e = self.ehr_proj(ehr)
        combined = torch.cat((i, e), dim=1)
        return self.fc(combined)
    
