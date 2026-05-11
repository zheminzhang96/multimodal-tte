import torch
from torch import nn
#from network.Image_features import *
from network.projection_heads import *

############################################################
# cox_surv 
############################################################



class SurvivalNet512(nn.Module):
    def __init__(self, input_size=256+256, hidden_size=256, output_size=1, dropout_prob=0.3):
        super(SurvivalNet512, self).__init__()
        self.in_linear = input_size
        self.fc = nn.Sequential(
            nn.Linear(self.in_linear, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_prob), 
            nn.Linear(hidden_size, hidden_size//2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_prob), 
            nn.Linear(hidden_size//2, hidden_size//8),
            nn.LeakyReLU(),
            nn.Dropout(dropout_prob),  
            nn.Linear(hidden_size//8, output_size),
        )
    def forward(self, x):
        x = self.fc(x)
        return x
    
class SurvivalNet256(nn.Module):
    def __init__(self, input_size=256, hidden_size=128, output_size=1, dropout_prob=0.3):
        super(SurvivalNet256, self).__init__()
        self.in_linear = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.in_linear, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_prob),  
            nn.Linear(hidden_size, hidden_size//2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_prob), 
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.LeakyReLU(),
            nn.Dropout(dropout_prob),  
            nn.Linear(hidden_size//4, output_size),
        )
        
    def forward(self, x):
        x = self.fc(x)
        return x

# class SurvivalNetConcat(nn.Module):
#     def __init__(self, image_input_size=1024, ehr_input_size=768, project_size=256, output_size=1, dropout_prob=0.3):
#         super(SurvivalNetConcat, self).__init__()
#         self.concat_size = project_size*2
#         #self.image_encoder = Image_Features(emb_size=image_input_size, dropout_prob=dropout_prob)
#         self.image_projection = ProjectionHead(embedding_dim=image_input_size, projection_dim=project_size, dropout_rate=dropout_prob)
#         self.ehr_projection = ProjectionHead(embedding_dim=ehr_input_size, projection_dim=project_size, dropout_rate=dropout_prob)

#         self.fc1 = nn.Sequential(
#             nn.Linear(self.concat_size, project_size),
#             nn.LeakyReLU(),
#             nn.Dropout(dropout_prob),  
#             nn.Linear(project_size, project_size//2),
#             nn.LeakyReLU(),
#             nn.Dropout(dropout_prob), 
#             nn.Linear(project_size//2, project_size//4),
#             nn.LeakyReLU(),
#             nn.Dropout(dropout_prob),  
#             nn.Linear(project_size//4, output_size),
#         )
       
#     def forward(self, img, ehr):
#         #img = self.image_encoder(img)
#         img = self.image_projection(img)
#         ehr = self.ehr_projection(ehr)
#         x = torch.concat([img, ehr], dim=1)
#         x = self.fc1(x)
#         return x