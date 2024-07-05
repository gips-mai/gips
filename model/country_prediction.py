import torch
import torch.nn as nn

class CountryClassifier(nn.Module):
    def __init__(self, clue_embedding_size=512, image_embedding_size=512, alpha=0.75, n_countries=219) -> None:
        #TODO: adjust image_embedding size
        super().__init__()

        self.alpha = alpha
        self.n_countries = n_countries
        self.image_embedding_size = image_embedding_size
        self.clue_embedding_size = clue_embedding_size
        self.embedding_size = clue_embedding_size + image_embedding_size

        self.init_layers()

        self.bce_loss =  torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1)) # can be changed
        self.ce_loss = torch.nn.CrossEntropyLoss()


    def init_layers(self):
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_size, 1024), #TODO: over 1400 features in original paper
            nn.Linear(1024, 1024),
            nn.Linear(1024, self.n_countries),
        )
        
    def get_loss(self, output, target):
        ce_loss = self.ce_loss(output, target)
        if self.alpha > 0:
            attn_loss = self.bce_loss(output, target)
            total_loss = ce_loss * (1 - self.alpha) + attn_loss * self.alpha
        else:
            total_loss = ce_loss
        return total_loss

    def forward(self, x):
        return nn.functional.softmax(self.classifier(x), dim=1)
    
    def training_step(self, x, target):
        output = self.forward(x)
        loss = self.get_loss(output, target)
        return loss
