import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LinearAttention(nn.Module):
    def __init__(
        self,
        attn_input_img_size: int,
        text_features_size: int,
        hidden_layer_size_0: int,
        hidden_layer_size_1: int,
        beta=-1.0,
    ):
        """
        A simple linear layer that only takes an image embedding as input.

        Args:
            attn_input_img_size (_type_): input dim
            text_features_size (_type_): embed dim
            normalization (str): how to normalize the attention scores as probabilities, options: softmax or sigmoid
            norm_type (str): normalize inputs. values: batch_norm, layer_norm, or None
            beta: -1 (default) means we learn a beta param, else we use the hardcoded value that is specified.
        """
        super().__init__()
        if beta == -1:
            # Learn beta:
            self.beta = torch.nn.Parameter(torch.rand(1).squeeze() + 0.7)
        else:
            # Hardcode beta:
            self.beta = torch.nn.Parameter(torch.tensor(beta), requires_grad=False)
        logging.info(f"Using attn_beta = {self.beta}, type = SimpleLinearProjectionAttention")


        self.norm =  torch.nn.BatchNorm1d(attn_input_img_size)

        self.layers = nn.Sequential(
            nn.Linear(attn_input_img_size, hidden_layer_size_0),
            nn.ReLU(),
            nn.Linear(hidden_layer_size_0, hidden_layer_size_1),
            nn.ReLU(),
            nn.Linear(hidden_layer_size_1, text_features_size)
        )

        # self.f0 = F.relu(torch.nn.Linear(attn_input_img_size, hidden_layer_size_0))
        # self.f1 = F.relu(torch.nn.Linear(hidden_layer_size_0, hidden_layer_size_1))
        # self.fc = torch.nn.Linear(hidden_layer_size_1, text_features_size)

    def forward(self, img_embedding: torch.Tensor):
        x = img_embedding
        # Only apply batch normalization if we have a batch
        if x.shape[0] > 1:
            x = self.norm(img_embedding)  # Batched vs nonbatched training
        attention_scores = self.layers(x) + self.beta
        return attention_scores


class AttentionWeightedAggregation:

    def __init__(self, clues, temperature=1):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.temperature = temperature
        self.weighting_f = F.sigmoid

        self.clue_embeddings = torch.tensor(clues['encoding']).to(device)
    
    def forward(self, attention: torch.Tensor):

        batch_size = attention.shape[0]

        f1 = self.weighting_f(self.temperature * attention)

        # adjust dimensions to allow for element-wise multiplication in batches
        f1 = f1.unsqueeze(dim=2)
        f2 = self.clue_embeddings.repeat(batch_size, 1, 1)

        # compute element-wise multiplication for each batch
        f2 = f2 * f1


        return torch.sum(f2, dim=1) / self.clue_embeddings.shape[1]
