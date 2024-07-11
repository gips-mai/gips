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
    ):
        """
        A simple linear layer that only takes an image embedding as input.

        Args:
            attn_input_img_size (_type_): input dim
            text_features_size (_type_): embed dim
            normalization (str): how to normalize the attention scores as probabilities, options: softmax or sigmoid
            norm_type (str): normalize inputs. values: batch_norm, layer_norm, or None
        """
        super().__init__()

        self.norm = torch.nn.BatchNorm1d(attn_input_img_size)

        self.layers = nn.Sequential(
            nn.Linear(attn_input_img_size, hidden_layer_size_1),
            nn.ReLU(),
            nn.Linear(hidden_layer_size_1, text_features_size),
            nn.ReLU(),
        )

    def forward(self, img_embedding: torch.Tensor):
        x = img_embedding
        # Only apply batch normalization if we have a batch
        if x.shape[0] > 1:
            x = self.norm(img_embedding)
        attention_scores = self.layers(x)
        return attention_scores


class AttentionWeightedAggregation:

    def __init__(self, clues, temperature=1):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.temperature = temperature
        self.weighting_f = F.sigmoid

        self.clue_embeddings = torch.tensor(clues['encoding']).to(device)
    
    def forward(self, attention: torch.Tensor):

        batch_size = attention.shape[0]

        # normalize with sigmoid to [0,1]
        batched_clues_weights = self.weighting_f(self.temperature * attention)

        # adjust dimensions to allow for element-wise multiplication in batches
        # f1: [batch_size, num_clues, 1]
        # f2: [batch_size, num_clues, clue_embedding_size]
        batched_clues_weights = batched_clues_weights.unsqueeze(dim=2)
        unsqueezed_clue_embeddings = self.clue_embeddings.unsqueeze(dim=0)
        batched_clue_embs = unsqueezed_clue_embeddings.expand(batch_size, -1, -1)

        # compute element-wise multiplication for each batch
        weighted_clue_embs = batched_clue_embs * batched_clues_weights

        # Normalize the weighted sum of clue embeddings
        return torch.sum(weighted_clue_embs, dim=1) / self.clue_embeddings.shape[0]
