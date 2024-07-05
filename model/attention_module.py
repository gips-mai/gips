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
        norm_type: str = "batch_norm",
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

        self.norm = (
            torch.nn.BatchNorm1d(attn_input_img_size)
            if norm_type == "batch_norm"
            else nn.LayerNorm(attn_input_img_size)
            if norm_type == "layer_norm"
            else None
        )

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
        if self.norm is not None:
            x = self.norm(img_embedding)
        attention_scores = self.model(x) + self.beta
        return attention_scores


class AttentionWeightedAggregation(nn.Module):

    def __init__(self, temperature):

        self.temperature = temperature
        self.weighting_f = F.sigmoid

    
    def forward(self, clue_embeddings: torch.Tensor, img_embedding:torch.Tensor, attention: torch.Tensor):
        x = clue_embeddings
        if self.norm is not None:
            x = self.norm(img_embedding) #TODO: self.norm
        
        aggregated_embedding = torch.sum(self.weighting_f(self.temperature * attention) * x) / x.size
        return aggregated_embedding


def get_pseudo_label_loss(clue_countries, hot_enc_size=221):

    l2_loss = torch.nn.MSEloss()
    mat = torch.zeros((len(clue_countries), hot_enc_size))
    for i, enc in enumerate(clue_countries):
        mat[i] = enc

    def pseudo_label_loss(attention_prediction, gt_country_encoding):

        return l2_loss(mat @ gt_country_encoding.view(-1, 1), attention_prediction)

    return pseudo_label_loss()