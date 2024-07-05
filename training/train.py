from pathlib import Path
import sys
sys.path.append(Path.cwd().parent)

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from model.attention_module import AttentionWeightedAggregation, LinearAttention
from model.backbone import LatLongHead, LocationAttention, StreetCLIP, TextEncoder #TODO remove unused imports
from model.country_prediction import CountryClassifier
from model.head.geolocation_head import MLPCentroid, HybridHeadCentroid

from datasets import load_dataset

### HYPER PARAMETERS ###
lr = 0.001
alpha = 0.75
use_tanh=True
scale_tanh=1.2
### HYPER PARAMETERS ###

device = 'cuda'

labels = pd.read_csv('path/to/labels.csv')
country_encoding = pd.read_csv('path/to/country/encoding.csv')

clues = pd.read_pickle("path/to/clues.pkl")
clue_embedding_size:int = 512

encoded_descriptions = pd.read_csv('path/to/encoded/descriptions.csv')
text_embedding_size:int = 512

clip_embedding_size:int = 768
attention_aggregation = AttentionWeightedAggregation() #TODO definde temperature
linear_attention = LinearAttention(attn_input_img_size=clip_embedding_size, text_features_size=text_embedding_size, hidden_layer_size_0=1024, hidden_layer_size_1=1024) #TODO hidden layer size
country_classifier = CountryClassifier(clue_embedding_size=clue_embedding_size, image_embedding_size=clip_embedding_size, alpha=alpha)

previous_stage_output = text_embedding_size+clip_embedding_size+clue_embedding_size
geohead = MLPCentroid(initial_dim=previous_stage_output, hidden_dim=[previous_stage_output, 1024, 512])
hybrid_head_centroid = HybridHeadCentroid(final_dim=11398, path='../data/quadtree/quadtree_10_1000.csv', use_tanh=use_tanh, scale_tanh=scale_tanh)

optimizer = optim.Adam(list(country_classifier.parameters()) + list(geohead.parameters()))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5)

cell_loss = nn.CrossEntropyLoss()
coordinate_loss = nn.MSELoss()
pseudo_label_loss = nn.MSELoss()

clues = load_dataset("gips-mai/all_clues_enc")
descriptions = load_dataset("gips-mai/enc_descr")
data_loader = torch.utils.data.DataLoader(descriptions, batch_size=32, shuffle=True)

country_losses = []
geo_losses = []
for epoch in range(10):
    for batch in data_loader:
        imgs, descriptions, country_target, cell_target, coordinate_target = batch
        imgs, descriptions, country_target, cell_target, coordinate_target = imgs.to(device), descriptions.to(device), country_target.to(device), cell_target.to(device), coordinate_target.to(device)

        optimizer.zero_grad()

        attention = linear_attention.forward(img_embedding=imgs)
        weighted_aggregation = attention_aggregation.forward(clues, imgs, attention)
        country_loss = country_classifier.training_step(x=weighted_aggregation, target=country_encoding[labels['ISO2']]) # target: get the iso2 of actual country and then look at the one hot encoding
        country_losses.append(country_loss)

        aggregated_input = torch.cat([imgs, descriptions, weighted_aggregation], dim=1)

        intermediate = geohead.forward(aggregated_input)
        prediction = hybrid_head_centroid.forward(intermediate, cell_target)

        total_loss = cell_loss.apply(prediction['label'], cell_target) + coordinate_loss.apply(prediction['gps'], coordinate_target) + country_loss

        total_loss.backward()
        optimizer.step()
    
    scheduler.step()

    # Print the loss at each epoch
    print(f"Epoch {epoch+1}, Loss: {total_loss.item():.4f}")
    print(f"Epoch {epoch+1}, Loss: {country_loss.item():.4f}")
