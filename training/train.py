from pathlib import Path
import sys
sys.path.append(Path.cwd().parent)

import pandas as pd
import torch
import torch.nn as nn

from model.attention_module import AttentionWeightedAggregation, LinearAttention
from model.backbone import LatLongHead, LocationAttention, StreetCLIP, TextEncoder #TODO remove unused imports
from model.country_prediction import CountryClassifier
from model.geolocation_head import MLPCentroid

labels = pd.read_csv('path/to/labels.csv')
country_encoding = pd.read_csv('path/to/country/encoding.csv')

clues = pd.read_pickle("path/to/clues.pkl")
clue_embedding_size:int = 512 #TODO adjust

img_path = 'path/to/img'
img_embedding_size:int = 512 #TODO adjust

encoded_descriptions = pd.read_csv('path/to/encoded/descriptions.csv')
text_embedding_size:int = 2000 #TODO change to actual size

clip = StreetCLIP('path/to/model')
clip_output_size = 2000 #TODO change to actual size
attention_aggregation = AttentionWeightedAggregation() #TODO definde temperature
linear_attention = LinearAttention(attn_input_img_size=clip_output_size, text_features_size=text_embedding_size, hidden_layer_size_0=1024, hidden_layer_size_1=1024) #TODO hidden layer size
country_classifier = CountryClassifier(clue_embedding_size=clue_embedding_size, image_embedding_size=img_embedding_size)
geohead = MLPCentroid()


country_losses = []
geo_losses = []
for epoch in range(100):
    for i, (imgs, descriptions) in enumerate(zip(images, descriptions)):
        #TODO add checkpoints
        clip_encodings = clip.forward(imgs) #TODO condition output on text description
        attention = linear_attention.forward(img_embedding=clip_encodings)
        weighted_aggregation = attention_aggregation.forward(clues, clip_encodings, attention)
        country_loss = country_classifier.training_step(x=weighted_aggregation, target=country_encoding[labels['ISO2']]) # target: get the iso2 of actual country and then look at the one hot encoding
        country_losses.append(country_loss)
        #TODO adjust parameters for country_classifier

        aggregated_input = weighted_aggregation + clip_encodings

        geo_loss = geohead.training_step(aggregated_input, labels[i]['coordinates'])
        geo_losses.append(geo_loss)
        #TODO adjust parameters for geolocation head

        print(f"{i}:\ngeo_loss:     {geo_loss:.4f}\ncountry_loss: {country_loss:.4f}")
