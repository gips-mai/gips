from pathlib import Path
import sys
sys.path.append(Path.cwd().parent)

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from model.modules.attention_module import AttentionWeightedAggregation, LinearAttention
from model.modules.heads.country_pred_head import CountryClassifier
from model.modules.heads.geoloc_head import MLPCentroid, HybridHeadCentroid

from model.modules.attention_module import get_pseudo_label_loss

from datasets import load_dataset

### HYPER PARAMETERS ###
lr = 0.001
alpha = 0.75
use_tanh = True
scale_tanh = 1.2
### HYPER PARAMETERS ###

device = 'cpu'

clue_embedding_size:int = 768
text_embedding_size:int = 512
clip_embedding_size:int = 716

country_encoding = pd.read_csv('../data/encodings.csv')

attention_aggregation = AttentionWeightedAggregation(temperature=0.01) #TODO definde temperature
linear_attention = LinearAttention(attn_input_img_size=clip_embedding_size, text_features_size=clue_embedding_size, hidden_layer_size_0=1024, hidden_layer_size_1=1024) #TODO hidden layer size
country_classifier = CountryClassifier(clue_embedding_size=clue_embedding_size, image_embedding_size=clip_embedding_size, alpha=alpha)

previous_stage_output = text_embedding_size+clip_embedding_size+clue_embedding_size
geohead = MLPCentroid(initial_dim=previous_stage_output, hidden_dim=[previous_stage_output, 1024, 512])
hybrid_head_centroid = HybridHeadCentroid(final_dim=11398, quadtree_path='../data/quad_tree/quadtree_10_1000.csv', use_tanh=use_tanh, scale_tanh=scale_tanh)

optimizer = optim.Adam(list(country_classifier.parameters()) + list(geohead.parameters()))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5)

cell_loss = nn.CrossEntropyLoss()
coordinate_loss = nn.MSELoss()

# clues = load_dataset("gips-mai/all_clues_enc")['train'][:20]
# clue_embeddings = torch.tensor(clues['encoding'])

clue_embeddings = pd.read_pickle('../data/guidebook_roberta_base_ch_in.pkl')
clue_embeddings = torch.tensor(list(clue_embeddings.values()))
clues = load_dataset("gips-mai/all_clues_enc")['train'][:len(clue_embeddings)]
clues['encoding'] = None

# clues = load_dataset("gips-mai/all_clues_enc")['train'][:2]

pseudo_label_loss = get_pseudo_label_loss(clues["country_one_hot_enc"])
# descriptions = load_dataset("gips-mai/enc_descr")
# data_loader = torch.utils.data.DataLoader(descriptions, batch_size=32, shuffle=True)

dataset = build_dataset()
data_loader = torch.utils.data.DataLoader(dataset, batch_size = 16, shuffle=False)

country_losses = []
geo_losses = []
for epoch in range(10):
    for batch in data_loader:
        descriptions, imgs, country_target, cell_target, coordinate_target = batch
        imgs, descriptions, country_target, cell_target, coordinate_target = imgs.to(device), descriptions.to(device), country_target.to(device), cell_target.to(device), coordinate_target.to(device)

        optimizer.zero_grad()

        attention = linear_attention.forward(img_embedding=imgs)
        weighted_aggregation = attention_aggregation.forward(clue_embeddings=clue_embeddings, attention=attention)

        #TODO: debug from here
        country_loss = country_classifier.training_step(x=weighted_aggregation, target=country_target) # target: get the iso2 of actual country and then look at the one hot encoding
        country_losses.append(country_loss)

        # pseudo label loss
        current_pseudo_label_loss = pseudo_label_loss(country_target, attention)

        aux_attention_loss = alpha * current_pseudo_label_loss + (1-alpha) * country_loss

        aggregated_input = torch.cat([imgs, descriptions, weighted_aggregation], dim=1)

        intermediate = geohead.forward(aggregated_input)
        prediction = hybrid_head_centroid.forward(intermediate, cell_target)

        total_loss = cell_loss.apply(prediction['label'], cell_target) + \
                     coordinate_loss.apply(prediction['gps'], coordinate_target) + \
                     aux_attention_loss

        total_loss.backward()
        optimizer.step()
    
    scheduler.step()

    # Print the loss at each epoch
    print(f"Epoch {epoch+1}, Loss: {total_loss.item():.4f}")
    print(f"Epoch {epoch+1}, Loss: {country_loss.item():.4f}")


def build_dataset():
    # Description Emb., Img Emb., Country Hot Encoding, Cell Target, Coordinate Target (Lat, Lon)
    n = 124
    descripts = [torch.randn(716).numpy() for i in range(n)]
    img_embeds = [torch.randn(716).numpy() for i in range(n)]

    country_encodings = []
    for i in range(n):
        enc = torch.zeros(221).numpy()
        enc[torch.randint(0, 221, (1, ))[0]] = 1
        country_encodings.append(enc)

    cell_targets = [torch.randint(0, 10000, (1, ))[0].numpy() for i in range(n)]

    coordinate_targets = []
    for i in range(n):
        enc = torch.randn(2).numpy()
        enc[0] *= 180
        enc[1] *= 90
        coordinate_targets.append(enc)

    return CustomDataset(pd.DataFrame(list(zip(descripts, img_embeds, country_encodings, cell_targets, coordinate_targets)),
               columns =['descriptions', 'img_emb', "country_enc", "cell_target", "coordinate_target"]))


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        descript = torch.tensor(row['descriptions'])
        img_embed = torch.tensor(row['img_emb'])
        country_encoding = torch.tensor(row['country_enc'])
        cell_target = torch.tensor(row['cell_target'])
        coordinate_target = torch.tensor(row['coordinate_target'])

        return descript, img_embed, country_encoding, cell_target, coordinate_target

