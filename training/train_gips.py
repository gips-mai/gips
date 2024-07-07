import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.gips import Gips

from datasets import load_dataset
osv5m_ann = load_dataset("gips-mai/osv5m_ann")

def test_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Gips(img_embedding_size=1024,
                 descript_embedding_size=768,
                 clue_embedding_size=768,
                 use_multimodal_inputs=True,
                 is_training=False).to(device)

    # Create dummy inputs
    sample = osv5m_ann['01'][0]

    enc_descr = torch.tensor(sample['desc_encoding'], dtype=torch.float32).unsqueeze(0).to(device)
    print(f"Description encoding: {enc_descr.shape}")
    enc_img = torch.tensor(sample['img_encoding'], dtype=torch.float32).unsqueeze(0).to(device)
    print(f"Image encoding: {enc_img.shape}")

    pred = model(enc_img,  enc_descr)

    print(f"Prediction: {pred.items()}")

def test_batched_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Gips(img_embedding_size=1024,
                 descript_embedding_size=768,
                 clue_embedding_size=768,
                 use_multimodal_inputs=True,
                 is_training=False).to(device)

    # Create dummy inputs
    data_loader = DataLoader(osv5m_ann['01'], batch_size=2)
    for batch in data_loader:
        print(batch)

        img_enc, text_enc, target_cell, target_country = (batch["img_encoding"],
                                                          batch["desc_encoding"],
                                                          batch["quadtree_10_1000"],
                                                          batch["country_one_hot_enc"])

        # Reshape and transpose the data
        img_enc = torch.stack(batch['img_encoding']).t()
        text_enc = torch.stack(batch['desc_encoding']).t()

        img_enc_tensor = img_enc.to(device)
        text_enc_tensor = text_enc.to(device)

        pred = model(img_enc_tensor, text_enc_tensor)
        print(f"Prediction: {pred.items()}")

        break

test_batched_inference()

def train_gips(epochs=10, use_multimodal_inputs=True):

    model = Gips(img_embedding_size=1024,
                 descript_embedding_size=768,
                 clue_embedding_size=768,
                 use_multimodal_inputs=use_multimodal_inputs,
                 is_training=True)

    device = 'cuda'

    if use_multimodal_inputs:
        params = (list(model.geohead.geohead_mid_network.parameters()) +
                  list(model.lin_att.parameters()) +
                  list(model.guiding_head.parameters()))
    else:
        params = model.geohead.geohead_mid_network.parameters()

    optimizer = optim.Adam(params)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5)

    train_data = DataLoader(load_dataset("gips-mai/osv5m_ann", split='01'), batch_size=2)

    for epoch in range(epochs):
        for batch in train_data:

            optimizer.zero_grad()

            img_enc, text_enc, target_cell, target_country = (batch["img_encoding"],
                                                             batch["desc_encoding"],
                                                             batch["quadtree_10_1000"],
                                                             batch["country_one_hot_enc"])
            coordinate_target = torch.cat((batch["latitude"], batch["longitude"]), dim=1)

            img_enc, text_enc, target_country, target_cell, coordinate_target = img_enc.to(device), \
                                                                                text_enc.to(device), \
                                                                                target_country.to(device), \
                                                                                target_cell.to(device), \
                                                                                coordinate_target.to(device)

            total_loss = model.get_losses(img_enc, text_enc, target_cell, target_country, coordinate_target)

            total_loss.backward()
            optimizer.step()

        scheduler.step()


