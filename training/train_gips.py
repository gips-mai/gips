import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
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

    pred = model(enc_img, enc_descr)

    print(f"Prediction: {pred.items()}")


def test_batched_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Gips(img_embedding_size=1024,
                 descript_embedding_size=768,
                 clue_embedding_size=768,
                 use_multimodal_inputs=True,
                 is_training=False).to(device)

    # Ensure parameters are in float32
    model = model.float()

    # Create dummy inputs
    data_loader = DataLoader(osv5m_ann['01'], batch_size=2)
    for batch in data_loader:
        # Reshape and transpose the data
        img_enc = torch.stack(batch['img_encoding']).t().float()
        text_enc = torch.stack(batch['desc_encoding']).t().float()

        print(f"Image encoding: {img_enc.shape}")
        print(f"Description encoding: {text_enc.shape}")

        img_enc_tensor = img_enc.to(device)
        text_enc_tensor = text_enc.to(device)

        pred = model(img_enc_tensor, text_enc_tensor)
        print(f"Prediction: {pred.items()}")

        break


def test_single_sample_training(epochs=2, use_multimodal_inputs=True):
    # fix random seed
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Gips(img_embedding_size=1024,
                 descript_embedding_size=768,
                 clue_embedding_size=768,
                 use_multimodal_inputs=use_multimodal_inputs,
                 is_training=True).to(device)

    if use_multimodal_inputs:
        params = (list(model.lat_long_head.geoloc_head_mid_network.parameters()) +
                  list(model.lin_att.parameters()) +
                  list(model.guiding_head.parameters()))
    else:
        params = model.geohead.geohead_mid_network.parameters()

    optimizer = optim.Adam(params)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5)

    dummy_dataset = load_dataset("gips-mai/osv5m_ann", split="01")
    data_loader = DataLoader(dummy_dataset, batch_size=2)

    for epoch in tqdm(range(epochs), desc="Epochs"):
        epoch_loss = 0.0

        for batch in tqdm(data_loader, desc="Batches"):
            optimizer.zero_grad()

            # Reshape and transpose the data
            img_enc = torch.stack(batch["img_encoding"]).t().float().to(device)
            text_enc = torch.stack(batch["desc_encoding"]).t().float().to(device)
            # Target cell is an index, therefore it should be an integer
            target_cell = torch.stack(list(batch["quadtree_10_1000"])).t().to(device).unsqueeze(dim=1)
            # The dataloader returns a list of one-hot encodings,
            # so we need to convert it to a numpy array and then to a tensor
            target_country = torch.tensor(np.array(batch["country_one_hot_enc"][0])).t().float().to(device)
            latitude_target = torch.stack(list(batch["latitude"])).t().float().to(device).unsqueeze(dim=1)
            longitude_target = torch.stack(list(batch["longitude"])).t().float().to(device).unsqueeze(dim=1)

            coordinate_target = torch.cat((latitude_target, longitude_target), dim=1).to(device)

            total_loss = model.get_losses(img_enc, text_enc, target_cell, target_country, coordinate_target)

            epoch_loss += total_loss.item()

            total_loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")


test_single_sample_training()


def train_gips(epochs=2, use_multimodal_inputs=True):
    # fix random seed
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Gips(img_embedding_size=1024,
                 descript_embedding_size=768,
                 clue_embedding_size=768,
                 use_multimodal_inputs=use_multimodal_inputs,
                 is_training=True).to(device)

    if use_multimodal_inputs:
        params = (list(model.lat_long_head.geoloc_head_mid_network.parameters()) +
                  list(model.lin_att.parameters()) +
                  list(model.guiding_head.parameters()))
    else:
        params = model.geohead.geohead_mid_network.parameters()

    optimizer = optim.Adam(params)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5)

    train_data = DataLoader(load_dataset("gips-mai/osv5m_ann", split='01'), batch_size=2)

    for epoch in tqdm(range(epochs), desc="Epochs"):
        epoch_loss = 0.0

        for split in ["01", "02", "03", "04"]:
            split_loss = 0.0

            train_data = DataLoader(load_dataset("gips-mai/osv5m_ann", split=split), batch_size=1)

            for batch in tqdm(train_data, desc="Batches"):
                optimizer.zero_grad()

                # Reshape and transpose the data
                img_enc = torch.stack(batch["img_encoding"]).t().float().to(device)
                text_enc = torch.stack(batch["desc_encoding"]).t().float().to(device)
                target_cell = torch.stack(list(batch["quadtree_10_1000"])).t().float().to(device).unsqueeze(dim=1)
                target_country = torch.stack(batch["country_one_hot_enc"]).t().float().to(device)
                latitude_target = torch.stack(batch["latitude"]).t().float().to(device)
                longitude_target = torch.stack(batch["longitude"]).t().float().to(device)

                coordinate_target = torch.cat((latitude_target, longitude_target), dim=1).to(device)

                total_loss = model.get_losses(img_enc, text_enc, target_cell, target_country, coordinate_target)

                split_loss += total_loss.item()
                epoch_loss += total_loss.item()

                total_loss.backward()
                optimizer.step()

            print(f"Epoch {epoch}, Split {split} - Loss: {split_loss:.4f}")

        use_multi = "use_multimodality" if use_multimodal_inputs else "no_multimodality"

        # Save the model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_loss
        }, os.path.join("checkpoints", "epoch_{}_split_{}_{}.pt".format(epoch, split, use_multi)))

        # Adapt the learning rate for every epoch
        scheduler.step()
