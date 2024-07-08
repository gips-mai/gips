import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from gips import Gips
from torch.utils.data import Dataset
from datasets import load_dataset
import pandas
from dotenv import load_dotenv
from torch.utils.tensorboard import SummaryWriter

load_dotenv()
HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
osv5m_ann = load_dataset("gips-mai/osv5m_ann")


def test_upload():
    model = Gips(img_embedding_size=1024,
                 descript_embedding_size=768,
                 clue_embedding_size=768,
                 use_multimodal_inputs=True,
                 is_training=False)
    model.push_to_hub("gips-mai/gips", token=HF_AUTH_TOKEN)


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


def filter_dataset(dummy_dataset):
    # Filter the dataset to keep only the specified columns
    columns_to_keep = ["img_encoding", "desc_encoding", "quadtree_10_1000", "country_one_hot_enc", "latitude",
                       "longitude"]

    def filter_columns(example):
        return {key: example[key] for key in columns_to_keep if key in example}

    filtered_dataset = dummy_dataset.map(filter_columns, remove_columns=[col for col in dummy_dataset.column_names if
                                                                         col not in columns_to_keep])
    return filtered_dataset


def batched_training_gips(epochs=2, use_multimodal_inputs=True, use_reg_head=True):
    # fix random seed
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Gips(img_embedding_size=1024,
                 descript_embedding_size=768,
                 clue_embedding_size=768,
                 use_multimodal_inputs=use_multimodal_inputs,
                 is_training=True,
                 use_reg_head=use_reg_head).to(device)

    if use_reg_head:
        lat_long_params = model.lat_long_head.parameters()
    else:
        lat_long_params = model.lat_long_head.geoloc_head_mid_network.parameters()

    if use_multimodal_inputs:
        params = (list(lat_long_params) +
                  list(model.lin_att.parameters()) +
                  list(model.guiding_head.parameters()))
    else:
        params = lat_long_params

    optimizer = optim.Adam(params, lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Initialize TensorBoard writer
    writer = SummaryWriter(f'runs/gips_training_multimodal_{use_multimodal_inputs}')

    for epoch in tqdm(range(epochs), desc="Epochs"):
        epoch_loss = 0.0
        batch_count = 0

        for split in ["01", "02", "03", "04"]:
            dummy_dataset = filter_dataset(load_dataset("gips-mai/osv5m_ann", split=split))

            dummy_dataset = dummy_dataset.select(
                i for i in range(len(dummy_dataset))
                if dummy_dataset[i]["country_one_hot_enc"][0] is not None
            )

            for batch_idx, batch in enumerate(
                    tqdm(DataLoader(dummy_dataset, batch_size=2, shuffle=True), desc="Batches")):
                optimizer.zero_grad()

                # Reshape and transpose the data
                img_enc = torch.stack(batch["img_encoding"]).t().float().to(device)
                text_enc = torch.stack(batch["desc_encoding"]).t().float().to(device)
                target_cell = torch.stack(list(batch["quadtree_10_1000"])).t().to(device).unsqueeze(dim=1)
                target_country = torch.tensor(np.array(batch["country_one_hot_enc"][0])).t().float().to(device)
                latitude_target = torch.stack(list(batch["latitude"])).t().float().to(device).unsqueeze(dim=1)
                longitude_target = torch.stack(list(batch["longitude"])).t().float().to(device).unsqueeze(dim=1)

                coordinate_target = torch.cat((latitude_target, longitude_target), dim=1).to(device)

                total_loss = model.get_losses(img_enc, text_enc, target_cell, target_country, coordinate_target)

                epoch_loss += total_loss.item()
                batch_count += 1

                total_loss.backward()
                optimizer.step()

                # Log batch loss
                writer.add_scalar('Loss/batch', total_loss.item(), epoch * len(dummy_dataset) + batch_idx)

        # Log epoch loss
        avg_epoch_loss = epoch_loss / batch_count
        writer.add_scalar('Loss/epoch', avg_epoch_loss, epoch)

        # Log learning rate
        writer.add_scalar('Learning rate', scheduler.get_last_lr()[0], epoch)

        #model.push_to_hub("gips-mai/gips_1", token=HF_AUTH_TOKEN)

        print(f"Epoch {epoch} - Loss: {avg_epoch_loss}")
        scheduler.step()

    writer.close()
    print("Training finished. TensorBoard logs saved.")


if __name__ == '__main__':
    # test_inference()
    # test_batched_inference()
    batched_training_gips(epochs=10, use_multimodal_inputs=False)
