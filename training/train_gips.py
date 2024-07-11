import os
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from model.gips import Gips
from datasets import load_dataset
from dotenv import load_dotenv
from torch.utils.tensorboard import SummaryWriter

load_dotenv()
HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
osv5m_ann = load_dataset("gips-mai/osv5m_ann")


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
    model_id = "gips_reg_head_multimod"
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ######## Initialize model structure based on experiment setting #################

    model = Gips(img_embedding_size=1024,
                 descript_embedding_size=768,
                 clue_embedding_size=768,
                 use_multimodal_inputs=use_multimodal_inputs,
                 is_training=True,
                 use_reg_head=use_reg_head).to(device)

    # Set model to train
    model.train()
    ###################################################################################

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


    # Init optimizer and splits
    optimizer = optim.Adam(params, lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Initialize TensorBoard writer
    writer = SummaryWriter(f'runs/{model_id}_multimodal_{use_multimodal_inputs}')

    for epoch in tqdm(range(epochs), desc=f"Epochs - {model_id}"):
        epoch_loss = 0.0
        batch_count = 0

        # OSV5m is separated into splits
        for split in ["01", "02", "03", "04"]:

            training_data = filter_dataset(load_dataset("gips-mai/osv5m_ann", split=split))

            # Remove invalid country encodings
            training_data = training_data.select(
                i for i in range(len(training_data))
                if training_data[i]["country_one_hot_enc"][0] is not None
            )

            for batch_idx, batch in enumerate(
                    tqdm(DataLoader(training_data, batch_size=32, shuffle=True), desc=f"Batches - {model_id}")):
                optimizer.zero_grad()

                # Reshape and transpose the data
                img_enc = torch.stack(batch["img_encoding"]).t().float().to(device)
                text_enc = torch.stack(batch["desc_encoding"]).t().float().to(device)
                target_cell = torch.stack(list(batch["quadtree_10_1000"])).t().to(device).unsqueeze(dim=1)
                target_country = torch.tensor(np.array(batch["country_one_hot_enc"][0])).t().float().to(device)
                latitude_target = torch.stack(list(batch["latitude"])).t().float().to(device).unsqueeze(dim=1)
                longitude_target = torch.stack(list(batch["longitude"])).t().float().to(device).unsqueeze(dim=1)

                coordinate_target = torch.cat((latitude_target, longitude_target), dim=1).to(device)

                # Get combination of location prediction loss + country prediction loss + pseudo label loss
                total_loss = model.get_individual_losses(img_enc, text_enc, target_cell, target_country, coordinate_target)[1]

                epoch_loss += total_loss.item()
                batch_count += 1

                total_loss.backward()

                # Check gradient norms
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                writer.add_scalar('Gradients/norm', total_norm, epoch * len(training_data) + batch_idx)

                optimizer.step()

                # Log batch loss
                writer.add_scalar('Loss/batch', total_loss.item(), epoch * len(training_data) + batch_idx)

        # Log epoch loss
        avg_epoch_loss = epoch_loss / batch_count
        writer.add_scalar('Loss/epoch', avg_epoch_loss, epoch)

        # Log learning rate
        writer.add_scalar('Learning rate', scheduler.get_last_lr()[0], epoch)

        # Upload to hugginface
        model.push_to_hub(f"gips-mai/{model_id}", token=HF_AUTH_TOKEN)

        print(f"Epoch {epoch} - Loss: {avg_epoch_loss}")
        scheduler.step()

    writer.close()
    print("Training finished. TensorBoard logs saved.")


if __name__ == '__main__':
    batched_training_gips(epochs=10, use_multimodal_inputs=True, use_reg_head=True)
