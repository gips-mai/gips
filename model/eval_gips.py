from pathlib import Path
import sys
sys.path.append(str(Path(__file__).cwd().parent))
sys.path.append(str(Path(__file__).cwd()))

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from gips import Gips
from datasets import load_dataset
from utils.metrics import Metric
import json

# import os
# from torch.utils.data import Dataset
# import torch.nn as nn
# import torch.optim as optim
# import pandas
# from dotenv import load_dotenv
# load_dotenv()
# HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")


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


def batched_evaluation_gips():
    import pickle
    # fix random seed
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    evaluation = {}

    for use_multimodal_inputs in [True, False]:

        metric = Metric()

        model = Gips(img_embedding_size=1024,
                    descript_embedding_size=768,
                    clue_embedding_size=768,
                    use_multimodal_inputs=use_multimodal_inputs,
                    is_training=True).to(device)

        loss = 0.0
        lat_long_loss = 0.0
        guiding_loss = 0.0

        dataset = filter_dataset(load_dataset("gips-mai/osv5m_ann", split="01")) #TODO: load test dataset

        dataset = dataset.select(
            i for i in range(len(dataset))
            if dataset[i]["country_one_hot_enc"][0] is not None
        )

        # dataset = dataset[:8]

        batch_size=128
        for idx, batch in enumerate(DataLoader(dataset, batch_size=batch_size, shuffle=False)):

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

            prediction, total_loss, ll_loss, g_loss = model.get_individual_losses(img_enc, text_enc, target_cell, target_country, coordinate_target)

            metric.update(prediction.pred_geoloc_head['gps'], coordinate_target)

            loss += total_loss.item()
            lat_long_loss += ll_loss.item()
            if use_multimodal_inputs:
                guiding_loss += g_loss.item()

            if use_multimodal_inputs:
                print('Multimodal Input:')
            else:
                print('Just Images:')
            # çprint(f"    Total loss:    {loss/((idx+1)*batch_size)}")
            print(f"    Lat Long Loss: {lat_long_loss/((idx+1)*batch_size)}")
            if use_multimodal_inputs:
                print(f"    Guiding Loss:  {guiding_loss/((idx+1)*batch_size)}")
            print()

            print('Metric:', metric.compute())

            break

        if use_multimodal_inputs:
            print('Multimodal Input:')
        else:
            print('Just Images:')
        # print(f"    Total loss:    {loss}")
        print(f"    Lat Long Loss: {lat_long_loss}")
        if use_multimodal_inputs:
            print(f"    Guiding Loss:  {guiding_loss}")

        if use_multimodal_inputs:
            s = 'multimodal_input'
        else:
            s = 'image_only'

        evaluation[s] = {
            'loss': loss,
            'lat_long_loss': lat_long_loss,
            'guiding_loss': guiding_loss,
            'metric': metric.compute(),
        }

    print(evaluation)
    
    with open("../data/eval.json", "w") as f:
        json.dump(evaluation, f)

if __name__ == '__main__':
    batched_evaluation_gips()
