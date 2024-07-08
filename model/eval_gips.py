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
from transformers import AutoModel
from transformers import pipeline
import joblib
from huggingface_hub import hf_hub_download
from huggingface_hub import PyTorchModelHubMixin
from dotenv import load_dotenv
import os

load_dotenv()
HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")

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

    for option, use_multimodal_inputs in zip(['gips', 'gips_reg_head_no_multimod', 'gips_baseline'], [True, False, False]):

        metric = Metric()
        REPO_ID = "gips-mai"

        model = Gips(img_embedding_size=1024,
                 descript_embedding_size=768,
                 clue_embedding_size=768,
                 use_multimodal_inputs=use_multimodal_inputs,
                 is_training=False).to(device)

        if option == 'gips':
            model.from_pretrained('gips-mai/'+option)
        elif option == 'gips_reg_head_no_multimod':
            model.from_pretrained('gips-mai/'+option)
        elif option == 'gips_baseline':
            model.from_pretrained('gips-mai/'+option)
        else:
            raise NotImplementedError
        
        loss = 0.0
        lat_long_loss = 0.0
        guiding_loss = 0.0

        dataset = filter_dataset(load_dataset("gips-mai/osv5m_ann", split="00"))

        dataset = dataset.select(
            i for i in range(len(dataset))
            if dataset[i]["country_one_hot_enc"][0] is not None
        )

        # dataset = dataset[:8]

        batch_size=128
        for idx, batch in enumerate(tqdm(DataLoader(dataset, batch_size=batch_size, shuffle=True), desc=option)):

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
        
        m = metric.compute()
        evaluation[option] = {
            'loss': loss,
            'lat_long_loss': lat_long_loss,
            'guiding_loss': guiding_loss,
            'n_loss': loss/len(dataset),
            'n_lat_long_loss': lat_long_loss/len(dataset),
            'n_guiding_loss': guiding_loss/len(dataset),
            'len_data': len(dataset),
            'metric': m
        }
    
    with open(str(Path(__file__).parent.parent / "data" / "eval.json"), "w") as f:
        json.dump(evaluation, f)

if __name__ == '__main__':
    batched_evaluation_gips()
