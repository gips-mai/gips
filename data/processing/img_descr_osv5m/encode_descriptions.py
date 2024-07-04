import os

from utils import encode_dataset, upload_on_hf
from datasets import Dataset, concatenate_datasets, DatasetDict
import pandas as pd


path = 'data/comb_descr'
def unzip_fct(row):
    return {'img_id': row['image_file'], 'resp': row['response']}


# Encode every split of the dataset and save it to disk

full_enc_dataset = DatasetDict()
for comb_split in os.listdir(path):
    if comb_split.startswith('comb_descr'):

        # Extract the split name
        comb_split_name = comb_split.split('.')[0]
        split_name = comb_split_name.split('_')[-1]
        print(f'Encoding: {split_name}')

        # Enocde the split in chunks to avoid memory issues
        df = pd.read_csv(os.path.join(path, comb_split))
        dataset_chunks = encode_dataset(df=df, unzip_fct=unzip_fct, data_key='resp',
                                        batch_size=128, writer_batch_size=5000)

        # Aggregate the chunks to one split
        enc_split = Dataset.from_list([])
        for chunk in dataset_chunks:
            if enc_split is None:
                enc_split = chunk
            else:
                enc_split = concatenate_datasets([enc_split, chunk])

        # Add split to the full dataset
        full_enc_dataset[split_name] = enc_split
        full_enc_dataset.save_to_disk('data/enc_descr')


print(full_enc_dataset)
# Upload the dataset to the Hugging Face Hub
upload_on_hf(full_enc_dataset, 'gips-mai/enc_descr')
