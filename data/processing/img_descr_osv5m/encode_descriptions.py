import os
from utils import encode_dataset, upload_on_hf
from datasets import Dataset, concatenate_datasets, DatasetDict
import pandas as pd

# Set the path to the directory containing the aggregated descriptions of each image directory
path = 'data/comb_descr'


def unzip_fct(row):
    """ Function to extract the image id and the response from a row of the dataset and return it as a dictionary."""
    return {'img_id': row['image_file'], 'resp': row['response']}


# Encode every split of the dataset and save it to disk
full_enc_dataset = DatasetDict()
# Iterate through files in the specified directory
for comb_split in os.listdir(path):
    if comb_split.startswith('comb_descr'):

        # Extract the split name from the filename
        comb_split_name = comb_split.split('.')[0]
        split_name = comb_split_name.split('_')[-1]
        print(f'Encoding: {split_name}')

        # Encode the split in chunks to avoid memory issues
        df = pd.read_csv(os.path.join(path, comb_split))
        dataset_chunks = encode_dataset(df=df, unzip_fct=unzip_fct, data_key='resp',
                                        batch_size=128, writer_batch_size=5000)

        # Aggregate the chunks into one split
        enc_split = Dataset.from_list([])
        for chunk in dataset_chunks:
            if enc_split is None:
                enc_split = chunk
            else:
                enc_split = concatenate_datasets([enc_split, chunk])

        # Add the encoded split to the full dataset
        full_enc_dataset[split_name] = enc_split
        full_enc_dataset.save_to_disk('data/enc_descr')

# Save the full encoded dataset locally
full_enc_dataset.save_to_disk('data/enc_descr')
print(full_enc_dataset)

# Upload the dataset to the Hugging Face Hub
upload_on_hf(full_enc_dataset, 'gips-mai/enc_descr')
