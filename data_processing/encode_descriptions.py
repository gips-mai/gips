from utils import encode_dataset, upload_on_hf
from datasets import Dataset, concatenate_datasets
import pandas as pd

# Load the dataset
df = pd.read_csv("raw_data/results_all_formated.csv")


# Define the unzip function and the keys to filter out
def unzip_fct(row):
    return {'img_id': row['image_file'], 'resp': row['response']}


# Encode the dataset in chunks to avoid memory issues
dataset_chunks = encode_dataset(df=df, unzip_fct=unzip_fct, data_key='resp', batch_size=32, writer_batch_size=10000)

# Aggregate the chunks
full_dataset = Dataset.from_list([])
for chunk in dataset_chunks:
    if full_dataset is None:
        full_dataset = chunk
    else:
        full_dataset = concatenate_datasets([full_dataset, chunk])

# Upload the dataset to the Hugging Face Hub
upload_on_hf(full_dataset, 'gips-mai/descriptions_enc')
