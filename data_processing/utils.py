from datasets import Dataset
from tqdm import tqdm
import torch
from transformers import RobertaTokenizer, RobertaModel
import os
from dotenv import load_dotenv

# Definitions
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base').to('cuda')
load_dotenv()
HF_AUTH_TOKEN = os.getenv('HF_AUTH_TOKEN')


def encode_dataset(df, unzip_fct, data_key, filter_keys=None, batch_size=32, writer_batch_size=10000):
    """ Encode a dataset using a given tokenizer and model. The dataset is expected to be a pandas DataFrame.
    Args:
        df: pd.DataFrame, the dataset to encode
        unzip_fct: function, the function to use to unzip a row, must return a dictionary
        data_key: str, the key of the raw_data to encode in the sample
        filter_keys: list of str, keys which should be removed from the sample before adding the encoding
        batch_size: int, the number of samples to process at once
        writer_batch_size: int, the batch size for writing the dataset
    Returns:
        encodings: datasets.Dataset, the encoded dataset """

    encodings = []

    # Process the data in batches
    for i in tqdm(range(0, len(df), batch_size), desc="Encoding batches"):
        batch = df.iloc[i:i + batch_size]
        batch_samples = [unzip_fct(row) for _, row in batch.iterrows()]

        # Prepare inputs for the batch
        inputs = tokenizer([sample[data_key] for sample in batch_samples], return_tensors='pt', truncation=True,
                           padding=True).to('cuda')

        # Encode the batch
        with torch.no_grad():
            outputs = model(**inputs)
        batch_encodings = outputs.last_hidden_state.cpu().numpy()  # Extract last hidden state as the encoding

        # Process each sample in the batch
        for j, sample in enumerate(batch_samples):
            encoding_dict = sample.copy()  # Copy the sample
            if filter_keys is not None:
                for key in filter_keys:
                    encoding_dict.pop(key, None)
            encoding_dict['encoding'] = batch_encodings[j]
            encodings.append(encoding_dict)

        # If we've accumulated enough samples, create a dataset and reset
        # This is done to avoid memory issues with too many samples
        if len(encodings) >= writer_batch_size:
            yield Dataset.from_list(encodings)
            encodings = []

    # Don't forget any remaining samples
    if encodings:
        yield Dataset.from_list(encodings)


def upload_on_hf(dataset, path):
    """ Upload a dataset on the Hugging Face Hub
    Args:
        dataset: datasets.Dataset, the dataset to upload on the Hub
        path: str, the path to save the dataset """

    # Save the dataset
    dataset.push_to_hub(path, token=HF_AUTH_TOKEN)
