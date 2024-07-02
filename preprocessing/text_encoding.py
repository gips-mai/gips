# Install the necessary libraries
import pandas as pd
import pickle
# Import the necessary libraries
from transformers import RobertaTokenizer, RobertaModel
import torch


def encode_with_Roberta(strings:str, output_path:str):
    # Load the pre-trained RoBERTa model and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')

    encodings = []

    for s in strings:
        # Tokenize the text and convert it to input IDs
        
        inputs = tokenizer(s, return_tensors='pt', truncation=True)

        # Get the embeddings from the model
        with torch.no_grad():
            outputs = model(**inputs)

        # The outputs include the last hidden state, pooler output, and hidden states
        # We are interested in the last hidden state
        encodings.append(outputs.last_hidden_state)

    with open(output_path, "wb") as f:
        pickle.dump(encodings, f)


def load_texts(text_file):

    with open(text_file, "r") as file:
        texts = file.readlines()
    
    return texts

def load_csv(path:str, text_column:str):
    df = pd.read_csv(path)
    df.dropna(axis=0, inplace=True)

    return list(df.loc[:,text_column])

if __name__ == '__main__':
    # txt_file = "data/results.txt"
    # text = load_texts(txt_file)

    csv_file = 'data/country_info_filtered_index.csv'
    text = load_csv(csv_file, 'Weather and Geography')
    encode_with_Roberta(strings=text, output_path="data/country_encoded.pkl")
