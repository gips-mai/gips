## Data Processing

Given the raw data from travelguides, geoguessr clues from G^3 and image-coordinate pairs from the OSV5M dataset, we
need to preprocess the data to create a comprehensive dataset that can be used for training the model.
The data preprocessing is done in the following steps:

1. Clues:
   - Here we combine the clues of G^3 with our scraped travelguide clues to create a comprehensive dataset of clues the model can choose from with its attention module
   - For more detailed information on the combining process see the `create_clues_dataframe.ipynb` notebook 
   - The resulting dataset is encoded with the sentence transformer and uploaded to the huggingface hub in `upload_encoded_clues_dataset.ipynb`
