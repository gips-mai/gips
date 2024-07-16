# GIPS: Global Image Positioning System

This is the repository with the code for the Mulimodal AI bonus project of group J.
The appraoch is called Global Image Positioning System (GIPS), which is a novel appraoch for geolocation via mulitmodal
LLMs, fusing image and text data.

## Repository Structure

The repository is structured as follows:

- [data](data): Overall data folder
    - [acquisition](data%2Facquisition): Contains the data acquisition scripts for the travelguides
    - [processing](data%2Fprocessing)`: Contains the preprocessing scripts used to create the data which is used for model training:
    - [quad_tree](data%2Fquad_tree): Contains the quad tree file used for the advanced classification head
- [evaluation](evaluation): Contains the evaluation scripts for the model 
- [model](model): Containes the scripts for building the model
- [training](training): Contains the training scripts for the model
- [utils](utils): Contains utility functions used for the metrics and the model

## Data Preprocessing
- Combining the clues from the travelguides and the G^3 data:
  - We combined the clues from the travelguides and the G^3 data to create a single dataset with the clues. These are used as static input to the model for every input image.
  - The idea is to provide the model with additional information about how likely it is that an image is in a certain country based on the clues.
  - Script can be found [here](data%2Fprocessing%2Fclues%2Fcreate_clues_dataframe.ipynb)
  - We additionally create an ISO-2 country code, which we use in the training process of the attention module to indicate to the model the set of clues that should have been used to predict the correct location.
  - 



All datasets and trained models are uploaded to the [gips-mai huggingface repository](https://huggingface.co/gips-mai) and can be accessed via the Huggingface API.