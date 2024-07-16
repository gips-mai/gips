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

Each of the folders contains a README file with more detailed information about the content of the respective folder.

All datasets and trained models are uploaded to the [gips-mai huggingface repository](https://huggingface.co/gips-mai) and can be accessed via the Huggingface API to download and use.

Delicate information, such as authentication tokens are stored in the `.env` file and are not uploaded to the repository.