## Data Processing

Given the raw data from travelguides, geoguessr clues from G^3 and image-coordinate pairs from the OSV5M dataset, we
need to preprocess the data to create a comprehensive dataset that can be used for training the model.
The data preprocessing is done in the following steps:

1. Clues:
    - Here we combine the clues of G^3 with our scraped travelguide clues to create a comprehensive dataset of clues the
      model can choose from with its attention module
    - For more detailed information on the combining process see the `create_clues_dataframe.ipynb` notebook
    - The resulting dataset is encoded with the sentence transformer and uploaded to the huggingface hub
      in `upload_encoded_clues_dataset.ipynb`

2. OSV5M descriptions:
    - Here we create the natural language descriptions of the images from the OSV5M dataset
    - For this we use the split the process into 3 steps:
        1. Create the descriptions
        2. Format the descriptions
        3. Encode the descriptions and upload them to the huggingface hub

3. OSV5M image encodings:
    - Here we encode the images from the OSV5M dataset using the CLIP model and upload them to the huggingface hub
    - We extracted the weights of the finetuned CLIP model used in
      the [OSV5M baseline model](https://huggingface.co/osv5m/baseline) and saved them in a new clip model which is also
      uploaded to the huggingface hub [here](https://huggingface.co/gips-mai/clip_finetuned_osv5m)
    - We have no separate file for this in the repository as the extraction required cloning into the OSV5M repository,
      then installing the package from source and running this extraction script:
        ```python
      from models.huggingface import Geolocalizer
      from transformers import CLIPVisionModel
      
      # Load the Geolocalizer model
      geolocalizer = Geolocalizer.from_pretrained('osv5m/baseline')
      
      # extract clip config
      clip_config = geolocalizer.backbone.clip.config
      clip_model = geolocalizer.backbone.clip
      
      clip_model.save_pretrained('clip_finetuned_osv5m')
      model = CLIPVisionModel.from_pretrained('clip_finetuned_osv5m')
      
      # upload to huggingface
      model.push_to_hub('gips-mai/clip_finetuned_osv5m')```

4. Combining the datasets:
   - Lastly, we download the encoded descriptions and images from the hub and combine them with the original OSV5M dataset
     to create a comprehensive dataset for training the model
   - This allows us during training time to circumvent the compute intensive encoding process and directly use the encoded
     data for training