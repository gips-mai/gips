from PIL import Image
import requests
from datasets import DatasetDict, Dataset
from dotenv import load_dotenv
import os
from pathlib import Path

from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPModel
import torch

load_dotenv()

# Hyperparameter
RAW_OSV5M_DIR = os.getenv("RAW_OSV5M_DIR")
RAW_OSV5M_DIR = Path(RAW_OSV5M_DIR)  # Convert to Path object because of windows
HF_AUTH_TOKEN = os.getenv('HF_AUTH_TOKEN')
model_name = 'geolocal/StreetCLIP'
batch_size = 128

def test_clip():
    model = CLIPModel.from_pretrained("geolocal/StreetCLIP").to("cuda")
    processor = CLIPImageProcessor.from_pretrained("geolocal/StreetCLIP")

    url = "https://huggingface.co/geolocal/StreetCLIP/resolve/main/sanfrancisco.jpeg"

    image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(images=image, return_tensors="pt").to("cuda")
    print(f"inputs: {inputs['pixel_values'].shape}")

    # Extract image features (image embeddings)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    print(f"image_features: {image_features.shape}")

def inference_loop():

    print(f"Processing images from {RAW_OSV5M_DIR}")

    # Initialize CLIP model and processor
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPImageProcessor.from_pretrained(model_name)

    # Set device and load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Device: {device}")


    splits = ['01', '02', '03', '04']
    full_enc_dataset = DatasetDict()

    for split in splits:

        # Set up variables for each split
        enc_split = []
        full_path = os.path.join(RAW_OSV5M_DIR, split)

        # Get all image files in the split directory
        image_files = [f for f in os.listdir(full_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

        # Process images in batches
        #for i in tqdm(range(0, len(image_files), batch_size), desc=f"Processing split {split}"):
        for i in tqdm(range(0, 256, batch_size), desc=f"Processing split {split}"):
            batch_files = image_files[i:i + batch_size]
            batch_images = []
            batch_ids = []

            # Extract images and IDs
            for img_file in batch_files:
                img_path = os.path.join(full_path, img_file)
                image = Image.open(img_path).convert('RGB')
                batch_images.append(image)
                batch_ids.append(img_file)

            # Process batch with CLIP
            inputs = processor(images=batch_images, return_tensors="pt").to(device)
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)

            # Store encodings in dataset dictionary
            for idx, img_id in enumerate(batch_ids):
                sample = {'img_id': img_id, 'encoding': image_features[idx].cpu().numpy()}
                enc_split.append(sample)

        # Add split to full dataset
        full_enc_dataset[split] = Dataset.from_list(enc_split)

    return full_enc_dataset

if __name__ == "__main__":
    #test_clip()
    full_enc_dataset = inference_loop()
    print(full_enc_dataset)

    # Save dataset locally
    #full_enc_dataset.save_to_disk("data/enc_img")

    # Upload dataset to Hugging Face
    #full_enc_dataset.push_to_hub('gips-mai/enc_img', token=HF_AUTH_TOKEN)
