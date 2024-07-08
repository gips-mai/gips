import datasets
from PIL import Image
import requests
from datasets import DatasetDict, Dataset, load_dataset
from dotenv import load_dotenv
import os
from pathlib import Path

from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModel
import torch

load_dotenv()

# Hyperparameter
RAW_OSV5M_DIR = os.getenv("RAW_OSV5M_DIR")
RAW_OSV5M_DIR = Path(RAW_OSV5M_DIR)  # Convert to Path object because of windows
HF_AUTH_TOKEN = os.getenv('HF_AUTH_TOKEN')
processor_name = 'geolocal/StreetCLIP'
model_name = 'gips-mai/clip_finetuned_osv5m'  # Adapted from osv5m paper
batch_size = 128


def test_clip():
    model = CLIPVisionModel.from_pretrained(model_name).to("cuda")
    processor = CLIPImageProcessor.from_pretrained(processor_name)

    url = "https://huggingface.co/geolocal/StreetCLIP/resolve/main/sanfrancisco.jpeg"

    image = Image.open(requests.get(url, stream=True).raw)

    # Get the model's config to determine the expected image size
    model_config = model.config
    image_size = model_config.image_size
    print(f"Expected image_size: {image_size}")

    # Resize the image explicitly
    image = image.resize((image_size, image_size))

    # Process the image without resizing
    inputs = processor(images=image, return_tensors="pt", do_resize=False, do_center_crop=False)
    inputs = inputs.to("cuda")
    print(f"inputs: {inputs['pixel_values'].shape}")

    # Extract image features (image embeddings)
    with torch.no_grad():
        image_features = model(**inputs)

    print(f"image_features: {image_features.pooler_output.shape}")


def init_model():
    # Initialize CLIP model and processor
    model = CLIPVisionModel.from_pretrained(model_name).to("cuda")
    processor = CLIPImageProcessor.from_pretrained(processor_name)

    # Get the model's config to determine the expected image size
    model_config = model.config
    image_size = model_config.image_size

    # Set device and load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Device: {device}")

    return model, processor, image_size, device


def inference_loop():
    print(f"Processing images from {RAW_OSV5M_DIR}")

    model, processor, image_size, device = init_model()

    splits = ['01', '02', '03', '04']
    full_enc_dataset = DatasetDict()

    for split in splits:
        # Set up variables for each split
        enc_split = []
        full_path = os.path.join(RAW_OSV5M_DIR, split)

        # Get all image files in the split directory
        image_files = [f for f in os.listdir(full_path) if f.endswith(('.jpg'))]

        # Process images in batches
        for i in tqdm(range(0, len(image_files), batch_size), desc=f"Processing split {split}"):
            batch_files = image_files[i:i + batch_size]
            batch_images = []
            batch_ids = []

            # Extract images and IDs
            for img_file in batch_files:
                img_path = os.path.join(full_path, img_file)
                image = Image.open(img_path)
                batch_images.append(image.resize((image_size, image_size)))
                batch_ids.append(img_file)

            # Process batch with CLIP
            inputs = processor(images=batch_images, return_tensors="pt", do_resize=False, do_center_crop=False).to(
                device)
            with torch.no_grad():
                image_features = model(**inputs).pooler_output

            # Store encodings in dataset dictionary
            for idx, img_id in enumerate(batch_ids):
                sample = {'img_id': img_id, 'encoding': image_features[idx].cpu().numpy()}
                enc_split.append(sample)

        # Add split to full dataset
        full_enc_dataset[split] = Dataset.from_list(enc_split)

    return full_enc_dataset


def inference_loop_test_split():
    # Download the enc_descr dataset to determine the ids of the images to process in split 00
    enc_descr_00 = load_dataset("gips-mai/enc_descr", split="00")
    image_ids = enc_descr_00['img_id']

    model, processor, image_size, device = init_model()

    enc_images_split_00 = []

    for img_id in tqdm(image_ids, desc="Processing split 00"):
        img_path = os.path.join(RAW_OSV5M_DIR, '00', img_id)
        image = Image.open(img_path)
        image = image.resize((image_size, image_size))

        inputs = processor(images=image, return_tensors="pt", do_resize=False, do_center_crop=False).to(device)
        with torch.no_grad():
            image_features = model(**inputs).pooler_output

        enc_images_split_00.append({'img_id': img_id, 'encoding': image_features.squeeze(dim=0).cpu().numpy()})
        #print(f"Processed image {img_id}")
        #print(f"Encoding: {image_features.squeeze(dim=0).cpu().numpy().shape}")

    enc_images_split_00 = Dataset.from_list(enc_images_split_00)

    # Upload as new split to the enc_img dataset
    enc_img = load_dataset("gips-mai/enc_img")
    enc_img['00'] = enc_images_split_00
    enc_img.save_to_disk("gips-mai/enc_img")


if __name__ == "__main__":
    # test_clip()
    # full_enc_dataset = inference_loop()
    # print(full_enc_dataset)

    # Save dataset locally
    # full_enc_dataset.save_to_disk("data/enc_img")

    # Upload dataset to Hugging Face
    # full_enc_dataset.push_to_hub('gips-mai/enc_img', token=HF_AUTH_TOKEN)

    inference_loop_test_split()
    dataset = datasets.load_from_disk("gips-mai/enc_img")
    dataset.push_to_hub('gips-mai/enc_img', token=HF_AUTH_TOKEN)
