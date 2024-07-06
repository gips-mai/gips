import numpy as np
from datasets import Dataset
from tqdm import tqdm
import torch
from transformers import RobertaTokenizer, RobertaModel
import os
from dotenv import load_dotenv
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from sentence_transformers import SentenceTransformer

# Definitions
model = SentenceTransformer('all-mpnet-base-v2')

load_dotenv()
HF_AUTH_TOKEN = os.getenv('HF_AUTH_TOKEN')


def encode_dataset(df, unzip_fct, data_key, filter_keys=None, batch_size=128, writer_batch_size=20000):
    """ Encode a dataset using a given tokenizer and model. The dataset is expected to be a pandas DataFrame.
    Args:
        df: pd.DataFrame, the dataset to encode
        unzip_fct: function, the function to use to unzip a row, must return a dictionary
        data_key: str, the key of the data to encode in the sample
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
        inputs = [sample[data_key] for sample in batch_samples]

        # Encode the batch
        with torch.no_grad():
            batch_encoding = model.encode(inputs)

        # Process each sample in the batch
        for j, sample in enumerate(batch_samples):
            encoding_dict = sample.copy()  # Copy the sample
            if filter_keys is not None:
                for key in filter_keys:
                    encoding_dict.pop(key, None)
            encoding_dict['encoding'] = batch_encoding[j]
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


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # Calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # Split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values
