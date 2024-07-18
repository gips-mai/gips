from transformers import AutoTokenizer, AutoModel
import torch
import os
import pandas as pd
import tqdm
from utils import load_image
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
RAW_OSV5M_DIR = os.getenv('RAW_OSV5M_DIR')
RAW_DESCR_DIR = os.getenv('RESULT_LOCATION_DIR')

# Set up command-line argument parsing
# Used to specify the directory of the images to be annotated (split 00 to 04)
parser = argparse.ArgumentParser(description='Annotate OSV5M dataset')
parser.add_argument("--img_fold_num", type=str, required=True, help='Directory number of the images to be annotated')
args = parser.parse_args()

# Define hyperparameters and configuration
img_fold_num = args.img_fold_num
folder_path = RAW_OSV5M_DIR + img_fold_num
result_folder = RAW_DESCR_DIR

parameter_size = torch.bfloat16
questions = "Given the image, analyze and provide the following details in a continuous text format if the details are not visible in the picture, please leave them out of the description:  1.General description: Provide a general description of the image.  2. Driving side: Determine if vehicles are driving on the left or right side of the road. Note the position of any visible traffic signs. 3. Road line color: Identify the color of the center and edge lines on the road. 4. Sign color: Describe the color and shape of any visible traffic signs. 5. Road condition: Describe the condition of the road surface. Note any signs of wear, damage, or maintenance. 6. Biome: Describe the biome visible in the image (e.g., desert, forest, grassland, urban). 7. Languages: Look at the picture, are there letters, and if so, which language do they come from? 7a. If you see letters, which alphabet do they come from and can you say what they say? Return the results in the following text format: \"The image shows a scene [general description of the image]. Vehicles are driving on the [left/right] side of the road. The road has [yellow/white/green, solid/dashed/no lines] center lines and [yellow/white/green, solid/dashed/no lines] edge lines. Visible traffic signs are [color and shape, distinctive features]. License plates are [present/absent], characterized by their [color], [shape], and positioned at the [front/back] of vehicles. The road surface appears [describe condition], indicating [describe maintenance or wear]. The biome visible in the image is [describe biome]. There  are letters from [language]. The letters are from [Latin, Greek, Cyrillic, Chinese, Japanese, Korean, Devanagari, Arabic, Hebrew, Bengali] and it says [].\""
batch_size = 16
generation_config = dict(
    num_beams=1,
    max_new_tokens=256,
    do_sample=False,
)

# Load model and tokenizer
path = "OpenGVLab/Mini-InternVL-Chat-4B-V1-5"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=parameter_size,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

# Initialize variables for batch processing
counter = 0
responses_to_save = []
num_files = 0

torch.cuda.empty_cache()
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg'))]
num_images = len(image_files)

# Create descriptions for each image in the folder in batches for time efficiency
for i in tqdm.tqdm(range(0, num_images, batch_size), desc="File: " + str(img_fold_num), unit="batch"):
    # Get the current batch of image files
    current_batch_files = image_files[i:i + batch_size]

    # Load images in batch
    batch_pixel_values = []
    for image_file in current_batch_files:
        image_path = os.path.join(folder_path, image_file)
        pixel_values = load_image(image_path, max_num=6).to(parameter_size).cuda()
        batch_pixel_values.append(pixel_values)

    # Prepare batch
    image_counts = [x.size(0) for x in batch_pixel_values]
    pixel_values = torch.cat(batch_pixel_values, dim=0)

    # Generate descriptions from the model
    responses = model.batch_chat(tokenizer, pixel_values,
                                 image_counts=image_counts,
                                 questions=questions,
                                 generation_config=generation_config)

    responses_to_save.append(
        {'image_file': img_file, 'response': resp} for img_file, resp in zip(current_batch_files, responses))
    counter += 1

    # Store responses every 6 batches to avoid recomputing the entire directory if the process is interrupted
    if counter >= 6:
        # Convert results to DataFrame
        df = pd.DataFrame(responses_to_save)
        # Save the results to a CSV file
        df.to_csv(f'{result_folder}/{img_fold_num}_image_descriptions' + str(num_files) + '.csv', index=False)
        print(f"Saved {num_files} files to {result_folder}")
        # Reset DataFrame and counter
        responses_to_save = []
        counter = 0
        num_files += 1

# Save any remaining samples
if responses_to_save:
    df = pd.DataFrame(responses_to_save)
    df.to_csv(f'{result_folder}/{img_fold_num}_image_descriptions' + str(num_files) + '.csv', index=False)
    print(f"Saved final file (number {num_files}) to {result_folder}")
