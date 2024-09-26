from datasets import load_dataset
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import os
import subprocess

# List of URLs to download
urls = [
    # "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/complex_reasoning_77k.json",
    # "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/conversation_58k.json",
    # "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/detail_23k.json",
    "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json",
    # "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_80k.json",
    # "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json"
]

# Download each file
for url in urls:
    subprocess.run(["wget", "-c", url])

print("All files have been downloaded.")
# Download and unzip the COCO train2017 dataset
coco_url = "http://images.cocodataset.org/zips/train2017.zip"
coco_zip = "train2017.zip"
coco_dir = "train2017"

print("Downloading COCO train2017 dataset...")
subprocess.run(["wget", "-c", coco_url])

print("Unzipping the dataset...")
subprocess.run(["unzip", "-q", coco_zip])

print(f"COCO train2017 dataset has been downloaded and extracted to {coco_dir}/")

import json
import shutil

# Create a directory to store the selected images
selected_images_dir = "selected_coco_images"
os.makedirs(selected_images_dir, exist_ok=True)

# Function to extract image names from a JSON file
def extract_image_names(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return set(item['image'] for item in data if 'image' in item)

# Collect image names from all JSON files
all_image_names = set()
for url in urls:
    json_file = url.split('/')[-1]
    if os.path.exists(json_file):
        all_image_names.update(extract_image_names(json_file))

print(f"Total unique images to process: {len(all_image_names)}")

# Copy the selected images from coco_dir to selected_images_dir
for image_name in tqdm(all_image_names, desc="Copying images"):
    src_path = os.path.join(coco_dir, image_name)
    dst_path = os.path.join(selected_images_dir, image_name)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
    else:
        print(f"Warning: Image {image_name} not found in {coco_dir}")

print(f"Selected images have been copied to {selected_images_dir}/")

import torch
from PIL import Image
# from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
import clip
import os
import numpy as np

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Define the image preprocessing pipeline
# preprocess = Compose([
#     Resize(224, interpolation=Image.BICUBIC),
#     CenterCrop(224),
#     ToTensor(),
#     Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
# ])

# Function to process an image and get its embedding
def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features.cpu().numpy()

# Process images and save embeddings
embeddings = {}
for image_name in tqdm(os.listdir(selected_images_dir), desc="Processing images"):
    image_path = os.path.join(selected_images_dir, image_name)
    embedding = get_image_embedding(image_path)
    embeddings[image_name] = embedding.squeeze()

# Save embeddings
embeddings_file = "coco_image_embeddings.npz"
np.savez_compressed(embeddings_file, **embeddings)

print(f"Image embeddings have been processed and saved to {embeddings_file}")

# Save embeddings to Google Drive
from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Define the directory path in Google Drive
save_dir = '/content/drive/MyDrive/multimodel_llm/image_embedding'

# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Define the full path for saving the embeddings file
embeddings_file_path = os.path.join(save_dir, embeddings_file)

# Save the embeddings to Google Drive
np.savez_compressed(embeddings_file_path, **embeddings)

print(f"Image embeddings have been saved to Google Drive: {embeddings_file_path}")


import numpy as np
import requests
from tqdm import tqdm
import os

# Function to download the file
def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            size = file.write(data)
            progress_bar.update(size)
    progress_bar.close()

# URL of the embeddings file (replace with your actual URL)
embeddings_url = '/content/drive/MyDrive/multimodel_llm/image_embedding/coco_image_embeddings.npz'

# Local filename to save the downloaded file
local_filename = "coco_image_embeddings.npz"

# Download the file if it doesn't exist locally
if not os.path.exists(embeddings_file):
    print(f"Downloading {embeddings_file}...")
    download_file(embeddings_url, embeddings_file)
else:
    print(f"{embeddings_file} already exists. Skipping download.")

# Load the embeddings
print("Loading embeddings...")
embeddings = np.load(local_filename, allow_pickle=True)

# Print embeddings and image names
for image_name, embedding in embeddings.items():
    print(f"Image: {image_name}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding preview: {embedding[:5]}...")  # Print first 5 values
    print("-" * 50)
    break

print(f"Total number of embeddings: {len(embeddings)}")




