# -*- coding: utf-8 -*-
"""multimodal_inference.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Tk05LUo1tIgG9fD80z1RTjSqABZQu49I
"""

!pip install git+https://github.com/openai/CLIP.git
!pip install transformers==4.45.2
!pip install -Uq accelerate peft bitsandbytes

# from google.colab import drive
# drive.mount('/content/drive')

import torch
transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, BitsAndBytesConfig
# from audio_pipeline import AudioTranscriptionPipeline
import clip
from PIL import Image
import os
import json
import random
from multimodal_inference import MultimodalInference
import subprocess
import os

def download_files():
    # URLs to download
    coco_url = "http://images.cocodataset.org/zips/train2017.zip"
    coco_zip = "train2017.zip"
    coco_dir = "train2017"
    llava_url = "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json"
    llava_json = "llava_instruct_150k.json"

    # Download COCO dataset
    if not os.path.exists(coco_zip):
        print("Downloading COCO train2017 dataset...")
        subprocess.run(["wget", "-c", coco_url])
    else:
        print("COCO dataset already downloaded.")

    # Unzip COCO dataset
    if not os.path.exists(coco_dir):
        print("Unzipping the dataset...")
        subprocess.run(["unzip", "-q", coco_zip])
    else:
        print("COCO dataset already unzipped.")

    # Download LLaVA 150k instruction JSON file
    if not os.path.exists(llava_json):
        print("Downloading LLaVA 150k instruction JSON file...")
        subprocess.run(["wget", "-c", llava_url])
    else:
        print("LLaVA 150k instruction JSON file already downloaded.")

    print("All files have been downloaded and extracted.")

# Call the function to download files
download_files()

import json
import random

def prepare_dataset(conversations):
    # Assuming conversations is a list of dictionaries with 'from' and 'value' keys
    # Process the conversations to create a single text input
    processed_text = ""
    expected_gt = ""
    for conv in conversations:
        if conv['from'] == 'human':
            processed_text += f"Human: {conv['value']}\n"
            break
    for conv in conversations:
        if conv['from'] == 'gpt':
            expected_gt += f"Assistant: {conv['value']}\n"
        # elif conv['from'] == 'gpt':
        #     processed_text += f"Assistant: {conv['value']}\n"
    return processed_text.strip(), expected_gt.strip()

def select_random_sample(json_file, coco_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Select a random sample
    sample = random.choice(data)
    image_name = sample['image']
    conversations = sample['conversations']

    # Process the text using the prepare_dataset function
    text, expected_op = prepare_dataset(conversations)

    # Construct the image path
    image_path = os.path.join(coco_dir, image_name)

    return image_path, text, expected_op

# Example usage
image_path, text, expected_op = select_random_sample("llava_instruct_150k.json", "train2017")
print(f"Selected image: {image_path}")
print(f"Corresponding text: {text}")
print(f"Expected output: {expected_op}")

## lets see the image
img = Image.open(image_path)
display(img)

"""## Load model and generate"""

import gc
gc.collect()
torch.cuda.empty_cache()

# Initialize the inference class with pre-merged model
hf_inference = MultimodalInference(
    model_name= "sayanbanerjee32/multimodal-phi3_5-mini-instruct-llava",
    # 'sayanbanerjee32/multimodal-phi3-4k-instruct-llava',
    tokenizer_name="sayanbanerjee32/multimodal-phi3_5-mini-instruct-llava",
    debug = True
)

# Perform inference with pre-merged model
generated_text_merged = hf_inference.multimodal_inference(
    text_input=text,
    image_path=image_path
)

print("Generated text (pre-merged model):")
print(generated_text_merged)

# Initialize a new inference class with PEFT adapter
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_4bit = True
use_nested_quant = False
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)
peft_inference = MultimodalInference(
    model_name='microsoft/Phi-3.5-mini-instruct',
    tokenizer_name='microsoft/Phi-3.5-mini-instruct',
    peft_model_path="/content/drive/MyDrive/multimodal_llm/phi-3_5/multimodal-phi3_5-mini-instruct-llava_adapter/checkpoint-12",  # Adjust this path as needed
    bnb_config=bnb_config,  # Pass the bnb_config to the inference class
    debug=False  # Enable debug mode to see more information
)

# Perform inference with PEFT adapter
generated_text_peft = peft_inference.multimodal_inference(
    text_input=text,
    image_path=image_path
)

print("\nGenerated text (PEFT adapter):")
print(generated_text_peft)

# Compare the outputs
print("\nComparison:")
print("Pre-merged model output length:", len(generated_text_merged))
print("PEFT adapter model output length:", len(generated_text_peft))

# You can add more detailed comparison metrics here if needed

