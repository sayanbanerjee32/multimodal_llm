# -*- coding: utf-8 -*-
"""phi-3_QLoRA_instruct150k.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1sIyQFl-RN3d3jPVutndUA2EhUaHgDza6
"""
!pip install transformers==4.45.2
!pip install -Uq accelerate peft bitsandbytes trl dataset 
# !pip install -Uq flash_attn

import numpy as np
import requests
from tqdm import tqdm
import os, gc
import subprocess
import json
import random
### Download Phi-3 model
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig, PreTrainedModel
from transformers.trainer_callback import TrainerCallback
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
import joblib
from huggingface_hub import hf_hub_download

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

import dataclasses
from huggingface_hub import HfApi

from multimodal_inference import load_model_with_lora_and_projector


# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

"""### Download Phi-3 model"""

# Load the Phi-3 model and tokenizer
model_name = "microsoft/Phi-3.5-mini-instruct"
# "microsoft/Phi-3-mini-4k-instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right",
#                                            trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, add_eos_token=True, use_fast=True)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'left'

"""### Downlaod image embedding"""

from google.colab import drive
# Mount Google Drive
drive.mount('/content/drive')

# URL of the embeddings file (replace with your actual URL)
embeddings_url = '/content/drive/MyDrive/multimodal_llm/image_embedding/coco_image_embeddings.npz'

# Load the embeddings
print("Loading embeddings...")
embeddings = np.load(embeddings_url, allow_pickle=True)

# Print embeddings and image names
for image_name, embedding in embeddings.items():
    print(f"Image: {image_name}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding preview: {embedding[:5]}...")  # Print first 5 values
    print("-" * 50)
    break

print(f"Total number of embeddings: {len(embeddings)}")

"""### Data processing"""

# List of URLs to download
url = "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json"

# Download each file
subprocess.run(["wget", "-c", url])

# Load the downloaded JSON file
json_file = "llava_instruct_150k.json"
with open(json_file, 'r') as f:
    data = json.load(f)

# Function to convert conversation format
# def convert_conversation(conversation):
#     system_message = "<|system|>\nYou are a helpful assistant.<|end|>\n"
#     user_message = ""
#     assistant_message = ""

#     for item in conversation:
#         if item['from'] == 'human':
#             user_message = f"<|user|>\n{item['value']}<|end|>\n"
#         elif item['from'] == 'gpt':
#             assistant_message = f"<|assistant|>\n{item['value']}<|end|>\n"

#     return system_message + user_message + assistant_message

# Process and tokenize the data
# Process and tokenize the data


# tokenized_data = []
# for item in tqdm(data, desc="Tokenizing data"):
#     image_file = item['image']
#     if image_file in embeddings:
#         image_embedding = torch.tensor(embeddings[image_file], dtype=torch.float32, device=device)

#         # Use the existing conversation format
#         conversation = [
#             {"role": "system", "content": "You are a helpful assistant."}
#         ] + [{"role": "user" if msg['from'] == 'human' else "assistant", "content": msg['value']}
#              for msg in item['conversations']]

#         # Apply chat template and tokenize directly
#         tokenized_conversation = tokenizer.apply_chat_template(conversation, return_tensors='pt').to(device)

#         tokenized_item = {
#             'image': image_file,
#             'image_embedding': image_embedding,
#             'tokenized_conversation': tokenized_conversation
#         }
#         tokenized_data.append(tokenized_item)

# print(f"Total tokenized items: {len(tokenized_data)}")
# print(f"Sample tokenized item:")
# print(f"Image: {tokenized_data[0]['image']}")
# print(f"Image embedding shape: {tokenized_data[0]['image_embedding'].shape}")
# print(f"Tokenized conversation shape: {tokenized_data[0]['tokenized_conversation'].shape}")
# print(f"Image embedding device: {tokenized_data[0]['image_embedding'].device}")
# print(f"Tokenized conversation device: {tokenized_data[0]['tokenized_conversation'].device}")

data = data[0:10000]

def create_dataset():
    processed_data = []
    print("Processing data...")
    with tqdm(total=len(data)) as pbar:
        for item in data:
            image_file = item['image']
            if image_file in embeddings:
                processed_data.append({
                    'image': image_file,
                    'image_embedding': embeddings[image_file].tolist(),
                    'conversation': item['conversations']
                })
            pbar.update(1)

    print(f"Data processing completed. Total processed items: {len(processed_data)}")

    return Dataset.from_dict({
        "image": [item['image'] for item in processed_data],
        "image_embedding": [item['image_embedding'] for item in processed_data],
        "conversation": [item['conversation'] for item in processed_data]
    })

print("Creating HuggingFace dataset...")
hf_dataset = create_dataset()

print("HuggingFace dataset creation completed.")
print(f"Total samples in dataset: {len(hf_dataset)}")

# print("Applying tokenization and preparing the dataset...")

# def prepare_dataset(examples):
#     image_embeddings = torch.stack([torch.tensor(item) for item in examples['image_embedding']])

#     conversations = []
#     for conv in examples['conversation']:
#         dialogue = [{"role": "system", "content": "You are a helpful assistant."}]

#         for i, message in enumerate(conv):
#             if message['from'] == 'human':
#                 content = message['value'].replace('<image>', '').strip()  # Remove '<image>' and strip whitespace
#                 if i == 0:
#                     content = f"Given the following information, provide a detailed and accurate response:\n{content}\n[An image is provided for this task.]\n"
#                 dialogue.append({"role": "user", "content": content})
#             elif message['from'] == 'gpt':
#                 dialogue.append({"role": "assistant", "content": message['value']})

#         conversations.append(dialogue)

#     tokenized_conversations = tokenizer.apply_chat_template(conversations,
#                                                             return_tensors='pt', padding=True)

#     return {
#         "image_embeddings": image_embeddings,
#         "input_ids": tokenized_conversations,
#         "attention_mask": torch.ones_like(tokenized_conversations),
#         "labels": tokenized_conversations.clone()
#     }

def prepare_dataset(examples):
    image_embeddings = []
    conversations = []

    for idx, conv in enumerate(examples['conversation']):
        image_embedding = torch.tensor(examples['image_embedding'][idx])
        dialogue_pairs = []

        for i in range(0, len(conv), 2):
            if i + 1 < len(conv):  # Ensure we have a pair
                human_msg = conv[i]['value'].replace('<image>', '').strip()
                gpt_msg = conv[i + 1]['value']

                dialogue = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Given the following information, provide a detailed and accurate response:\n{human_msg}\n[An image is provided for this task.]\n"},
                    {"role": "assistant", "content": gpt_msg}
                ]

                dialogue_pairs.append(dialogue)
                image_embeddings.append(image_embedding)

        conversations.extend(dialogue_pairs)

    image_embeddings = torch.stack(image_embeddings)

    tokenized_conversations = tokenizer.apply_chat_template(conversations,
                                                            return_tensors='pt', padding=True)

    return {
        "image_embeddings": image_embeddings,
        "input_ids": tokenized_conversations,
        "attention_mask": torch.ones_like(tokenized_conversations),
        "labels": tokenized_conversations.clone()
    }

# Test the prepare_dataset function with a real training example
def test_prepare_dataset():
    # Get a batch of examples from the dataset
    batch_size = 1  # You can adjust this as needed
    sample_batch = hf_dataset[5:5+batch_size]

    print("Original conversations:")
    # for i, sample in enumerate(sample_batch):
    #     print(f"\nSample {i + 1}:")
    for message in sample_batch['conversation'][0]:
        print(f"{message['from']}: {message['value']}")

    # Process the sample batch
    result = prepare_dataset(sample_batch)

    # Print the structure of the result
    print("\nResult keys:", result.keys())
    print("Image embeddings shape:", result['image_embeddings'].shape)
    print("Input IDs shape:", result['input_ids'].shape)
    print("Attention mask shape:", result['attention_mask'].shape)
    print("Labels shape:", result['labels'].shape)

    for i in range(batch_size):
        decoded_input = tokenizer.decode(result['input_ids'][i])
        decoded_labels = tokenizer.decode(result['labels'][i])

        print(f"\nRestructured input for sample {i + 1}:")
        print(decoded_input)

        print(f"\nLabels for sample {i + 1}:")
        print(decoded_labels)

        # Optionally, you can print a more readable version of the labels
        print("\nReadable labels (non-padding tokens):")
        readable_labels = tokenizer.decode([token for token in result['labels'][i] if token != -100])
        print(readable_labels)

    # Optionally, you can print attention mask to see where it's applied
    print("\nAttention Mask:")
    print(result['attention_mask'][0])
# Run the test
test_prepare_dataset()

# Apply tokenization and prepare the dataset
print("Applying tokenization and preparing the dataset...")


# def prepare_dataset(examples):
#     image_embeddings = torch.stack([torch.tensor(item) for item in examples['image_embedding']])

#     conversations = [
#         [{"role": "system", "content": "You are a helpful assistant."}] +
#         [{"role": "user" if msg['from'] == 'human' else "assistant", "content": msg['value']}
#          for msg in conv]
#         for conv in examples['conversation']
#     ]

#     tokenized_conversations = tokenizer.apply_chat_template(conversations,
#                                                              return_tensors='pt', padding=True)

#     return {
#         "image_embeddings": image_embeddings,
#         "input_ids": tokenized_conversations,
#         "attention_mask": torch.ones_like(tokenized_conversations),
#         "labels": tokenized_conversations.clone()
#     }


hf_dataset_mapped = hf_dataset.map(
    prepare_dataset,
    batched=True,
    remove_columns=hf_dataset.column_names,
    batch_size=1024  # Adjust based on your memory constraints
).with_format("torch")

# Split the dataset
train_test_split = hf_dataset_mapped.train_test_split(test_size=0.05)

# Create a DatasetDict
dataset_dict = DatasetDict({
    'train': train_test_split['train'],
    'test': train_test_split['test']
})

print(f"Train dataset size: {len(dataset_dict['train'])}")
print(f"Test dataset size: {len(dataset_dict['test'])}")

# Example of accessing an item:
sample = dataset_dict['train'][0]
print(f"Input IDs shape: {len(sample['input_ids'])}")
print(f"Attention mask shape: {len(sample['attention_mask'])}")
print(f"Labels shape: {len(sample['labels'])}")

"""### Projection Layer"""
from phi3_with_projector import ImageProjector, Phi3WithProjector

### Projection Layer
# class SimpleResBlock(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.pre_norm = nn.LayerNorm(input_dim)
#         self.proj = nn.Sequential(
#             nn.Linear(input_dim, output_dim),
#             nn.GELU(),
#             nn.Linear(output_dim, output_dim)
#         )

#     def forward(self, x):
#         x = self.pre_norm(x)
#         return x + self.proj(x)

# class ImageProjector(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.proj = nn.Linear(input_dim, output_dim)

#     def forward(self, x):
#         return self.proj(x)

# import torch
# from torch.utils.data import Dataset, random_split

# class Phi3Dataset(Dataset):
#     def __init__(self, tokenized_data, projector, tokenizer):
#         self.data = tokenized_data
#         self.projector = projector
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         image_embedding = item['image_embedding']
#         conversation = item['tokenized_conversation']

#         projected_image = self.projector(image_embedding.unsqueeze(0)).squeeze(0)

#         # Combine projected image and conversation
#         combined_input = torch.cat([projected_image, conversation.squeeze(0)])

#         # Create attention mask
#         attention_mask = torch.ones_like(combined_input)

#         # Prepare labels (shift right, set first token to -100)
#         labels = torch.cat([-100 * torch.ones(projected_image.shape[0], dtype=torch.long), conversation.squeeze(0)[:-1]])

#         return {
#             "input_ids": combined_input,
#             "attention_mask": attention_mask,
#             "labels": labels
#         }

# # Usage:
# image_embedding_dim = tokenized_data[0]['image_embedding'].shape[-1]
# projection_dim = 1024  # Adjust as needed
# projector = SimpleResBlock(image_embedding_dim, projection_dim)
# full_dataset = Phi3Dataset(tokenized_data, projector, tokenizer)

# # Split the dataset
# train_size = int(0.9 * len(full_dataset))
# test_size = len(full_dataset) - train_size
# train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# print(f"Train dataset size: {len(train_dataset)}")
# print(f"Test dataset size: {len(test_dataset)}")

# # Example of accessing an item:
# sample = train_dataset[0]
# print(f"Input IDs shape: {sample['input_ids'].shape}")
# print(f"Attention mask shape: {sample['attention_mask'].shape}")
# print(f"Labels shape: {sample['labels'].shape}")

"""### QLoRA set up"""

# new_model = "ms-phi3-custom"
lora_r = 16 #32 #64
lora_alpha = 16
lora_dropout = 0.05
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
output_dir = "./multimodal-phi3_5-mini-instruct-llava_adapter"
num_train_epochs = 1
fp16 = False
bf16 = False
per_device_train_batch_size = 8
per_device_eval_batch_size = 4
gradient_accumulation_steps = 4
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 5e-4
weight_decay = 0.001
optim = "adamw_torch" #"paged_adamw_32bit"
lr_scheduler_type = "linear" #"constant"
max_steps = 52 #-1
warmup_ratio = 0.1 #0.03
group_by_length = True
save_steps = 25
logging_steps = 25
eval_steps = 50 # Evaluate every 25 steps
max_seq_length = 256
packing = False
device_map = {"": 0}
hf_adapter_repo="sayanbanerjee32/multimodal-phi3_5-mini-instruct-llava_adapter"

# Define the path in Google Drive where you want to save the checkpoints
gdrive_checkpoint_dir = "/content/drive/MyDrive/multimodal_llm/phi-3_5/checkpoints"

# Ensure the directory exists
os.makedirs(gdrive_checkpoint_dir, exist_ok=True)


class SaveLatestCheckpointAndLoraCallback(TrainerCallback):
    # def __init__(self, tokenizer):
    #     super().__init__()
    #     self.tokenizer = tokenizer

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            checkpoint_dir = os.path.join(gdrive_checkpoint_dir, f"checkpoint-{state.global_step}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Save the model
            kwargs["model"].save_pretrained(checkpoint_dir)

            # Save the tokenizer
            kwargs["tokenizer"].save_pretrained(checkpoint_dir)
            # self.tokenizer.save_pretrained(checkpoint_dir)

            # Log and save the projector
            projector = kwargs["model"].base_model.model.projector
            weight_change = projector.get_weight_change()
            print(f"Projector weight change at step {state.global_step}: {weight_change}")

            # Add gradient checking here
            for name, param in projector.named_parameters():
                if param.grad is None:
                    print(f"No gradient for {name}")
                else:
                    print(f"Gradient norm for {name}: {param.grad.norm().item()}")

            projector_path = os.path.join(checkpoint_dir, "image_projector.pth")
            torch.save(projector.state_dict(), projector_path)

            # Save LoRA weights
            lora_state_dict = {}
            for name, module in kwargs["model"].base_model.model.phi3.named_modules():
                if isinstance(module, lora.Linear4bit):
                    if hasattr(module, 'lora_A'):
                        lora_state_dict[f"{name}.lora_A.weight"] = module.lora_A.default.weight.data.cpu()
                        lora_state_dict[f"{name}.lora_B.weight"] = module.lora_B.default.weight.data.cpu()
                        lora_state_dict[f"{name}.scaling"] = module.scaling
                    
                    # Save lora_embedding_A and lora_embedding_B if they exist and are not empty
                    if hasattr(module, 'lora_embedding_A') and module.lora_embedding_A:
                        lora_state_dict[f"{name}.lora_embedding_A"] = {k: v.cpu() for k, v in module.lora_embedding_A.items()}
                    if hasattr(module, 'lora_embedding_B') and module.lora_embedding_B:
                        lora_state_dict[f"{name}.lora_embedding_B"] = {k: v.cpu() for k, v in module.lora_embedding_B.items()}

            torch.save(lora_state_dict, os.path.join(checkpoint_dir, "lora_weights.pt"))

            # Explicitly save the trainer state
            trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.json")
            state_dict = dataclasses.asdict(state)
            with open(trainer_state_path, "w") as f:
                json.dump(state_dict, f, indent=2)

            # Remove previous checkpoint
            prev_checkpoint = os.path.join(gdrive_checkpoint_dir, f"checkpoint-{state.global_step - args.save_steps}")
            if os.path.exists(prev_checkpoint):
                import shutil
                shutil.rmtree(prev_checkpoint)

            print(f"Saved checkpoint to {checkpoint_dir}")

    #         # Upload the checkpoint to Hugging Face Hub
    #         self.upload_to_hub(checkpoint_dir, args.hub_model_id)

    # def upload_to_hub(self, checkpoint_dir, hub_model_id):
    #     api = HfApi()
    #     api.upload_folder(
    #         folder_path=checkpoint_dir,
    #         repo_id=hub_model_id,
    #         repo_type="model",
    #     )
    #     print(f"Uploaded checkpoint {checkpoint_dir} to Hugging Face Hub")

# Function to get the latest checkpoint
def get_latest_checkpoint(checkpoint_dir):
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[1]))
    return os.path.join(checkpoint_dir, latest_checkpoint)

if torch.cuda.is_bf16_supported():
  compute_dtype = torch.bfloat16
#   attn_implementation = 'flash_attention_2'
else:
  compute_dtype = torch.float16
#   attn_implementation = 'sdpa'

# print(attn_implementation)
print(compute_dtype)

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)
# # Load the model again for quantization
# ### Download Phi-3 model
# phi3_model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     trust_remote_code=True,
#     quantization_config=bnb_config,
#     device_map=device_map,
#     torch_dtype=compute_dtype,
#     # attn_implementation=attn_implementation
# )

# print(phi3_model)

# # Initialize the projector
# image_embedding_dim = len(hf_dataset[0]['image_embedding'])
# projection_dim = phi3_model.config.hidden_size  # Get dimension from the model
# projector = ImageProjector(image_embedding_dim, projection_dim).to(device)

# # Combine Phi-3 with the projector
# model = Phi3WithProjector(phi3_model, projector)

# Get the latest checkpoint
latest_checkpoint = get_latest_checkpoint(gdrive_checkpoint_dir)
eval_first = False
if latest_checkpoint:
    print(f"Loading model from checkpoint: {latest_checkpoint}")
    model = load_model_with_lora_and_projector(latest_checkpoint, device, bnb_config=bnb_config)
    eval_first = True
else:
    print("No checkpoint found. Starting training from scratch.")
    phi3_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=compute_dtype,
        attn_implementation='eager'
    )
    image_embedding_dim = len(hf_dataset[0]['image_embedding'])
    projection_dim = phi3_model.config.hidden_size
    projector = ImageProjector(image_embedding_dim, projection_dim).to(device)
    model = Phi3WithProjector(phi3_model, projector)

    # Prepare the model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Define LoRA configuration
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)

def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )


print_trainable_parameters(model)

print(model)

def verify_projector_weight_changes(loaded_state_dict, initial_state_dict):
    for key in loaded_state_dict:
        if torch.norm(loaded_state_dict[key] - initial_state_dict[key]).item() > 0:
            print(f"Weights changed for {key}")
            return True
    print("WARNING: No weight changes detected in the projector!")
    return False

# Use this when loading the model
import copy
initial_projector_state =  copy.deepcopy(model.base_model.model.projector.state_dict())

"""### Training"""

gc.collect()
torch.cuda.empty_cache()

# Define training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="all",
    eval_strategy="steps",
    eval_steps=eval_steps,
    do_eval=True,
    eval_delay=0,
    gradient_checkpointing=gradient_checkpointing,
    ddp_find_unused_parameters=False,
    save_total_limit=1,
    hub_model_id=hf_adapter_repo,
    push_to_hub=False,  # We'll handle this in our custom callback
)

# Custom data collator to handle pre-tokenized inputs
def custom_data_collator(features):
    batch = {k: [d[k] for d in features] for k in features[0].keys()}

    # Stack image embeddings
    batch['image_embeddings'] = torch.stack(batch['image_embeddings']).to(torch.float16)

    # Pad the sequences
    batch['input_ids'] = torch.nn.utils.rnn.pad_sequence(batch['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id)
    batch['attention_mask'] = torch.nn.utils.rnn.pad_sequence(batch['attention_mask'], batch_first=True, padding_value=0)
    batch['labels'] = torch.nn.utils.rnn.pad_sequence(batch['labels'], batch_first=True, padding_value=-100)

    return batch

# Function to select a random subset of the dataset
def select_subset(dataset, fraction=0.05):
    num_samples = int(len(dataset) * fraction)
    indices = random.sample(range(len(dataset)), num_samples)
    return dataset.select(indices)

# Select 5% of the training and test datasets
small_train_dataset = select_subset(dataset_dict['train'], fraction=0.1)
small_test_dataset = select_subset(dataset_dict['test'], fraction=0.1)

# Create a new DatasetDict with the smaller datasets
small_dataset_dict = DatasetDict({
    'train': small_train_dataset,
    'test': small_test_dataset
})

print(f"Small train dataset size: {len(small_dataset_dict['train'])}")
print(f"Small test dataset size: {len(small_dataset_dict['test'])}")

# Before initializing the SFTTrainer
model.projector.train()  # Set projector to training mode
for param in model.projector.parameters():
    param.requires_grad = True

# After model definition and before SFTTrainer initialization

def hook_fn(module, grad_input, grad_output):
    print(f"Gradient for {module.__class__.__name__}:")
    for idx, grad in enumerate(grad_input):
        if grad is not None:
            print(f"  Input gradient {idx}: {grad.norm().item()}")

# # Apply the hook to projector layers
# model.projector.layer1.register_backward_hook(hook_fn)
# model.projector.layer2.register_backward_hook(hook_fn)

# Initialize the SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=small_dataset_dict['train'],
    eval_dataset=small_dataset_dict['test'],
    tokenizer=tokenizer,
    data_collator=custom_data_collator,
    peft_config=lora_config,
    max_seq_length=max_seq_length,
    packing=packing,
    callbacks=[SaveLatestCheckpointAndLoraCallback()],
)

# Perform initial evaluation
if eval_first:
    print("Performing initial evaluation...")
    eval_results = trainer.evaluate()
    print(f"Initial evaluation results: {eval_results}")
    # Start or resume training
    trainer.train(resume_from_checkpoint=latest_checkpoint)
else:
    # Start training
    trainer.train()

# # Save the fine-tuned model
# trainer.save_model()
# # trainer.model.save_pretrained(new_model)
# final_model_path = os.path.join(gdrive_checkpoint_dir, "final_model")
# trainer.model.save_pretrained(final_model_path)
# tokenizer.save_pretrained(final_model_path)

# torch.save(model.base_model.model.projector.state_dict(), final_model_path + '/image_projector.pth')
# print(f"Projector saved to: {final_model_path}/image_projector.pth")

# # Save the projector
# projector_path = '/content/drive/MyDrive/multimodal_llm/phi-3_5/image_projector.pth'
# os.makedirs(os.path.dirname(projector_path), exist_ok=True)
# torch.save(model.projector.state_dict(), projector_path)
# print(f"Projector saved to: {projector_path}")

# Get the latest checkpoint
latest_checkpoint = get_latest_checkpoint(gdrive_checkpoint_dir)

loaded_projector_state = torch.load(latest_checkpoint + '/image_projector.pth')
verify_projector_weight_changes(loaded_projector_state, initial_projector_state)

# trainer.push_to_hub()
api = HfApi()
api.upload_folder(
    folder_path=latest_checkpoint,
    repo_id=hf_adapter_repo,
    repo_type="model",
)
# !cp -r results_phi-3_5 /content/drive/MyDrive/multimodal_llm/phi-3_5

# Add this function after your CustomTextGenerator class
def evaluate_model(model, tokenizer, eval_dataset, device):
    model = model.to(torch.float16).to(device)
    model.eval()
    total_loss = 0
    num_batches = 0

    dataloader = DataLoader(eval_dataset, batch_size=2, collate_fn=custom_data_collator)

    with torch.no_grad():
        for batch in dataloader:
            batch = {
                'input_ids': batch['input_ids'].to(device).int(),
                'attention_mask': batch['attention_mask'].to(device).int(),
                'image_embeddings': batch['image_embeddings'].to(device).to(torch.float16),
                'labels': batch['labels'].to(device).int()
            }

            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                image_embeddings=batch['image_embeddings'],
                labels=batch['labels']
            )
            total_loss += outputs.loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0

# Add these lines before merging the model
print("Evaluating model before merging...")
pre_merge_loss = evaluate_model(model, tokenizer, small_dataset_dict['test'], device)
print(f"Pre-merge loss: {pre_merge_loss}")

"""## sample inference code"""

gc.collect()
torch.cuda.empty_cache()

# Create a custom text generation class
class CustomTextGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, input_text, image_embedding, **generate_kwargs):
        # Tokenize the input text
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        # Ensure image_embedding is a tensor and move it to the correct device
        if not isinstance(image_embedding, torch.Tensor):
            image_embedding = torch.tensor(image_embedding)
        # image_embedding = image_embedding.to(self.model.device).unsqueeze(0)  # Add batch dimension
        image_embedding = image_embedding.to(self.model.device)#.to(next(self.model.parameters()).dtype)

        # Generate text
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # image_embeddings=image_embedding,
            image_embeddings=image_embedding.unsqueeze(0),  # Add batch dimension
            **generate_kwargs
        )

        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

# Get a sample from the validation set
sample = dataset_dict['test'][1]
image_embedding = sample['image_embeddings']


def get_first_user_input(decoded_text):
    # Find the position of the first <|assistant|> tag
    assistant_pos = decoded_text.find('<|assistant|>')

    # If <|assistant|> is found, truncate the text
    if assistant_pos != -1:
        return decoded_text[:assistant_pos].strip()
    else:
        return decoded_text.strip()

# Initialize the custom text generator
generator = CustomTextGenerator(model=model, tokenizer=tokenizer)
# Decode the input_ids
full_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)

# Extract only the first user input
input_text = get_first_user_input(full_text) + '<|assistant|>'
print(input_text)

def test_inference(model, tokenizer, image_embedding):
    # Log projector weights before inference
    projector = model.projector
    weight_change = projector.get_weight_change()
    print(f"Projector weight change: {weight_change}")

    # Ensure the model is in the correct precision (e.g., float16)
    # model = model.to(next(model.parameters()).dtype)

    generator = CustomTextGenerator(model=model, tokenizer=tokenizer)
    input_text = """<|system|> You are a helpful assistant.<|end|><|user|> Given the following information, provide a detailed and accurate response:
Describe the image in detail.
[An image is provided for this task.]
<|end|><|assistant|>
"""
    generated_text = generator.generate(
        input_text,
        image_embedding=image_embedding,
        max_new_tokens=150,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
    )
    return generated_text

# Generate text
generated_text = generator.generate(
    input_text,
    image_embedding=image_embedding,
    # max_length=200,
    # num_return_sequences=1,
    # do_sample=True,
    # temperature=0.7,
    # top_k=50,
    # top_p=0.95,
    max_new_tokens=150,
    num_return_sequences=1,
    do_sample=True,
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
)

print("Input text:")
print(input_text)
print("\nGenerated text:")
print(generated_text)

# Before merging
print("Testing inference before merging...")
pre_merge_output = test_inference(model, tokenizer, sample['image_embeddings'])
print(pre_merge_output)

"""### merge models and save in gdrive"""

del trainer
del model
gc.collect()
torch.cuda.empty_cache()

# Merge the fine-tuned adapter with the base model
from peft import AutoPeftModelForCausalLM
from peft import PeftModel

# Load the fine-tuned model with the LoRA adapter
# Reload model in FP16 and merge it with LoRA weights
# base_model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     low_cpu_mem_usage=True,
#     return_dict=True,
#     torch_dtype=torch.float16,
#     device_map=device_map,
# )
# print(base_model)



loaded_model = load_model_with_lora_and_projector(hf_adapter_repo,
                                                  device,
                                                  bnb_config=bnb_config)

def evaluate_loaded_model(model, tokenizer, eval_dataset, device):
    model = model.to(device)
    model.eval()
    
    # Print model parameter dtypes
    for name, param in model.named_parameters():
        if 'projector' in name:
            print(f"{name} dtype: {param.dtype}")

    total_loss = 0
    num_batches = 0

    dataloader = DataLoader(eval_dataset, batch_size=2, collate_fn=custom_data_collator)

    with torch.no_grad():
        for batch in dataloader:
            batch = {
                'input_ids': batch['input_ids'].to(device).int(),
                'attention_mask': batch['attention_mask'].to(device).int(),
                'image_embeddings': batch['image_embeddings'], #.to(device).to(torch.float16),
                'labels': batch['labels'].to(device).int()
            }

            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                image_embeddings=batch['image_embeddings'],
                labels=batch['labels']
            )
            total_loss += outputs.loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0


print("Evaluating loaded model...")
loded_model_loss = evaluate_loaded_model(loaded_model, tokenizer, small_dataset_dict['test'], device)
print(f"loded model loss: {loded_model_loss}")

# after loading model from PEFT adapter
print("Testing inference after loading model from peft adapter...")
peft_adapter_output = test_inference(loaded_model, tokenizer, sample['image_embeddings'])
print(peft_adapter_output)

peft_model_id = hf_adapter_repo
# peft_model_id = "/content/drive/MyDrive/multimodal_llm/phi-3_5/checkpoints"
tr_model_id = model_name

merged_model = AutoModelForCausalLM.from_pretrained(tr_model_id,
                                             trust_remote_code=True,
                                             torch_dtype=torch.float16,
                                            attn_implementation='eager')
merged_model = PeftModel.from_pretrained(merged_model, peft_model_id)
merged_model = merged_model.merge_and_unload()
print(merged_model)

# gdrive_checkpoint_dir = "/content/drive/MyDrive/multimodal_llm/phi-3_5/checkpoints"
# final_model_path = os.path.join(gdrive_checkpoint_dir, "final_model")

# new_model = PeftModel.from_pretrained(base_model, final_model_path)
# print(new_model)

# print(new_model)

# for name, param in new_model.named_parameters():
#     if 'lora' in name:
#         print(f"{name}: {param.data.abs().mean()}")

# Merge the LoRA adapter with the base model
# merged_model = new_model.merge_and_unload()
# print(merged_model)

# Define the path to save the merged model in Google Drive
merged_model_path = '/content/drive/MyDrive/multimodal_llm/phi-3_5/merged_phi-3_5_llava_model'

# Save the merged model
# merged_model.save_pretrained(merged_model_path)

# Initialize the projector
image_embedding_dim = len(hf_dataset[0]['image_embedding'])
projection_dim = merged_model.config.hidden_size  # Get dimension from the model
projector = ImageProjector(image_embedding_dim, projection_dim).to(device)
projector.load_state_dict(torch.load(final_model_path + '/image_projector.pth'))
print(projector)

# Combine Phi-3 with the projector
phi3_with_projector = Phi3WithProjector(merged_model, projector)
print(phi3_with_projector)

# def compare_models(base_model, fine_tuned_model):
#     differences = []
#     for (name1, p1), (name2, p2) in zip(base_model.named_parameters(), fine_tuned_model.named_parameters()):
#         if name1 == name2:
#             diff = (p1 - p2).abs().max().item()  # Use max absolute difference
#             differences.append((name1, diff))
#     return differences
# # Compare the models
# differences = compare_models(base_model, new_model.model)

# # Sort differences by magnitude
# differences.sort(key=lambda x: x[1], reverse=True)
# print("Top 10 layers with the largest differences:")
# for name, diff in differences[:10]:
#     print(f"{name}: {diff}")

torch.cuda.empty_cache()
gc.collect()

# After merging the model
print("Evaluating merged model...")
post_merge_loss = evaluate_model(phi3_with_projector, tokenizer, small_dataset_dict['test'], device)
print(f"Post-merge loss: {post_merge_loss}")

if post_merge_loss > pre_merge_loss * 1.1:  # Allow for some variance
    print("Warning: Significant performance degradation after merging. Consider using the non-merged model.")

# After merging
print("Testing inference after merging...")
post_merge_output = test_inference(phi3_with_projector, tokenizer, sample['image_embeddings'])
print(post_merge_output)

torch.cuda.empty_cache()
gc.collect()

# Save the merged model with the projector
phi3_with_projector.save_pretrained(merged_model_path)

# Save the tokenizer
# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
tokenizer.save_pretrained(merged_model_path)

print(f"Merged model and tokenizer saved to: {merged_model_path}")

from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path=merged_model_path,
    repo_id="sayanbanerjee32/multimodal-phi3_5-mini-instruct-llava",
    repo_type="model",
    delete_patterns = "*.safetensors",
)
print("Model uploaded to Hugging Face Hub")

print('end')

