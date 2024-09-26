### Downlaod image embedding

import numpy as np
import math
from tqdm import tqdm
import os, gc
import subprocess
import json
### Download Phi-3 model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
import torch.nn as nn
from datasets import Dataset, DatasetDict
import joblib

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Load the Phi-3 model and tokenizer
model_name = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
# model = AutoModelForCausalLM.from_pretrained(model_name)

from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# URL of the embeddings file (replace with your actual URL)
embeddings_url = '/content/drive/MyDrive/multimodel_llm/image_embedding/coco_image_embeddings.npz'

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

### Data processing
# List of URLs to download
url = "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json"

# Download each file

subprocess.run(["wget", "-c", url])

import json

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
# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



#### creating the dataset
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

def prepare_dataset(examples):
    image_embeddings = torch.stack([torch.tensor(item) for item in examples['image_embedding']])
    
    conversations = [
        [{"role": "system", "content": "You are a helpful assistant."}] +
        [{"role": "user" if msg['from'] == 'human' else "assistant", "content": msg['value']} 
         for msg in conv]
        for conv in examples['conversation']
    ]
    
    tokenized_conversations = tokenizer.apply_chat_template(conversations,
                                                             return_tensors='pt', padding=True)
    
    return {
        "image_embeddings": image_embeddings,
        "input_ids": tokenized_conversations,
        "attention_mask": torch.ones_like(tokenized_conversations),
        "labels": tokenized_conversations.clone()
    }


print("Creating HuggingFace dataset...")
hf_dataset = create_dataset()

print("HuggingFace dataset creation completed.")
print(f"Total samples in dataset: {len(hf_dataset)}")

# Apply tokenization and prepare the dataset
print("Applying tokenization and preparing the dataset...")
hf_dataset = hf_dataset.map(
    prepare_dataset,
    batched=True,
    remove_columns=hf_dataset.column_names,
    batch_size= 1024  # Adjust based on your memory constraints
).with_format("torch")

# Split the dataset
train_test_split = hf_dataset.train_test_split(test_size=0.1)

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


### Projection Layer

# class ImageProjector(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.proj = nn.Linear(input_dim, output_dim)
    
#     def forward(self, x):
#         return self.proj(x)


#### Hyperparameter
new_model = "ms-phi3-custom"
lora_r = 16
lora_alpha = 16
lora_dropout = 0.05
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
output_dir = "./results"
num_train_epochs = 1
fp16 = False
bf16 = False
per_device_train_batch_size = 4
per_device_eval_batch_size = 1
gradient_accumulation_steps = 4
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 5e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "constant"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 25
logging_steps = 25
max_seq_length = 1024
packing = False
device_map = {"": 0}

from transformers import PreTrainedModel

class ImageProjector(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class Phi3WithProjector(PreTrainedModel):
    supports_gradient_checkpointing = True

    def __init__(self, phi3_model, projector):
        super().__init__(phi3_model.config)
        self.phi3 = phi3_model
        self.projector = projector

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Load the base Phi-3 model
        phi3_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        
        # Load the projector weights
        projector_path = os.path.join(pretrained_model_name_or_path, "image_projector.pth")
        if os.path.exists(projector_path):
            projector_state_dict = torch.load(projector_path, map_location=phi3_model.device)
            
            # Check if the state dict has the expected structure
            if 'linear.weight' in projector_state_dict:
                input_dim = projector_state_dict['linear.weight'].size(1)
                output_dim = projector_state_dict['linear.weight'].size(0)
            else:
                # If not, try to infer dimensions from the first layer's weight
                first_key = next(iter(projector_state_dict))
                input_dim = projector_state_dict[first_key].size(1)
                output_dim = phi3_model.config.hidden_size  # Assuming this is the correct output dimension
            
            projector = ImageProjector(input_dim, output_dim)
            
            # Try to load the state dict, ignoring mismatched keys
            projector.load_state_dict(projector_state_dict, strict=False)
            print(f"Loaded projector with input_dim={input_dim}, output_dim={output_dim}")
        else:
            print(f"Projector weights not found at {projector_path}. Initializing with default dimensions.")
            input_dim = 512  # Default CLIP embedding size
            output_dim = phi3_model.config.hidden_size
            projector = ImageProjector(input_dim, output_dim)
        
        # Create and return the Phi3WithProjector instance
        model = cls(phi3_model, projector)
        return model

    def save_pretrained(self, save_directory):
        # Save the base model
        self.phi3.save_pretrained(save_directory)
        
        # Save the projector weights
        projector_path = os.path.join(save_directory, "image_projector.pth")
        torch.save(self.projector.state_dict(), projector_path)
        
        # Save the config
        self.config.save_pretrained(save_directory)

    def forward(self, input_ids=None, attention_mask=None, image_embeddings=None, labels=None, **kwargs):
        device = next(self.parameters()).device
        
        if image_embeddings is not None:
            image_embeddings = image_embeddings.to(device)
            projected_images = self.projector(image_embeddings)
            projected_images = projected_images.unsqueeze(1)
            
            if 'inputs_embeds' in kwargs and kwargs['inputs_embeds'] is not None:
                inputs_embeds = kwargs['inputs_embeds']
                inputs_embeds = torch.cat([projected_images, inputs_embeds], dim=1)
                kwargs['inputs_embeds'] = inputs_embeds
            elif input_ids is not None:
                inputs_embeds = self.get_input_embeddings()(input_ids.to(device))
                inputs_embeds = torch.cat([projected_images, inputs_embeds], dim=1)
                kwargs['inputs_embeds'] = inputs_embeds
                input_ids = None  # Set to None to avoid conflict
            
            if attention_mask is not None:
                attention_mask = torch.cat([torch.ones(image_embeddings.size(0), 1, device=device), attention_mask.to(device)], dim=1)
            else:
                attention_mask = torch.ones(image_embeddings.size(0), inputs_embeds.size(1), device=device)
            
            if labels is not None:
                # Adjust labels to match the new sequence length
                labels = torch.cat([torch.full((labels.size(0), 1), -100, device=device), labels], dim=1)
        
        if labels is not None:
            labels = labels.to(device)
        
        # Ensure attention_mask matches the sequence length
        if 'inputs_embeds' in kwargs:
            seq_length = kwargs['inputs_embeds'].size(1)
        elif input_ids is not None:
            seq_length = input_ids.size(1)
        else:
            raise ValueError("Either input_ids or inputs_embeds should be provided")
        
        if attention_mask is not None and attention_mask.size(1) != seq_length:
            attention_mask = attention_mask[:, :seq_length]
        
        return self.phi3(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, **kwargs):
        inputs = self.phi3.prepare_inputs_for_generation(input_ids, past=past, attention_mask=attention_mask, **kwargs)
        if 'image_embeddings' in kwargs:
            inputs['image_embeddings'] = kwargs['image_embeddings']
            
            # Adjust attention_mask if it's present
            if attention_mask is not None:
                # Ensure attention_mask has the correct shape
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                # Add an extra attention mask token for the image embedding
                image_attention = torch.ones((attention_mask.size(0), 1, 1, 1), device=attention_mask.device)
                attention_mask = torch.cat([image_attention, attention_mask], dim=-1)
            inputs['attention_mask'] = attention_mask
        
        return inputs

    def get_input_embeddings(self):
        return self.phi3.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.phi3.set_input_embeddings(value)

    def gradient_checkpointing_enable(self, **kwargs):
        self.phi3.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.phi3.gradient_checkpointing_disable()
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.phi3, name)

    def generate(self, input_ids=None, attention_mask=None, image_embeddings=None, **kwargs):
        if image_embeddings is not None:
            kwargs['image_embeddings'] = image_embeddings
        
        if attention_mask is not None and image_embeddings is not None:
            # Add an extra attention mask token for the image embedding
            attention_mask = torch.cat([torch.ones(attention_mask.size(0), 1, device=attention_mask.device), attention_mask], dim=1)
        
        return super().generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)
# Load the model again for quantization
### Download Phi-3 model
phi3_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map=device_map
)

# Initialize the projector
image_embedding_dim = len(hf_dataset[0]['image_embedding'])
projection_dim = phi3_model.config.hidden_size  # Get dimension from the model
projector = ImageProjector(image_embedding_dim, projection_dim).to(device)

# Combine Phi-3 with the projector
model = Phi3WithProjector(phi3_model, projector)
# Prepare the model for k-bit training
model = prepare_model_for_kbit_training(model)
print(model)


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

# Define LoRA configuration
lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=["query_key_value"],
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
print_trainable_parameters(model)

print(model)


gc.collect()
torch.cuda.empty_cache()
# Define training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
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
    eval_steps=25, # Evaluate every 20 steps
    # Enable gradient checkpointing
    gradient_checkpointing=gradient_checkpointing,
    # Disable data parallelism if not needed
    ddp_find_unused_parameters=False,
)

# Custom data collator to handle pre-tokenized inputs
def custom_data_collator(features):
    batch = {k: [d[k] for d in features] for k in features[0].keys()}
    
    # Stack image embeddings
    batch['image_embeddings'] = torch.stack(batch['image_embeddings'])
    
    # Pad the sequences
    batch['input_ids'] = torch.nn.utils.rnn.pad_sequence(batch['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id)
    batch['attention_mask'] = torch.nn.utils.rnn.pad_sequence(batch['attention_mask'], batch_first=True, padding_value=0)
    batch['labels'] = torch.nn.utils.rnn.pad_sequence(batch['labels'], batch_first=True, padding_value=-100)
    
    return batch

from datasets import DatasetDict
import random

# Function to select a random subset of the dataset
def select_subset(dataset, fraction=0.05):
    num_samples = int(len(dataset) * fraction)
    indices = random.sample(range(len(dataset)), num_samples)
    return dataset.select(indices)

# Select 5% of the training and test datasets
small_train_dataset = select_subset(dataset_dict['train'], fraction=0.05)
small_test_dataset = select_subset(dataset_dict['test'], fraction=0.05)

# Create a new DatasetDict with the smaller datasets
small_dataset_dict = DatasetDict({
    'train': small_train_dataset,
    'test': small_test_dataset
})

print(f"Small train dataset size: {len(small_dataset_dict['train'])}")
print(f"Small test dataset size: {len(small_dataset_dict['test'])}")

# Update the trainer to use the smaller datasets
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
)

# Start training
trainer.train()

# Save the fine-tuned model
trainer.model.save_pretrained(new_model)



# Sample inference code
# Sample inference code using the fine-tuned model
import torch

# Create a custom text generation class
class CustomTextGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, input_text, image_embedding, **generate_kwargs):
        # Tokenize the input text
        inputs = self.tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)
        
        # Ensure image_embedding is a tensor and move it to the correct device
        if not isinstance(image_embedding, torch.Tensor):
            image_embedding = torch.tensor(image_embedding)
        image_embedding = image_embedding.to(self.model.device)
        
        # Adjust attention_mask to account for the image embedding token
        image_attention = torch.ones((1, 1), dtype=torch.long, device=self.model.device)
        attention_mask = torch.cat([image_attention, attention_mask], dim=1)
        
        # Generate text
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_embeddings=image_embedding.unsqueeze(0),  # Add batch dimension
            **generate_kwargs
        )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

# Initialize the custom text generator
generator = CustomTextGenerator(model=model, tokenizer=tokenizer)

# Get a sample from the validation set
sample = dataset_dict['test'][0]
image_embedding = sample['image_embeddings']


def get_first_user_input(decoded_text):
    # Find the position of the first question mark
    question_mark_pos = decoded_text.find('?')
    
    # If a question mark is found, truncate the text
    if question_mark_pos != -1:
        return decoded_text[:question_mark_pos + 1]  # Include the question mark
    else:
        return decoded_text.strip()  # Return the full text if no question mark is found

# Decode the input_ids
full_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)

# Extract only the first user input
input_text = get_first_user_input(full_text)

# Generate text
generated_text = generator.generate(
    input_text,
    image_embedding=image_embedding,
    max_length=200,
    num_return_sequences=1,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
)

print("Input text:")
print(input_text)
print("\nGenerated text:")
print(generated_text)

# Note: This sample code doesn't include the image embedding in the generation process.
# To fully utilize the multimodal capabilities, you'd need to modify the model's forward
# pass to incorporate the image embedding, which is beyond the scope of this simple example.


### merge models and save in gdrive

gc.collect()
torch.cuda.empty_cache()

# # Save the projector
projector_path = '/content/drive/MyDrive/multimodel_llm/image_projector.pth'
torch.save(model.projector.state_dict(), projector_path)
print(f"Projector saved to: {projector_path}")

# Merge the fine-tuned adapter with the base model
# from peft import AutoPeftModelForCausalLM
from peft import PeftModel

# Load the fine-tuned model with the LoRA adapter
# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, new_model)

# Merge the LoRA adapter with the base model
merged_model = model.merge_and_unload()

# Initialize the projector
image_embedding_dim = len(hf_dataset[0]['image_embedding'])
projection_dim = merged_model.config.hidden_size  # Get dimension from the model
projector = ImageProjector(image_embedding_dim, projection_dim).to(device)
projector.load_state_dict(torch.load(projector_path))
# Combine Phi-3 with the projector
phi3_with_projector = Phi3WithProjector(merged_model, projector)

# Define the path to save the merged model in Google Drive
merged_model_path = '/content/drive/MyDrive/multimodel_llm/merged_phi3_llava_model'

# Save the merged model with the projector
phi3_with_projector.save_pretrained(merged_model_path)

# Save the tokenizer
# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.save_pretrained(merged_model_path)

print(f"Merged model and tokenizer saved to: {merged_model_path}")



# Optionally, you can also push the model to the Hugging Face Hub
# Uncomment the following lines if you want to upload the model
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path=merged_model_path,
    repo_id="sayanbanerjee32/multimodal-phi3-4k-instruct-llava",
    repo_type="model",
)
print("Model uploaded to Hugging Face Hub")
