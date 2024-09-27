import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from phi3_with_projector import Phi3WithProjector, ImageProjector

# Check for CUDA availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("CUDA not available, using CPU")

# Define paths
gdrive_checkpoint_dir = "/content/drive/MyDrive/multimodel_llm/phi3_checkpoints"
model_name = "microsoft/Phi-3-mini-4k-instruct"  # Replace with your base model name

# Find the latest checkpoint
checkpoints = [d for d in os.listdir(gdrive_checkpoint_dir) if d.startswith("checkpoint-")]
latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
latest_checkpoint_path = os.path.join(gdrive_checkpoint_dir, latest_checkpoint)

# Set up model loading arguments based on device
model_args = {
    "low_cpu_mem_usage": True,
    "return_dict": True,
    "torch_dtype": torch.float16 if device.type == "cuda" else torch.float32,
}

if device.type == "cuda":
    model_args["device_map"] = "auto"

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(model_name, **model_args)

# Load the fine-tuned model with the LoRA adapter
model = PeftModel.from_pretrained(base_model, latest_checkpoint_path)

# Merge the LoRA adapter with the base model
merged_model = model.merge_and_unload()

# Load the projector
projector_path = os.path.join(latest_checkpoint_path, "image_projector.pth")
image_embedding_dim = 512  # Replace with your actual image embedding dimension
projection_dim = merged_model.config.hidden_size
projector = ImageProjector(image_embedding_dim, projection_dim).to(device)
projector.load_state_dict(torch.load(projector_path, map_location=device))

# Combine Phi-3 with the projector
phi3_with_projector = Phi3WithProjector(merged_model, projector)

# Define the path to save the merged model in Google Drive
merged_model_path = '/content/drive/MyDrive/multimodel_llm/merged_phi3_llava_model'

# Save the merged model with the projector
phi3_with_projector.save_pretrained(merged_model_path)

# Save the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.save_pretrained(merged_model_path)

print(f"Merged model and tokenizer saved to: {merged_model_path}")

# Optionally, upload to Hugging Face Hub
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path=merged_model_path,
    repo_id="sayanbanerjee32/multimodal-phi3-4k-instruct-llava",
    repo_type="model",
)
print("Model uploaded to Hugging Face Hub")