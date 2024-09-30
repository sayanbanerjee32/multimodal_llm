import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
import numpy as np

def load_model(model_path, device='cpu'):
    return AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True).to(device)

def compare_models(base_model, fine_tuned_model):
    differences = []
    for (name1, p1), (name2, p2) in zip(base_model.named_parameters(), fine_tuned_model.named_parameters()):
        if name1 == name2:
            diff = torch.norm(p1 - p2).item()
            differences.append((name1, diff))
    return differences

# Load the base model
base_model = load_model("microsoft/Phi-3-mini-4k-instruct")

# Load your fine-tuned model
fine_tuned_model = load_model("/path/to/your/fine_tuned_model")

# Compare the models
differences = compare_models(base_model, fine_tuned_model)

# Sort differences by magnitude
differences.sort(key=lambda x: x[1], reverse=True)

# Print the top 10 differences
print("Top 10 layers with the largest differences:")
for name, diff in differences[:10]:
    print(f"{name}: {diff}")

# Calculate and print statistics
diff_values = [d for _, d in differences]
print(f"\nMean difference: {np.mean(diff_values)}")
print(f"Median difference: {np.median(diff_values)}")
print(f"Max difference: {np.max(diff_values)}")
print(f"Min difference: {np.min(diff_values)}")

# Count layers with significant changes (e.g., difference > 0.01)
significant_changes = sum(1 for _, d in differences if d > 0.01)
print(f"\nLayers with significant changes (>0.01): {significant_changes} out of {len(differences)}")