import torch
import numpy as np
from PIL import Image
import clip
from multimodal_inference import MultimodalInference

def load_saved_embedding(npz_file, image_name):
    embeddings = np.load(npz_file, allow_pickle=True)
    return torch.from_numpy(embeddings[image_name])

def generate_embedding(image_path, clip_model_name="ViT-B/32"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load(clip_model_name, device=device)
    
    image = clip_preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = clip_model.encode_image(image).squeeze()
    
    return image_embedding

def cosine_similarity(embedding1, embedding2):
    return torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0)

def euclidean_distance(embedding1, embedding2):
    return torch.norm(embedding1 - embedding2)

def compare_embeddings(npz_file, image_name, image_path, threshold=0.95):
    # Load saved embedding
    saved_embedding = load_saved_embedding(npz_file, image_name)
    
    # Generate new embedding
    new_embedding = generate_embedding(image_path)
    
    # Ensure both embeddings are on the same device and have the same dtype
    saved_embedding = saved_embedding.to(new_embedding.device).to(new_embedding.dtype)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(saved_embedding, new_embedding)
    
    # Calculate Euclidean distance
    distance = euclidean_distance(saved_embedding, new_embedding)
    
    print(f"Cosine Similarity: {similarity.item():.4f}")
    print(f"Euclidean Distance: {distance.item():.4f}")
    
    if similarity > threshold:
        print(f"The embeddings are approximately close (similarity > {threshold}).")
    else:
        print(f"The embeddings are not very close (similarity <= {threshold}).")

if __name__ == "__main__":
    npz_file = "coco_image_embeddings.npz"
    image_name = "000000000139.jpg"  # Replace with an actual image name from your dataset
    image_path = "path/to/000000000139.jpg"  # Replace with the actual path to the image file
    
    compare_embeddings(npz_file, image_name, image_path)
