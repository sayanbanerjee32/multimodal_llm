import torch
from transformers import AutoTokenizer
from PIL import Image
import clip
# from phi3_with_projector import Phi3WithProjector  # Assuming this is the correct import

# class MultimodalInference:
#     def __init__(self, model_name, tokenizer_name, clip_model_name="ViT-B/32"):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
#         # Load the fine-tuned model and tokenizer using the new from_pretrained method
#         self.model = Phi3WithProjector.from_pretrained(model_name).to(self.device)
#         self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
#         # Initialize CLIP for image processing
#         self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)
        
#         # Determine the model's dtype
#         self.model_dtype = next(self.model.parameters()).dtype

#     def process_image(self, image_path):
#         image = self.clip_preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
#         with torch.no_grad():
#             image_embedding = self.clip_model.encode_image(image).squeeze()
#         return image_embedding.to(self.model_dtype)  # Convert to model's dtype

#     def multimodal_inference(self, text_input, image_path=None, audio_file=None):
#         # input_text = text_input

#         # Process image if provided
#         image_embedding = None
#         if image_path is not None:
#             image_embedding = self.process_image(image_path)
#             # input_text = f"[IMAGE] {input_text}"

#         # # Process audio if provided
#         # if audio_file is not None:
#         #     audio_transcription = self.audio_pipeline(audio_file)
#         #     input_text = f"{input_text} [AUDIO TRANSCRIPTION: {audio_transcription}]"

#         # Tokenize the input
#         inputs = self.tokenizer(text_input, return_tensors="pt")
#         input_ids = inputs["input_ids"].to(self.device)
#         attention_mask = inputs["attention_mask"].to(self.device)

#         # Generate output
#         with torch.no_grad():
#             if image_embedding is not None:
#                 # Adjust attention_mask to account for the image embedding token
#                 image_attention = torch.ones((1, 1), dtype=torch.long, device=self.device)
#                 attention_mask = torch.cat([image_attention, attention_mask], dim=1)
                
#                 # Use the generate method of Phi3WithProjector
#                 outputs = self.model.generate(
#                     input_ids=input_ids,
#                     attention_mask=attention_mask,
#                     image_embeddings=image_embedding.unsqueeze(0),
#                     max_new_tokens=100,
#                     num_return_sequences=1,
#                     do_sample=True,
#                     temperature=0.7,
#                     top_k=50,
#                     top_p=0.95,
#                     pad_token_id=self.tokenizer.eos_token_id,
#                     use_cache=True,
#                 )
#             else:
#                 outputs = self.model.generate(
#                     input_ids=input_ids,
#                     attention_mask=attention_mask,
#                     max_new_tokens=100,
#                     num_return_sequences=1,
#                     do_sample=True,
#                     temperature=0.7,
#                     top_k=50,
#                     top_p=0.95,
#                     pad_token_id=self.tokenizer.eos_token_id,
#                     use_cache=True,
#                 )

#         # Decode and return the generated text
#         generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return generated_text


# Example usage:
# inference = MultimodalInference("path/to/your/fine-tuned/model", "path/to/your/fine-tuned/tokenizer", "path/to/your/image_projector.pth")
# generated_text = inference.multimodal_inference(
#     "What's in this image?",
#     image_path="path/to/image.jpg",
#     audio_file="path/to/audio.wav"
# )
# print(generated_text)

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

# import os
# import torch
from transformers import PreTrainedModel

class ImageProjector(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class Phi3WithProjector(PreTrainedModel):
    supports_gradient_checkpointing = True

    def __init__(self, phi3_model, projector, debug=False):
        super().__init__(phi3_model.config)
        self.phi3 = phi3_model
        self.projector = projector
        self.debug = debug

    def debug_print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, debug=False, **kwargs):
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
        model = cls(phi3_model, projector, debug=debug)
        return model

    def save_pretrained(self, save_directory):
        # Save the base model
        self.phi3.save_pretrained(save_directory)

        # Save the projector weights
        projector_path = os.path.join(save_directory, "image_projector.pth")
        torch.save(self.projector.state_dict(), projector_path)

        # Save the config
        self.config.save_pretrained(save_directory)

    def forward(self, input_ids=None, attention_mask=None, image_embeddings=None, labels=None, past_key_values=None, **kwargs):
        device = next(self.parameters()).device

        if image_embeddings is not None:
            image_embeddings = image_embeddings.to(device)
            projected_images = self.projector(image_embeddings)
            projected_images = projected_images.unsqueeze(1)
            self.debug_print(f"forward projected_images: {projected_images.size()}")
            
            if past_key_values is None:  # This is the first forward pass
                self.debug_print(f"forward before: {attention_mask.size() if attention_mask is not None else None}")
                if 'inputs_embeds' in kwargs and kwargs['inputs_embeds'] is not None:
                    inputs_embeds = kwargs['inputs_embeds']
                    self.debug_print(f"forward before inputs_embeds: {inputs_embeds.size()}")
                    inputs_embeds = torch.cat([projected_images, inputs_embeds], dim=1)
                    kwargs['inputs_embeds'] = inputs_embeds
                    self.debug_print(f"forward after inputs_embeds: {inputs_embeds.size()}")
                elif input_ids is not None:
                    self.debug_print(f"forward input_ids: {input_ids.size()}")
                    inputs_embeds = self.get_input_embeddings()(input_ids.to(device))
                    self.debug_print(f"forward before inputs_embeds: {inputs_embeds.size()}")
                    inputs_embeds = torch.cat([projected_images, inputs_embeds], dim=1)
                    self.debug_print(f"forward after inputs_embeds: {inputs_embeds.size()}")
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

        # Determine sequence length
        if 'inputs_embeds' in kwargs and kwargs['inputs_embeds'] is not None:
            seq_length = kwargs['inputs_embeds'].size(1)
        elif input_ids is not None:
            seq_length = input_ids.size(1)
        else:
            seq_length = attention_mask.size(1) if attention_mask is not None else None

        if seq_length is None:
            raise ValueError("Unable to determine sequence length. Provide either input_ids, inputs_embeds, or attention_mask.")

        # Ensure attention_mask matches the sequence length
        if attention_mask is not None:
            attention_mask = attention_mask[:, :seq_length]
        
        self.debug_print(f"forward final: input_ids shape: {input_ids.shape if input_ids is not None else None}")
        self.debug_print(f"forward final: attention_mask shape: {attention_mask.shape if attention_mask is not None else None}")
        self.debug_print(f"forward final: inputs_embeds shape: {kwargs.get('inputs_embeds', {}).shape if kwargs.get('inputs_embeds') is not None else None}")
        
        return self.phi3(input_ids=input_ids, attention_mask=attention_mask, labels=labels, past_key_values=past_key_values, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, **kwargs):
        inputs = self.phi3.prepare_inputs_for_generation(input_ids, past=past, attention_mask=attention_mask, **kwargs)
        
        if 'image_embeddings' in kwargs:
            inputs['image_embeddings'] = kwargs['image_embeddings']
            
            if past is None:  # First forward pass
                # Adjust attention_mask to account for the image token
                if attention_mask is not None:
                    inputs['attention_mask'] = torch.cat([torch.ones((attention_mask.size(0), 1), device=attention_mask.device), attention_mask], dim=1)
            else:  # Subsequent passes
                # Ensure attention_mask matches the current sequence length
                if attention_mask is not None:
                    current_seq_length = past[0][0].size(2) + 1  # past key's sequence length + 1 for the new token
                    inputs['attention_mask'] = attention_mask[:, :current_seq_length]

            inputs.pop('position_ids', None)

        # Safe printing of shapes
        self.debug_print(f"prepare_inputs_for_generation: input_ids shape: {inputs['input_ids'].shape if 'input_ids' in inputs else None}")
        self.debug_print(f"prepare_inputs_for_generation: attention_mask shape: {inputs['attention_mask'].shape if 'attention_mask' in inputs else None}")
        self.debug_print(f"prepare_inputs_for_generation: inputs_embeds shape: {inputs.get('inputs_embeds', {}).shape if inputs.get('inputs_embeds') is not None else None}")
        
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
            self.debug_print(f"generate input_ids: {input_ids.size()}")
            self.debug_print(f"generate image_embedding: {image_embeddings.size()}")

        if attention_mask is not None and image_embeddings is not None:
            # Add an extra attention mask token for the image embedding
            self.debug_print(f"generate before: {attention_mask.size()}")
            attention_mask = torch.cat([torch.ones(attention_mask.size(0), 1, device=attention_mask.device), attention_mask], dim=1)
            self.debug_print(f"generate after: {attention_mask.size()}")

        return super().generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)


class MultimodalInference:
    def __init__(self, model_name, tokenizer_name, clip_model_name="ViT-B/32", debug=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.debug = debug

        # Load the fine-tuned model and tokenizer using the new from_pretrained method
        self.model = Phi3WithProjector.from_pretrained(
            model_name,
            debug=self.debug  # Pass the debug flag to Phi3WithProjector
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Initialize CLIP for image processing
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)

    def debug_print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def process_image(self, image_path):
        image = self.clip_preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_embedding = self.clip_model.encode_image(image).squeeze()
        return image_embedding#.to(self.model_dtype)  # Convert to model's dtype

    def multimodal_inference(self, text_input, image_path=None, audio_file=None):
        # input_text = text_input

        # Process image if provided
        image_embedding = None
        if image_path is not None:
            image_embedding = self.process_image(image_path)
            # input_text = f"[IMAGE] {input_text}"

        # # Process audio if provided
        # if audio_file is not None:
        #     audio_transcription = self.audio_pipeline(audio_file)
        #     input_text = f"{input_text} [AUDIO TRANSCRIPTION: {audio_transcription}]"

        # Tokenize the input
        inputs = self.tokenizer(text_input, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)


        if image_embedding is not None:
            # Prepare a 1-element attention mask for the image embedding
            image_attention = torch.ones((input_ids.size(0), 1), dtype=torch.long, device=self.device)
            self.debug_print(f"multimodal_inference input_ids: {input_ids.size()}")
            self.debug_print(f"multimodal_inference image_embedding: {image_embedding.size()}")
            # Concatenate image attention with text attention
            self.debug_print(f"multimodal_inference before: {attention_mask.size()}")
            attention_mask = torch.cat([image_attention, attention_mask], dim=1)
            self.debug_print(f"multimodal_inference after: {attention_mask.size()}")

        # Generate output
        with torch.no_grad():
            if image_embedding is not None:
                # # Adjust attention_mask to account for the image embedding token
                # image_attention = torch.ones((1, 1), dtype=torch.long, device=self.device)
                # attention_mask = torch.cat([image_attention, attention_mask], dim=1)

                # Use the generate method of Phi3WithProjector
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_embeddings=image_embedding.unsqueeze(0),
                    max_new_tokens=100,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
            else:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=100,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )

        # Decode and return the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

# Initialize the inference class
inference = MultimodalInference(
    # model_name='/content/drive/MyDrive/multimodel_llm/merged_phi3_llava_model',
    # tokenizer_name='/content/drive/MyDrive/multimodel_llm/merged_phi3_llava_model',
    model_name = 'sayanbanerjee32/multimodal-phi3-4k-instruct-llava',
    tokenizer_name = 'sayanbanerjee32/multimodal-phi3-4k-instruct-llava'
)

# Perform inference
generated_text = inference.multimodal_inference(
    text_input=text,
    image_path=image_path
)

print("Generated text:")
print(generated_text)


