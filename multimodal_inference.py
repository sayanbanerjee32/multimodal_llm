import torch
from PIL import Image
import clip
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from phi3_with_projector import Phi3WithProjector, ImageProjector
from audio_pipeline import AudioTranscriptionPipeline
import os
from peft.tuners import lora
from bitsandbytes.nn import LinearFP4 #, LinearFP8
from huggingface_hub import hf_hub_download

def load_lora_weights(checkpoint_path, device):
    # Check if the checkpoint_path is a local directory or a Hugging Face repo
    if os.path.isdir(checkpoint_path):
        # Local directory
        lora_weights_path = os.path.join(checkpoint_path, "lora_weights.pt")
        if os.path.exists(lora_weights_path):
            lora_state_dict = torch.load(lora_weights_path, map_location=device)
        else:
            raise FileNotFoundError(f"LoRA weights file not found in {checkpoint_path}")
    else:
        # Assume it's a Hugging Face repo
        try:
            # Try to download the file from the Hugging Face repo
            lora_weights_path = hf_hub_download(repo_id=checkpoint_path, filename="lora_weights.pt")
            lora_state_dict = torch.load(lora_weights_path, map_location=device)
        except Exception as e:
            raise ValueError(f"Error loading LoRA weights from Hugging Face repo: {str(e)}")

    return lora_state_dict

def load_model_with_lora_and_projector(checkpoint_path, device, bnb_config=None, debug=False, for_training=False):
    is_cpu = device == "cpu"
    
    common_kwargs = {
        "trust_remote_code": True,
        "debug": debug
    }
    
    if is_cpu:
        device_map = None
        model = Phi3WithProjector.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map=device_map,
            **common_kwargs
        )
    else:
        device_map = {"": device}  # This maps all modules to the specified device
        model = Phi3WithProjector.from_pretrained(
            checkpoint_path,
            quantization_config=bnb_config,
            device_map=device_map,  # Use the corrected device_map
            torch_dtype=torch.float16,
            attn_implementation='eager',
            **common_kwargs
        )

    model = model.to(device)

    # Load LoRA weights
    lora_state_dict = load_lora_weights(checkpoint_path, device)
    print(f"Total keys in lora_state_dict: {len(lora_state_dict)}")
    
    new_state_dict = {}
    scaling_factors_loaded = 0

    for name, module in model.phi3.named_modules():
        if isinstance(module, lora.Linear4bit):
            lora_A_key = f"{name}.lora_A.weight"
            lora_B_key = f"{name}.lora_B.weight"
            scaling_key = f"{name}.scaling"
            
            if lora_A_key in lora_state_dict and lora_B_key in lora_state_dict:
                new_state_dict[f"{name}.lora_A.default.weight"] = lora_state_dict[lora_A_key]
                new_state_dict[f"{name}.lora_B.default.weight"] = lora_state_dict[lora_B_key]
                
                if scaling_key in lora_state_dict:
                    module.scaling = lora_state_dict[scaling_key]
                    scaling_factors_loaded += 1
            else:
                print(f"Warning: LoRA weights for {name} not found in checkpoint")

    # Load the filtered state dict
    model.phi3.load_state_dict(new_state_dict, strict=False)

    print(f"Loaded LoRA weights: {len(new_state_dict)} / {len(lora_state_dict)}")
    print(f"Loaded scaling factors: {scaling_factors_loaded}")
    print(f"Total LoRA modules processed: {len(new_state_dict) // 2}")

    if not for_training:
        # Prepare the model for inference only if not for training
        for name, module in model.named_modules():
            if isinstance(module, LinearFP4): # or isinstance(module, LinearFP8):
                module.prepare_for_inference()
    else:
        # Ensure the model is in training mode
        model.train()

    return model


class MultimodalInference:
    def __init__(self, model_name, tokenizer_name,
                  peft_model_path=None, clip_model_name="ViT-B/32", debug=False, bnb_config=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.debug = debug

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.tokenizer.padding_side = 'left'
        # Load the model
        if peft_model_path:
            # Use load_model_with_lora_and_projector function
            self.model = load_model_with_lora_and_projector(peft_model_path, self.device, bnb_config=bnb_config, debug=self.debug)
        else:
            # If no PEFT adapter is provided, load the pre-merged model
            is_cpu = self.device == "cpu"
            if is_cpu:
                self.model = Phi3WithProjector.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    debug=self.debug
                )
            else:
                self.model = Phi3WithProjector.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    quantization_config=bnb_config,
                    device_map=self.device,
                    torch_dtype=torch.float16,
                    attn_implementation='eager',
                    debug=self.debug
                )

        self.model = self.model.to(self.device)

        # Load CLIP model
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)

        # Initialize the audio transcription pipeline if needed
        # self.audio_pipeline = AudioTranscriptionPipeline()

    def debug_print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def process_image(self, image_path):
        image = self.clip_preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_embedding = self.clip_model.encode_image(image).squeeze()
        return image_embedding

    def process_audio(self, audio_file):
        # Use the __call__ method of AudioTranscriptionPipeline
        transcription = self.audio_pipeline(audio_file)
        return transcription

    def multimodal_inference(self, text_input, image_path=None, audio_file=None):
        projector = self.model.projector
        weight_change = projector.get_weight_change()
        print(f"Projector weight change: {weight_change}")

        dialogue = [{"role": "system", "content": "You are a helpful assistant."}]

        prompt = f"Given the following information, provide a detailed and accurate response:\n{text_input}\n"
        # if image_path is not None:
        #     prompt += "[An image is provided for this task.]\n"

        # if audio_file is not None:
            # audio_transcription = self.process_audio(audio_file)
            # prompt += f"[AUDIO TRANSCRIPTION: {audio_transcription}]\n"

        dialogue.append({"role": "user", "content": prompt})

        input_text = self.tokenizer.apply_chat_template(dialogue, tokenize=False, add_generation_prompt=True)

        print("Input text:")
        print(input_text)

        # Tokenize the input text
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        image_embedding = None
        if image_path is not None:
            image_embedding = self.process_image(image_path)
            image_embedding = image_embedding.to(self.device).to(next(self.model.parameters()).dtype)

            self.debug_print(f"multimodal_inference input_ids: {input_ids.size()}")
            self.debug_print(f"multimodal_inference image_embedding: {image_embedding.size()}")
            self.debug_print(f"multimodal_inference attention_mask: {attention_mask.size()}")

        with torch.no_grad():
            generate_kwargs = {
                "max_new_tokens": 150,
                "num_return_sequences": 1,
                "do_sample": True,
                "temperature": 0.8,
                "top_k": 40,
                "top_p": 0.9,
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 3,
                # "pad_token_id": self.tokenizer.eos_token_id,
                # "use_cache": True,
            }

            if image_embedding is not None:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_embeddings=image_embedding.unsqueeze(0),
                    **generate_kwargs
                )
            else:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generate_kwargs
                )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
