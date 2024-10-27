import os
import torch
from transformers import PreTrainedModel, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
import torch.nn as nn

class ImageProjector(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=1024):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.05)
        
        # Store initial weights for both layers
        self.register_buffer('initial_weights1', self.layer1.weight.data.clone())
        self.register_buffer('initial_weights2', self.layer2.weight.data.clone())

    def forward(self, x):
        # # Print dtypes
        # print(f"Input dtype: {x.dtype}")
        # print(f"Layer1 weight dtype: {self.layer1.weight.dtype}")
        # print(f"Layer1 bias dtype: {self.layer1.bias.dtype}")
        # print(f"Layer2 weight dtype: {self.layer2.weight.dtype}")
        # print(f"Layer2 bias dtype: {self.layer2.bias.dtype}")

        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

    def get_weight_change(self):
        current_weights1 = self.layer1.weight.data
        current_weights2 = self.layer2.weight.data
        
        # Ensure all tensors are on the same device
        device = current_weights1.device
        initial_weights1 = self.initial_weights1.to(device)
        initial_weights2 = self.initial_weights2.to(device)
        
        weight_diff1 = torch.norm(current_weights1 - initial_weights1).item()
        weight_diff2 = torch.norm(current_weights2 - initial_weights2).item()
        return weight_diff1 + weight_diff2  # Total weight change across both layers

class Phi3WithProjector(PreTrainedModel):
    supports_gradient_checkpointing = True
    _supports_sdpa = True # Add this line

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

        # Determine if it's a local path or a Hugging Face model ID
        is_local = os.path.isdir(pretrained_model_name_or_path)

        if is_local:
            projector_path = os.path.join(pretrained_model_name_or_path, "image_projector.pth")
        else:
            try:
                # Try to download the projector weights from the Hugging Face Hub
                projector_path = hf_hub_download(repo_id=pretrained_model_name_or_path, filename="image_projector.pth")
            except Exception as e:
                print(f"Failed to download projector weights: {e}")
                projector_path = None

        if projector_path and os.path.exists(projector_path):
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

            # Convert projector weights and biases to the same dtype as the main model
            target_dtype = kwargs.get('torch_dtype', torch.float32)
            projector_state_dict = {k: v.to(target_dtype) for k, v in projector_state_dict.items()}

            # Load the state dict with converted weights and biases
            projector.load_state_dict(projector_state_dict, strict=False)
            
            # Ensure all parameters (including biases) are in the correct dtype
            for param in projector.parameters():
                param.data = param.data.to(target_dtype)

            print(f"Loaded projector with input_dim={input_dim}, output_dim={output_dim}, dtype={target_dtype}")
        else:
            print(f"Projector weights not found. Initializing with default dimensions.")
            input_dim = 512  # Default CLIP embedding size
            output_dim = phi3_model.config.hidden_size
            target_dtype = kwargs.get('torch_dtype', torch.float32)
            projector = ImageProjector(input_dim, output_dim)
            # Ensure all parameters (including biases) are in the correct dtype
            for param in projector.parameters():
                param.data = param.data.to(target_dtype)

        # Move the projector to the same device as phi3_model
        projector = projector.to(phi3_model.device)

        # Create and return the Phi3WithProjector instance
        model = cls(phi3_model, projector, debug=debug)
        return model

    def save_pretrained(self, save_directory):
        print(f"Saving model to {save_directory}")
        
        # Save the base model
        self.phi3.save_pretrained(save_directory)

        # Save the projector weights
        projector_path = os.path.join(save_directory, "image_projector.pth")
        projector_state = self.projector.state_dict()
        print(f"Projector weights stats before saving:")
        for name, param in projector_state.items():
            print(f"  {name}: mean={param.mean().item():.4f}, std={param.std().item():.4f}")
        torch.save(projector_state, projector_path)

        # Save the config
        self.config.save_pretrained(save_directory)

        print(f"Model saved successfully to {save_directory}")

    def forward(self, input_ids=None, attention_mask=None, labels=None, image_embeddings=None, past_key_values=None, **kwargs):
        device = next(self.parameters()).device

        if image_embeddings is not None:
            projected_embeddings = self.projector(image_embeddings)
            # Ensure projected_embeddings requires grad
            if not projected_embeddings.requires_grad:
                projected_embeddings.requires_grad_(True)
            projected_embeddings = projected_embeddings.unsqueeze(1)
            self.debug_print(f"forward projected_embeddings: {projected_embeddings.size()}")

            if past_key_values is None:  # This is the first forward pass
                self.debug_print(f"forward before: {attention_mask.size() if attention_mask is not None else None}")
                if 'inputs_embeds' in kwargs and kwargs['inputs_embeds'] is not None:
                    inputs_embeds = kwargs['inputs_embeds']
                    self.debug_print(f"forward before inputs_embeds: {inputs_embeds.size()}")
                    inputs_embeds = torch.cat([projected_embeddings, inputs_embeds], dim=1)
                    kwargs['inputs_embeds'] = inputs_embeds
                    self.debug_print(f"forward after inputs_embeds: {inputs_embeds.size()}")
                elif input_ids is not None:
                    self.debug_print(f"forward input_ids: {input_ids.size()}")
                    inputs_embeds = self.get_input_embeddings()(input_ids.to(device))
                    self.debug_print(f"forward before inputs_embeds: {inputs_embeds.size()}")
                    inputs_embeds = torch.cat([projected_embeddings, inputs_embeds], dim=1)
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
