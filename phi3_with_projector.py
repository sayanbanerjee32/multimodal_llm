import os
import torch
from transformers import PreTrainedModel, AutoModelForCausalLM

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
                inputs['attention_mask'] = torch.cat([torch.ones(attention_mask.size(0), 1, device=attention_mask.device), attention_mask], dim=1)
            
            # Remove position_ids as they're not used
            inputs.pop('position_ids', None)
        
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