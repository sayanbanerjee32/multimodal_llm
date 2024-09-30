import torch
from PIL import Image
import clip
from transformers import AutoTokenizer
from phi3_with_projector import Phi3WithProjector
from audio_pipeline import AudioTranscriptionPipeline

class MultimodalInference:
    def __init__(self, model_name, tokenizer_name, clip_model_name="ViT-B/32", debug=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.debug = debug

        self.model = Phi3WithProjector.from_pretrained(
            model_name,
            debug=self.debug
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)

        # Initialize the audio transcription pipeline
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
        dialogue = [{"role": "system", "content": "You are a helpful assistant."}]
        
        prompt = "Given the following information, provide a detailed and accurate response:\n"

        if text_input:
            prompt += f"{text_input}\n"
        else:
            prompt += "Please describe what you observe in the provided media.\n"

        if image_path is not None:
            prompt += "[An image is provided for this task.]\n"
        
        # if audio_file is not None:
            # audio_transcription = self.process_audio(audio_file)
            # prompt += f"[AUDIO TRANSCRIPTION: {audio_transcription}]\n"
        
        dialogue.append({"role": "user", "content": prompt})
        
        # Apply chat template without adding the end token
        inputs = self.tokenizer.apply_chat_template(dialogue, return_tensors="pt", add_generation_prompt=True)
        
        # Remove the last token if it's the end-of-text token
        if inputs[0][-1] == self.tokenizer.eos_token_id:
            inputs = inputs[:, :-1]
        
        # Print the decoded input to verify the prompt structure
        print("Decoded input:")
        print(self.tokenizer.decode(inputs[0]))

        input_ids = inputs.to(self.device)
        attention_mask = torch.ones_like(input_ids)

        image_embedding = None
        if image_path is not None:
            image_embedding = self.process_image(image_path)

        if image_embedding is not None:
            image_attention = torch.ones((input_ids.size(0), 1), dtype=torch.long, device=self.device)
            self.debug_print(f"multimodal_inference input_ids: {input_ids.size()}")
            self.debug_print(f"multimodal_inference image_embedding: {image_embedding.size()}")
            self.debug_print(f"multimodal_inference before: {attention_mask.size()}")
            attention_mask = torch.cat([image_attention, attention_mask], dim=1)
            self.debug_print(f"multimodal_inference after: {attention_mask.size()}")

        with torch.no_grad():
            if image_embedding is not None:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_embeddings=image_embedding.unsqueeze(0),
                    max_new_tokens=150,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.8,
                    top_k=40,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
            else:
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    # max_new_tokens=100,
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
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text