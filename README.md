# Multi-Modal LLM Project: Phi-3 with Image Understanding

## Project Overview

This project aims to create a multi-modal Large Language Model (LLM) based on the Phi-3 architecture, capable of processing both text and image inputs. The model is fine-tuned using the Instruct 150k dataset and incorporates CLIP for image embedding.

### Key Features

- Base Model: Phi-3 (microsoft/Phi-3.5-mini-instruct)
- Image Processing: CLIP embeddings with custom projection layer
- Training Method: QLoRA (Quantized Low-Rank Adaptation)
- Dataset: Instruct 150k
- Deployment: Hugging Face Spaces with Gradio interface

## Detailed Architecture Description

1. **Base Model**: We use the Phi-3.5-mini-instruct model as our foundation.
2. **Image Processing**: 
   - CLIP is used to generate image embeddings.
   - A custom projection layer maps CLIP embeddings to Phi-3 input dimensions.
3. **QLoRA Adaptation**: 
   - Enables efficient fine-tuning with reduced memory footprint.
   - Allows for training on consumer-grade GPUs.
4. **Custom Phi3WithProjector Class**: 
   - Integrates the image projector with the Phi-3 model.
   - Handles both text and image inputs seamlessly.

## Performance Metrics and Benchmarks

### Training Logs

```
# Paste training logs here
```

### Evaluation Results

Pre-merge loss: [Insert value here]
Post-merge loss: [Insert value here]

## Gradio App Interface

The project includes a user-friendly Gradio interface for interacting with the multimodal AI assistant. The interface allows users to upload images and ask questions about them using either text or voice input. Here are the key features of the Gradio app:

1. **Image Upload**: Users can upload an image for analysis.
2. **Dual Input Methods**: 
   - Text input: Users can type their questions about the image.
   - Voice input: Users can ask questions verbally, which are then processed by the model.
3. **Conversation History**: The app maintains a chat-like interface, displaying the history of questions and answers.
4. **Clear Conversation**: A button to clear the conversation history and start fresh.
5. **Multimodal Processing**: The backend uses the `MultimodalInference` class to process text, image, and potentially audio inputs together.

The app is designed to be intuitive and accessible, allowing users to easily interact with the advanced multimodal AI model without needing technical expertise.

[Insert screenshot of Gradio app here]

This interface demonstrates the practical application of the multimodal LLM, showcasing its ability to understand and respond to queries about visual content in a conversational manner.

## Audio Processing Pipeline

The project includes an audio processing pipeline that enables the model to handle speech input alongside text and images. This feature enhances the multimodal capabilities of the system, allowing for a more versatile user interaction.

### Key Components

1. **AudioTranscriptionPipeline**:
   - Utilizes the Whisper model from OpenAI for speech-to-text conversion.
   - Supports loading and processing of audio files.
   - Transcribes speech to text for further processing by the main model.

2. **Integration with MultimodalInference**:
   - The `MultimodalInference` class is designed to incorporate audio input.
   - Audio files can be passed alongside text and image inputs for comprehensive multimodal analysis.

### Features

- **Model Flexibility**: Uses the "openai/whisper-base" model by default, but can be configured to use other Whisper variants.
- **GPU Acceleration**: Automatically utilizes CUDA if available for faster processing.
- **Sample Rate Handling**: Processes audio at a 16kHz sample rate, which is standard for many speech recognition tasks.
- **Seamless Integration**: The audio transcription is incorporated into the main prompt for the multimodal model, allowing for contextual understanding of spoken content alongside visual and textual inputs.

### Usage in Multimodal Context

When an audio file is provided to the `multimodal_inference` method:
1. The audio is transcribed using the Whisper model.
2. The transcription is added to the prompt sent to the main language model.
3. This allows the model to consider spoken content in its response generation, alongside any text or image inputs.

### Current Implementation Status

As of the latest update, the audio processing functionality is implemented but commented out in the `MultimodalInference` class. To fully enable this feature:
1. Uncomment the `AudioTranscriptionPipeline` initialization in the `MultimodalInference` constructor.
2. Uncomment the audio processing section in the `multimodal_inference` method.

This modular design allows for easy activation of the audio processing capabilities when needed, while maintaining flexibility in the system's configuration.

## Challenges Faced During Development

Throughout the development of this multi-modal LLM project, we encountered several significant challenges:

1. **Limited Training Iterations**: 
   - We initially trained the model for only 10 iterations, which proved insufficient for proper convergence.
   - This limited training led to suboptimal performance and difficulties in evaluating the model's true capabilities.

2. **Performance Degradation After Merging**:
   - We observed a significant increase in loss after merging the model (Pre-merge loss: 16.06, Post-merge loss: 21.15).
   - This degradation highlighted the need for more robust merging strategies and potentially longer training periods.

3. **Flash Attention Integration**:
   - Warnings about the absence of the `flash-attention` package indicated potential performance improvements that were not realized.

4. **Multi-modal Input Handling**:
   - Integrating image embeddings with text inputs required careful design of the `Phi3WithProjector` class and custom data collators.

These challenges underscored the complexity of developing multi-modal AI systems and highlighted areas for future improvement and optimization.

## Potential Improvements and Future Work

1. **Extended Training**: Increase the number of training iterations for better performance.
2. **Hyperparameter Tuning**: Optimize learning rates, batch sizes, and other hyperparameters.
3. **Larger Dataset**: Incorporate more diverse and extensive datasets for improved generalization.
4. **Model Architecture Enhancements**: 
   - Experiment with different projection layer architectures.
   - Explore alternative image embedding models beyond CLIP.
5. **Multi-modal Pretraining**: Implement pretraining tasks that jointly learn from text and images.
6. **Evaluation on Downstream Tasks**: Assess performance on specific multi-modal tasks like VQA or image captioning.
7. **Optimization for Inference**: Implement techniques like quantization for faster inference.
