# Multi-Modal LLM Project: Phi-3.5 with Image Understanding

## Project Overview

This project creates a multi-modal Large Language Model (LLM) based on the Phi-3.5 architecture, capable of processing text, image, and audio inputs. The model is fine-tuned using the Instruct 150k dataset and incorporates CLIP for image embedding.

### Key Features

- Base Model: Phi-3.5 (microsoft/Phi-3.5-mini-instruct)
- Image Processing: CLIP embeddings with custom projection layer
- Audio Processing: Whisper model for speech-to-text
- Training Method: QLoRA (Quantized Low-Rank Adaptation)
- Dataset: Instruct 150k
- Deployment: Hugging Face Spaces with Gradio interface

## Detailed Architecture Description

1. **Base Model**: Uses Phi-3.5-mini-instruct with 4-bit quantization
2. **Image Processing**: 
   - CLIP generates image embeddings. CLIP embeddings were generated beforehand for training using the 
   - Custom projection layer maps CLIP embeddings to Phi-3.5 input dimensions
3. **Audio Processing**:
   - Whisper model handles speech-to-text conversion
   - Automatic English transcription
4. **QLoRA Adaptation**: 
   - 4-bit quantization with float16 compute type
   - Nested quantization support
5. **Custom Phi3.5WithProjector Class**: 
   - Integrates image projector with Phi-3.5 model
   - Handles multi-modal inputs seamlessly

## Gradio App Interface

The project includes a user-friendly Gradio interface with:

1. **Image Input**:
   - Upload functionality
   - Example images with predefined questions
2. **Dual Input Methods**: 
   - Text input with Enter key support
   - Voice input with audio recording
3. **Interface Features**:
   - Tabbed interface for text/voice input
   - Conversation history display
   - Clear conversation button
   - Pre-loaded examples
4. **Processing Capabilities**:
   - Combined text-image processing
   - Voice-to-text conversion
   - Multi-turn conversations

## Performance Metrics

### Training Configuration
- Compute Type: float16
- Quantization: 4-bit
- Quantization Type: NF4
- Nested Quantization: Supported

### Model Evaluation
- Pre-merge Projector Weight Change: ~2.87
- Post-merge Projector Weight Change: ~5.07



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
