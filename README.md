# Multi-Modal LLM Project: Phi-3.5 with Image Understanding

## Project Overview

This project creates a multi-modal Large Language Model (LLM) based on the Phi-3.5 architecture, capable of processing text, image, and audio inputs. The model is fine-tuned using the Instruct 150k dataset and incorporates CLIP for image embedding. Audio transcription is performed using Whisper model.

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
   - CLIP generates image embeddings. CLIP embeddings were generated beforehand for training using [`instruct150k_image_embedding_clip.ipynb`](https://github.com/sayanbanerjee32/multimodal_llm/blob/main/instruct150k_image_embedding_clip.ipynb) and persisted to be used later.
   - Custom projection layer maps CLIP embeddings to Phi-3.5 input dimensions. The Image projector architecture is available at [`phi3_with_projector.py`](https://github.com/sayanbanerjee32/multimodal_llm/blob/main/phi3_with_projector.py)
3. **Audio Processing**:
   - Whisper model handles speech-to-text conversion
   - Automatic English transcription
   - Audio processing pipeline is available at [`audio_pipeline.py`](https://github.com/sayanbanerjee32/multimodal_llm/blob/main/audio_pipeline.py)
4. **QLoRA Adaptation**: 
   - 4-bit quantization with float16 compute type
   - Nested quantization support
5. **Custom Phi3WithProjector Class**: 
   - Integrates image projector with Phi-3.5 model
   - Handles multi-modal inputs seamlessly
   - The model architechture is available at [`phi3_with_projector.py`](https://github.com/sayanbanerjee32/multimodal_llm/blob/main/phi3_with_projector.py)

## Huggingface Spaces Gradio App

A recorded demo of the app is available at [YouTube](https://youtu.be/4mX8-28CjjY])  
![image](https://github.com/user-attachments/assets/18e003e9-e055-4803-8791-f78e46638260)

The project is deployed as [Huggingface Spaces Gradio App](https://huggingface.co/spaces/sayanbanerjee32/multimodal_llm_chatbot) with following features:

1. **Image Input**:
   - Upload functionality
   - Example images with predefined questions
2. **Dual Input Methods**: 
   - Text input
   - Voice input with audio recording  
     ![image](https://github.com/user-attachments/assets/e409957c-f86d-47ce-8d4d-b1797761bfd2)

3. **Interface Features**:
   - Tabbed interface for text/voice input
   - Conversation history display
   - Clear conversation button
   - Pre-loaded examples  
     ![image](https://github.com/user-attachments/assets/e3afd7bd-b8af-4add-a9b9-444dbdca6ac7)

4. **Processing Capabilities**:
   - Combined text-image processing
   - Voice-to-text conversion
   - Multi-turn conversations
  
## Training Methodology

The training script is available at [`phi_3_QLoRA_instruct150k.ipynb`](https://github.com/sayanbanerjee32/multimodal_llm/blob/main/phi_3_QLoRA_instruct150k.ipynb)

1. **Model Configuration**
   - Base Model: microsoft/Phi-3.5-mini-instruct
   - Tokenizer: AutoTokenizer with left-side padding
   - Device: CUDA-enabled GPU

2. **QLoRA Training Setup**
   - LoRA Rank (r): 16
   - LoRA Alpha: 16
   - LoRA Dropout: 0.05
   - Quantization: 4-bit (NF4)
   - Compute Type: float16
   - Nested Quantization: Disabled

3. **Training Parameters**
   - Epochs: 1
   - Batch Size: 8 (per device)
   - Gradient Accumulation Steps: 4
   - Learning Rate: 5e-4
   - Weight Decay: 0.0
   - Optimizer: AdamW (torch)
   - LR Scheduler: Linear with 10% warmup
   - Max Gradient Norm: 1.0
   - Gradient Checkpointing: Enabled

4. **Dataset Processing**
   - Train vs Test split: 95% vs 5%. However, only 5% of the training data was used for training.
   - Max Sequence Length: 256 tokens
   - Custom Data Collator for:
     - Image embedding stacking
     - Input sequence padding
     - Attention mask generation
     - Label padding

5. **Evaluation Strategy**
   - Evaluation Steps: Every 50 steps
   - Save Steps: Every 25 steps
   - Logging Steps: Every 25 steps
   - Model evaluation before and after merging
   - Loss tracking for performance monitoring

6. **Custom Components**
   - Image Projector Layer for CLIP embeddings
   - Custom Text Generator for inference
   - Checkpoint management system
   - Weight change detection for projector layer

7. **Model Saving and Checkpointing**
   - Model published at: [multimodal-phi3_5-mini-instruct-llava_adapter](https://huggingface.co/sayanbanerjee32/multimodal-phi3_5-mini-instruct-llava_adapter)
   
   a. **Checkpoint Components**
      - Base Phi-3.5 Model State
      - Tokenizer Configuration
      - Image Projector Weights
      - LoRA Adapter Weights
      - Trainer State
   
   b. **Saving Strategy**
      - Automatic checkpoint creation every 25 steps
      - Only latest checkpoint retained (previous checkpoints removed)
      - Checkpoint directory structure:
        ```
        checkpoint-{step}/
        ├── config.json
        ├── image_projector.pth
        ├── lora_weights.pt
        ├── model.safetensors
        ├── tokenizer_config.json
        └── trainer_state.json
        ```
   
   c. **Component-wise Saving**
      - **Base Model**
        - Saved using HuggingFace's save_pretrained
        - Includes model architecture and weights
        - Preserves model configuration
      
      - **Image Projector**
        - Saved as separate PyTorch state dict
        - Includes weight change monitoring
        - Stores mean and std statistics of weights
        - Gradient norm tracking for debugging
      
      - **LoRA Weights**
        - Separate storage for LoRA components:
          - LoRA A matrices
          - LoRA B matrices
          - Scaling factors
          - Embedding adaptations (if present)
        - CPU offloading before saving
      
      - **Tokenizer**
        - Complete tokenizer configuration
        - Vocabulary and special tokens
      
      - **Training State**
        - JSON format for trainer state
        - Includes global step and optimization state
        - Enables training resumption
   
   d. **Quality Assurance**
      - Weight statistics logging before saving
      - Gradient tracking for projector layers
      - Explicit verification of saved components
      - Checkpoint completion validation

8. **Model Loading from Checkpoint**
   - Model loaded from: [multimodal-phi3_5-mini-instruct-llava_adapter](https://huggingface.co/sayanbanerjee32/multimodal-phi3_5-mini-instruct-llava_adapter)

   a. **Device-Specific Loading**
      
      - GPU Configuration:
        - Half precision (float16)
        - 4-bit quantization support
        - Eager attention implementation
        - Automatic device mapping to CUDA

   b. **Component Loading Process**
      - **Base Model**
        - Loaded via Phi3WithProjector.from_pretrained
        - Includes model architecture and configuration
        - Quantization applied if on GPU
      
      - **LoRA Weights**
        - Two-step loading process:
          1. Load raw state dictionary
          2. Filter and remap weights to model structure
        - Components loaded:
          - LoRA A matrices (default weights)
          - LoRA B matrices (default weights)
          - Scaling factors
        - Strict=False loading to allow partial updates
      
      - **Mode-Specific Preparation**
        - Training Mode:
          - Preserves 4-bit quantization
          - Maintains training capabilities
        
        - Inference Mode:
          - Prepares LinearFP4 modules
          - Optimizes for inference
          - Converts necessary components

   c. **Loading Verification**
      - Tracks number of loaded LoRA weights
      - Monitors scaling factor loading
      - Reports total LoRA modules processed
      - Warns about missing weights

## Challenges Faced During Development

1. **Projector Training Issues**
   - **Problem Description**
     - Initially, projector weights remained static during training
     - No gradient updates were observed for projector layers
     - CLIP embeddings weren't being properly adapted to model space
   
   - **Technical Analysis**
     - Root cause identified:
       ```python
       # PEFT configuration was freezing all non-LoRA weights
       peft_config = LoraConfig(
           task_type=TaskType.CAUSAL_LM,
           inference_mode=False,
           r=lora_r,
           lora_alpha=lora_alpha,
           lora_dropout=lora_dropout
       )
       ```
     - Issues found:
       - PEFT's default behavior freezes all base model weights
       - Projector layer was considered part of base model
       - No gradient computation for projector parameters
   
   - **Solution Implemented**
     - Explicitly set projector to training mode:
       ```python
       # Ensure projector training
       model.projector.train()
       for param in model.projector.parameters():
           param.requires_grad = True
       ```
     - Added gradient tracking and monitoring
     - Implemented weight change detection

2. **LoRA Adapter Merging Issues**
   - **Problem Description**
     - Significant performance degradation after merging LoRA weights with base model
     - Loss increased from ~0.18 (with adapter) to ~5.07 (after merge)
     - Merged model showed inconsistent responses
   
   - **Technical Analysis**
     - Original approach:
       ```python
       # Problematic merging approach
       merged_model = model.merge_and_unload()
       merged_model.save_pretrained("merged_model")
       ```
     - Issues identified:
       - Quantization precision loss during merging
       - Weight scaling inconsistencies
       - Gradient information loss
   
   - **Solution Implemented**
     - Kept base model and LoRA weights separate
     - Modified deployment strategy:
       1. Load base model with 4-bit quantization
       2. Load LoRA weights separately
       3. Apply weights during runtime
     - Required GPU deployment instead of CPU
     - Added explicit weight verification during loading

3. **Deployment Impact**
   - **Infrastructure Changes**
     - Shifted from CPU to GPU deployment
     - Required Hugging Face Space with GPU support
     - Increased memory requirements
   
   - **Performance Benefits**
     - Maintained original model performance (~0.18 loss)
     - Preserved 4-bit quantization advantages
     - Enabled efficient weight loading
   
   - **Trade-offs**
     - Higher deployment costs
     - More complex loading process
     - Increased startup time
     - Required careful memory management

4. **Lessons Learned**
   - LoRA merging isn't always optimal for quantized models
   - Separate weight management can outperform merged models
   - GPU deployment essential for maintaining model quality
   - Importance of monitoring loss metrics during deployment
   - PEFT configurations can unexpectedly freeze custom layers
   - Explicit gradient tracking essential for custom components
   - Weight change monitoring crucial for validation
   - Custom layers may need manual training mode setting

5. **Checkpoint Resumption Issues (Unresolved)**
   - **Problem Description**
     - Unable to resume training from saved checkpoints
     - Critical for Colab environment due to frequent disconnections
     - Limits effective training duration and data exposure
     - Results in suboptimal model performance
   
   - **Technical Analysis**
     - Issues encountered:
       ```python
       # Loading checkpoint attempts fail with state mismatches
       trainer = Trainer(
           model=model,
           args=training_args,
           train_dataset=train_dataset,
           resume_from_checkpoint=checkpoint_path  # Fails to properly resume
       )
       ```
     - Complications:
       - Mismatch between saved and loaded optimizer states
       - LoRA weight restoration inconsistencies
       - Projector state synchronization problems
       - Loss of training progress tracking
   
   - **Current Impact**
     - Limited training duration (~10-15 hours per session)
     - Reduced training data exposure
     - Model outputs often misaligned with input images
     - Performance bottleneck for model improvement
   
   - **Attempted Solutions**
     - Custom checkpoint loading logic
     - State dict manipulation
     - Separate handling of LoRA and projector states
     - All attempts unsuccessful so far
   
   - **Workaround Strategy**
     - Complete training within single Colab session
     - Use smaller dataset for initial development
     - Focus on architecture validation
     - Accept suboptimal performance temporarily

6. **Future Work Required**
   - Implement robust checkpoint resumption
   - Extend training duration capability
   - Increase training data exposure
   - Improve image-text alignment in outputs
   - Consider alternative training environments
   - Develop better state preservation methods

## Potential Improvements and Future Work

1. **Training Infrastructure**
   - Resolve checkpoint resumption for interrupted training sessions
   - Implement distributed training support
   - Explore cloud platforms beyond Colab for longer training runs
   - Add training progress visualization and monitoring

2. **Quality Improvements**
   - Better image-text alignment in responses
   - Improve response coherence and relevance
   - Implement content filtering
   - Add support for multiple languages

3. **Model Architecture**
   - Experiment with different projection layer architectures
   - Explore alternative image embedding models beyond CLIP
   - Implement flash attention for better performance (This was not suppoted in collab)
   - Add support for larger context windows

4. **Training Process**
   - Extend training iterations for better performance
   - Optimize hyperparameters (learning rates, batch sizes)
   - Incorporate larger proportion of the present datasets

5. **Model Deployment**
   - Optimize CPU inference for wider accessibility
   - Implement model pruning for size reduction
   - Improve error handling and recovery
   - Implement caching for frequent queries

6. **Gradio Interface**
   - Add batch processing capability
   - Implement conversation memory management
   - Add support for multiple images in conversation

7. **Audio Processing**
   - Add language selection for transcription
   - Implement noise reduction
   - Add support for longer audio clips
   - Optimize audio processing pipeline
   - Add audio response generation

8. **Evaluation and Monitoring**
   - Implement comprehensive evaluation metrics
   - Add automated testing for model outputs
   - Create benchmark suite for performance tracking
   - Add monitoring for production deployment

## Development Experience with Cursor + Claude

### Benefits and Observations

1. **Prompt-Driven Development**
   - Majority of code was written through AI prompts
   - Rapid prototyping and implementation
   - Easy to explore multiple approaches
   - Quick generation of boilerplate code
   - Helpful for documentation generation

2. **Problem-Solving Patterns**
   - Claude often suggests workarounds rather than fundamental fixes
   - Example: LoRA merging issue
     - Initial suggestions focused on parameter tweaking
     - Eventually needed complete architectural change
   - Quick fixes can accumulate technical debt
   - Important to push for root cause analysis

3. **Code Understanding Challenges**
   - Accepting complex generated code without understanding leads to:
     - Difficulty in debugging
     - Inconsistent implementations
     - Redundant or unnecessary code
     - Integration problems
   - Required frequent code cleanup and refactoring
   - Important to review and understand each generated segment

4. **Code Oversight Importance**
   - Loss of code oversight can become exponentially problematic
   - Critical to maintain:
     - Clear architecture vision
     - Consistent coding patterns
     - Documentation of design decisions
     - Understanding of dependencies
   - Regular code review and cleanup essential
   - Technical debt can accumulate quickly

### Key Learnings

1. **Best Practices**
   - Review and understand all generated code
   - Question workarounds and seek root causes
   - Maintain clear project structure
   - Document architectural decisions
   - Regular code cleanup sessions

2. **Anti-Patterns to Avoid**
   - Blindly accepting complex solutions
   - Accumulating workarounds
   - Losing architectural oversight
   - Skipping code understanding
   - Delayed cleanup and refactoring

3. **Recommended Workflow**
   - Start with clear problem definition
   - Break down complex problems
   - Review and understand generated code
   - Test thoroughly before integration
   - Regular cleanup and documentation
   - Maintain architectural vision

4. **Balance Points**
   - Speed vs. Understanding
   - Quick fixes vs. Proper solutions
   - Feature addition vs. Code maintenance
   - AI assistance vs. Developer control

