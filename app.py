import gradio as gr
from multimodal_inference import MultimodalInference
import torch
from transformers import BitsAndBytesConfig
import os


bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_4bit = True
use_nested_quant = False
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)
inference = MultimodalInference(
    model_name='microsoft/Phi-3.5-mini-instruct',
    tokenizer_name= "sayanbanerjee32/multimodal-phi3_5-mini-instruct-llava_adapter",
    peft_model_path= "sayanbanerjee32/multimodal-phi3_5-mini-instruct-llava_adapter",
    bnb_config=bnb_config,
    debug=False
)

def process_input(message, image, audio=None, history=None):
    history = history or []
    
    # Allow audio-only submissions (when there's audio but no text message)
    if audio and not message:
        message = "Please describe what you hear in the audio."
    
    # Check for either image or audio
    if not (image or audio):
        return "", "", history  # Return empty message, empty audio, unchanged history
    
    print(f"Processing input with:")
    print(f"Message: {message}")
    print(f"Image path: {image}")
    print(f"Audio: {audio}")
    
    try:
        generated_text = inference.multimodal_inference(
            message,
            image_path=image,
            audio_file=audio
        )
        
        new_history = history + [(message, generated_text)]
        return "", None, new_history  # Return empty message, clear audio, updated history
    except Exception as e:
        print(f"Error in process_input: {str(e)}")
        new_history = history + [(message, f"Error processing request: {str(e)}")]
        return "", None, new_history  # Return empty message, clear audio, updated history

# Define example pairs with absolute paths
examples = [
    [
        "Can you describe the main features of this image for me?",
        os.path.join(os.getcwd(), "examples", "1.png")
    ],
    [
        "Why might these zebras choose to stay together in this environment?",
        os.path.join(os.getcwd(), "examples", "2.png")
    ],
    [
        "What color is the backsplash in the kitchen?",
        os.path.join(os.getcwd(), "examples", "3.png")
    ],
    [
        "What type of physical and social benefits can be gained from the activity shown in the image?",
        os.path.join(os.getcwd(), "examples", "4.png")
    ],
    [
        "How many police officers are riding horses in the image?",
        os.path.join(os.getcwd(), "examples", "5.png")
    ]
]

with gr.Blocks() as iface:
    gr.Markdown("# Multimodal AI Assistant")
    gr.Markdown("Upload an image, then ask a question about it using text or voice.")
    
    chatbot = gr.Chatbot(label="Conversation")
    with gr.Row():
        with gr.Column(scale=2):
            image_input = gr.Image(label="Upload Image", type="filepath")
            clear = gr.Button("Clear Conversation")
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("Text Question"):
                    with gr.Row():
                        text_input = gr.Textbox(
                            label="Type your question here",
                            placeholder="Press Shift+Enter to submit your question",
                            lines=3,
                            max_lines=3,
                            show_label=True,
                            container=True
                        )
                    with gr.Row():
                        submit_btn = gr.Button("Submit", variant="primary")
                with gr.TabItem("Voice Question"):
                    with gr.Row():
                        audio_input = gr.Audio(
                            label="Or ask your question by voice", 
                            type="filepath"
                        )
                    with gr.Row():
                        audio_submit = gr.Button(
                            "Submit Voice Question", 
                            variant="primary"
                        )
 
    # Add examples with proper handling
    gr.Examples(
        examples=examples,
        inputs=[text_input, image_input],
        outputs=[text_input, audio_input, chatbot],
        fn=process_input,
        cache_examples=False,
        run_on_click=True,
        preprocess=True,
        postprocess=True
    )

    # Handle text input submission
    text_input.submit(
        process_input, 
        [text_input, image_input, audio_input, chatbot], 
        [text_input, audio_input, chatbot]
    )
    submit_btn.click(
        process_input, 
        [text_input, image_input, audio_input, chatbot], 
        [text_input, audio_input, chatbot]
    )
    
    # Handle audio input
    audio_submit.click(
        process_input, 
        [text_input, image_input, audio_input, chatbot], 
        [text_input, audio_input, chatbot]
    )
    
    # Handle clear button
    clear.click(lambda: None, None, chatbot, queue=False)

# # for collab only
# iface.launch(debug = True)

if __name__ == "__main__":
    iface.launch()