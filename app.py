import gradio as gr
from multimodal_inference import MultimodalInference
import torch
from transformers import BitsAndBytesConfig


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

def process_input(message, image, audio, history):
    history = history or []
    if not message:
        return "", history
    generated_text = inference.multimodal_inference(
        message,
        image_path=image,
        audio_file=audio
    )
    
    if generated_text.startswith(message):
        generated_text = generated_text[len(message):].strip()
    
    history.append((message, generated_text))
    return "", history

# Define example pairs
examples = [
    [
        "examples/1.png",  # path to example image
        "Can you describe the main features of this image for me?"
    ],
    [
        "examples/2.png",  # path to example image
        "Why might these zebras choose to stay together in this environment?"
    ],
    [
        "examples/3.png",  # path to example image
        "What color is the backsplash in the kitchen?"
    ],
    [
        "examples/4.png",  # path to example image
        "What type of physical and social benefits can be gained from the activity shown in the image?"
    ]
    [
        "examples/5.png",  # path to example image
        "How many police officers are riding horses in the image?"
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
                            placeholder="Press Enter to submit your question",
                            lines=3,
                            max_lines=3,
                            show_label=True,
                            container=True
                        )
                        submit_btn = gr.Button("Submit", variant="primary")
                with gr.TabItem("Voice Question"):
                    audio_input = gr.Audio(label="Or ask your question by voice", type="filepath")
 
    # Add examples
    gr.Examples(
        examples=examples,
        inputs=[image_input, text_input],
        outputs=[chatbot],
        fn=process_input,
        cache_examples=True,
    )

    # Handle text input submission
    text_input.submit(process_input, [text_input, image_input, audio_input, chatbot], [text_input, chatbot])
    submit_btn.click(process_input, [text_input, image_input, audio_input, chatbot], [text_input, chatbot])
    
    # Handle audio input
    audio_input.change(process_input, [audio_input, image_input, audio_input, chatbot], [audio_input, chatbot])
    
    # Handle clear button
    clear.click(lambda: None, None, chatbot, queue=False)

# for collab only
iface.launch(debug = True)

# if __name__ == "__main__":
#     iface.launch()