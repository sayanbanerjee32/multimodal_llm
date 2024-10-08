import gradio as gr
from multimodal_inference import MultimodalInference

# Initialize the inference class
inference = MultimodalInference(
    model_name='sayanbanerjee32/multimodal-phi3-4k-instruct-llava',
    tokenizer_name='sayanbanerjee32/multimodal-phi3-4k-instruct-llava'
)

def process_input(message, image, audio, history):
    history = history or []
    generated_text = inference.multimodal_inference(
        message,
        image_path=image,
        audio_file=audio
    )
    
    if generated_text.startswith(message):
        generated_text = generated_text[len(message):].strip()
    
    history.append((message, generated_text))
    return "", history

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
                    text_input = gr.Textbox(label="Type your question here", lines=10)
                with gr.TabItem("Voice Question"):
                    audio_input = gr.Audio(label="Or ask your question by voice", type="filepath")
 
    text_input.submit(process_input, [text_input, image_input, audio_input, chatbot], [text_input, chatbot])
    audio_input.change(process_input, [audio_input, image_input, audio_input, chatbot], [audio_input, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    iface.launch()