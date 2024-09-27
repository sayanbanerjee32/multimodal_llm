import gradio as gr
from multimodal_inference import MultimodalInference

# Initialize the inference class
inference = MultimodalInference(
    model_name = 'sayanbanerjee32/multimodal-phi3-4k-instruct-llava',
    tokenizer_name = 'sayanbanerjee32/multimodal-phi3-4k-instruct-llava'
)

def process_input(message, history, image, audio):
    history = history or []
    generated_text = inference.multimodal_inference(
        message,
        image_path=image,
        audio_file=audio
    )
    history.append((message, generated_text))
    return "", history

def get_image_input(image_source):
    if image_source == "Upload":
        return gr.Image(label="Upload Image", type="filepath")
    elif image_source == "Webcam":
        return gr.Image(label="Capture Image", type="filepath", tool="editor")

def get_audio_input(audio_source):
    if audio_source == "Upload":
        return gr.Audio(label="Upload Audio", type="filepath")
    elif audio_source == "Microphone":
        return gr.Audio(label="Record Audio", type="filepath", source="microphone")

with gr.Blocks() as iface:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Text Input")
    
    with gr.Row():
        image_source = gr.Radio(["Upload", "Webcam"], label="Image Source", value="Upload")
        image_input = gr.State()
        image_source.change(fn=get_image_input, inputs=image_source, outputs=image_input)
    
    with gr.Row():
        audio_source = gr.Radio(["Upload", "Microphone"], label="Audio Source", value="Upload")
        audio_input = gr.State()
        audio_source.change(fn=get_audio_input, inputs=audio_source, outputs=audio_input)
    
    clear = gr.Button("Clear")
    
    msg.submit(process_input, [msg, chatbot, image_input, audio_input], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    iface.launch()