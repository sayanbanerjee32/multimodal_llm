import gradio as gr
from multimodal_inference import MultimodalInference

# Initialize the inference class
inference = MultimodalInference(
    model_name="path/to/your/fine-tuned/model",
    tokenizer_name="path/to/your/fine-tuned/tokenizer"
)

def process_input(text_input, image_input, audio_input):
    generated_text = inference.multimodal_inference(
        text_input,
        image_path=image_input,
        audio_file=audio_input
    )
    return generated_text

iface = gr.Interface(
    fn=process_input,
    inputs=[
        gr.Textbox(label="Text Input"),
        gr.Image(label="Image Input", type="filepath"),
        gr.Audio(label="Audio Input", type="filepath")
    ],
    outputs=gr.Textbox(label="Response"),
    title="Multimodal Chat Application",
    description="Enter text, upload an image, or provide audio input to get a response."
)

if __name__ == "__main__":
    iface.launch()