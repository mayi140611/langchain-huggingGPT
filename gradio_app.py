import gradio as gr
from dotenv import load_dotenv
import os

from app import Client


load_dotenv()
# setup_logging()
# logger = logging.getLogger(__name__)
# init_resource_dirs()

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")






css = ".json {height: 527px; overflow: scroll;} .json-holder {height: 527px; overflow: scroll;}"
with gr.Blocks(css=css) as demo:
    gr.Markdown("<h1><center>langchain-HuggingGPT</center></h1>")
    gr.Markdown(
        "<p align='center'><img src='https://i.ibb.co/qNH3Jym/logo.png' height='25' width='95'></p>"
    )
    gr.Markdown(
        "<p align='center' style='font-size: 20px;'>A lightweight implementation of <a href='https://arxiv.org/abs/2303.17580'>HuggingGPT</a> with <a href='https://docs.langchain.com/docs/'>langchain</a>. No local inference, only models available on the <a href='https://huggingface.co/inference-api'>Hugging Face Inference API</a> are used.</p>"
    )
    gr.HTML(
        """<center><a href="https://huggingface.co/spaces/camillevanhoffelen/langchain-HuggingGPT?duplicate=true"><img src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>Duplicate the Space and run securely with your OpenAI API Key and Hugging Face Token</center>"""
    )

    if not OPENAI_KEY:
        with gr.Row().style():
            with gr.Column(scale=0.85):
                openai_api_key = gr.Textbox(
                    show_label=False,
                    placeholder="Set your OpenAI API key here and press Enter",
                    lines=1,
                    type="password",
                ).style(container=False)
            with gr.Column(scale=0.15, min_width=0):
                btn1 = gr.Button("Submit").style(full_height=True)

    if not HUGGINGFACE_TOKEN:
        with gr.Row().style():
            with gr.Column(scale=0.85):
                hugging_face_token = gr.Textbox(
                    show_label=False,
                    placeholder="Set your Hugging Face Token here and press Enter",
                    lines=1,
                    type="password",
                ).style(container=False)
            with gr.Column(scale=0.15, min_width=0):
                btn3 = gr.Button("Submit").style(full_height=True)

    with gr.Row().style():
        with gr.Column(scale=0.6):
            chatbot = gr.Chatbot([], elem_id="chatbot").style(height=500)
        with gr.Column(scale=0.4):
            results = gr.JSON(elem_classes="json")

    with gr.Row().style():
        with gr.Column(scale=0.85):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter. The url must contain the media type. e.g, https://example.com/example.jpg",
                lines=1,
            ).style(container=False)
        with gr.Column(scale=0.15, min_width=0):
            btn2 = gr.Button("Send").style(full_height=True)


    state = gr.State(value={"client": Client()})

    def add_text(state, user_input, messages):
        return state["client"].add_text(user_input, messages)

    def bot(state, messages):
        return state["client"].bot(messages)
    txt.submit(add_text, [state, txt, chatbot], [txt, chatbot]).then(
        bot, [state, chatbot], [results, chatbot]
    )
    btn2.click(add_text, [state, txt, chatbot], [txt, chatbot]).then(
        bot, [state, chatbot], [results, chatbot]
    )

    # gr.Examples(
    #     examples=[
    #         "Draw me a sheep",
    #         "Write a poem about sheep, then read it to me",
    #         "Transcribe the audio file found at /audios/499e.flac. Then tell me how similar the transcription is to the following sentence: Sheep are nice.",
    #         "Tell me a joke about a sheep, then illustrate it by generating an image",
    #     ],
    #     inputs=txt,
    # )
demo.launch()