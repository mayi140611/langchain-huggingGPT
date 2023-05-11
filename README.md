# :parrot: :hugs: :robot: Langchain HuggingGPT

Implementation of [HuggingGPT](https://arxiv.org/abs/2303.17580) with [langchain](https://docs.langchain.com/docs/).

## Getting Started

Install the package with pip:

```commandline
pip install -r requirements.txt
pip install -e .
```

or with your preferred virtual environment manager (_this project uses [pdm](https://pdm.fming.dev/) for dependency management_).

Setup your OpenAI and Huggingface Hub credentials:

```commandline
cp .env.example .env
```

Then fill in the `.env` file with your credentials.

## Usage

```commandline
python main.py
```

Then converse with HuggingGPT, e.g:

```commandline
Please enter your request. End the conversation with 'exit'
User: : Draw me a sheep
Assistant:
I have carefully considered your request and based on the inference results, I have generated an image of a sheep for you. The model used for this task was CompVis/stable-diffusion-v1-4, which is a latent text-to-image diffusion model capable of generating high-quality images from text. The generated image can be found at the following URL: /images/81d7.png. I hope this image meets your expectations. Is there anything else I can help you with?
User: :
```

To use the application in standalone mode, use the `--prompt` flag:

```commandline
python main.py --prompt "Draw me a sheep"
```

## Examples

TODO

## Development

### Testing

Run tests with pytest:

```commandline
pytest
```

## Credits

* [JARVIS](https://github.com/microsoft/JARVIS)

