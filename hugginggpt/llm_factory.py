import logging
from collections import namedtuple

import tiktoken
from langchain import OpenAI

LLM_NAME = "text-davinci-003"
# Encoding for text-davinci-003
ENCODING_NAME = "p50k_base"
ENCODING = tiktoken.get_encoding(ENCODING_NAME)
# Max input tokens for text-davinci-003
LLM_MAX_TOKENS = 4096

# As specified in huggingGPT paper
TASK_PLANNING_LOGIT_BIAS = 0.1
MODEL_SELECTION_LOGIT_BIAS = 5

logger = logging.getLogger(__name__)

LLMs = namedtuple(
    "LLMs",
    [
        "task_planning_llm",
        "model_selection_llm",
        "model_inference_llm",
        "response_generation_llm",
        "output_fixing_llm",
    ],
)


def create_llms() -> LLMs:
    """Create various LLM agents according to the huggingGPT paper's specifications."""
    logger.info(f"Creating {LLM_NAME} LLMs")

    task_parsing_highlight_ids = get_token_ids_for_task_parsing()
    choose_model_highlight_ids = get_token_ids_for_choose_model()
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(
    #         openai_api_base='http://localhost:11434/v1/',
    #         openai_api_key='ollama',
    #         model='llama3',
    #         temperature=0,
    #         verbose=True,
    #         streaming=True
    #     )
    import os 
    from langchain.chat_models import AzureChatOpenAI
    llm = AzureChatOpenAI(
            openai_api_base=os.getenv('GPT35_AZURE_BASE_URL'),
            openai_api_version=os.getenv('GPT35_VERSION'),
            deployment_name=os.getenv('GPT35_AZURE_DEPLOYMENT_NAME'),
            openai_api_key=os.getenv('GPT35_AZURE_API_KEY'),
            openai_api_type='azure',
            temperature=0,
            streaming=True,
            verbose=True,
            # callbacks=[StreamingStdOutCallbackHandler()]
        )
    task_planning_llm = llm
    model_selection_llm = llm
    model_inference_llm = llm
    response_generation_llm = llm
    output_fixing_llm = llm
    return LLMs(
        task_planning_llm=task_planning_llm,
        model_selection_llm=model_selection_llm,
        model_inference_llm=model_inference_llm,
        response_generation_llm=response_generation_llm,
        output_fixing_llm=output_fixing_llm,
    )


def get_token_ids_for_task_parsing() -> list[int]:
    text = """{"task": "text-classification",  "token-classification", "text2text-generation", "summarization", "translation",  "question-answering", "conversational", "text-generation", "sentence-similarity", "tabular-classification", "object-detection", "image-classification", "image-to-image", "image-to-text", "text-to-image", "visual-question-answering", "document-question-answering", "image-segmentation", "text-to-speech", "automatic-speech-recognition", "audio-to-audio", "audio-classification", "args", "text", "path", "dep", "id", "<GENERATED>-"}"""
    res = ENCODING.encode(text)
    res = list(set(res))
    return res


def get_token_ids_for_choose_model() -> list[int]:
    text = """{"id": "reason"}"""
    res = ENCODING.encode(text)
    res = list(set(res))
    return res


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))
