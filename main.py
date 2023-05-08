import logging

import click
from dotenv import load_dotenv

from hugginggpt import generate_response, infer, plan_tasks, select_model
from hugginggpt.history import ConversationHistory
from hugginggpt.log import setup_logging
from hugginggpt.model_factory import MODEL_CHOICES, TEXT_DAVINCI_003, create_llms, LLMs
from hugginggpt.model_scraper import get_top_k_models
from hugginggpt.task_parsing import TaskSummary

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "-l",
    "--llm",
    "llm_type",
    type=click.Choice(MODEL_CHOICES, case_sensitive=False),
    default=TEXT_DAVINCI_003,
    help="Large language model to use as main conversational agent",
)
@click.option("-p", "--prompt", type=str, help="Prompt for huggingGPT")
def main(prompt, llm_type):
    _print_banner()
    models = create_llms(llm_type=llm_type)
    if prompt:
        standalone_mode(user_input=prompt, models=models)

    else:  # interactive mode
        interactive_mode(models=models)


def standalone_mode(user_input: str, models: LLMs):
    try:
        response = _compute(user_input=user_input, models=models)
        return response
    except Exception as e:
        logger.exception("")
        print(
            f"Sorry, encountered error: {e}. Please try again. Check logs if problem persists."
        )


def interactive_mode(models: LLMs):
    print("Please enter your request. End the conversation with 'exit'")
    history = ConversationHistory()
    while True:
        try:
            user_input = click.prompt("User: ")
            if user_input.lower() == "exit":
                break

            logger.info(f"User input: {user_input}")
            response = _compute(user_input=user_input, history=history, models=models)
            print(f"Assistant:{response}")

            history.add(role="user", content=user_input)
            history.add(role="assistant", content=response)
        except Exception as e:
            logger.exception("")
            print(
                f"Sorry, encountered error: {e}. Please try again. Check logs if problem persists."
            )


def _compute(user_input: str, models: LLMs, history: ConversationHistory | None = None) -> str:
    tasks = plan_tasks(
        user_input=user_input, history=history, llm=models.task_planning_llm
    )

    # TODO find way of parallelising tasks if possible
    # TODO in the meantime, sort tasks by dependency and execute sequentially
    sorted(tasks, key=lambda t: max(t.dep))
    logger.info(f"Sorted tasks: {tasks}")

    task_summaries = {}
    for task in tasks:
        logger.info(f"Starting task: {task}")
        if task.depends_on_generated_resources():
            task = task.replace_generated_resources(task_summaries=task_summaries)
        top_k_models = get_top_k_models(
            task=task.task, top_k=5, max_description_length=100
        )
        model = select_model(
            user_input=user_input,
            task=task,
            top_k_models=top_k_models,
            llm=models.model_selection_llm,
        )
        inference_result = infer(task=task, model_id=model.id)
        task_summaries[task.id] = TaskSummary(
            task=task, model=model, inference_result=inference_result
        )
        logger.info(f"Finished task: {task}")
    logger.info("Finished all tasks")
    logger.debug(f"Task summaries: {task_summaries}")

    response = generate_response(
        user_input=user_input,
        task_summaries=task_summaries,
        llm=models.response_generation_llm,
    )
    return response


def _print_banner():
    with open("resources/banner.txt", "r") as f:
        banner = f.read()
        print(banner)
        logger.info("\n" + banner)


if __name__ == "__main__":
    main()