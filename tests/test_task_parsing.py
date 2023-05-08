import pytest

from hugginggpt.task_parsing import GENERATED_RESOURCES_DIR, GENERATED_TOKEN, Task, parse_tasks


def test_parse_tasks(tasks_str):
    tasks = parse_tasks(tasks_str)
    assert len(tasks) == 1
    assert len(tasks[0].dict()) == 4


def test_parse_complex_tasks(complex_tasks_str):
    tasks = parse_tasks(complex_tasks_str)
    assert len(tasks) == 4


def test_parse_bad_tasks(bad_tasks_str):
    with pytest.raises(Exception):
        parse_tasks(bad_tasks_str)


def test_does_not_depend_on_generated_resources(task):
    assert not task.depends_on_generated_resources()


def test_depends_on_generated_resources(dependent_task):
    assert dependent_task.depends_on_generated_resources()


def test_replace_generated_resources(dependent_task, outputs):
    dependent_task.replace_generated_resources(outputs)
    assert dependent_task.args["image"] == GENERATED_RESOURCES_DIR + "/images/007.png"


def test_do_not_replace_generated_resources(task, outputs):
    original_args = task.args.copy()
    task.replace_generated_resources(outputs)
    assert task.args == original_args


@pytest.fixture
def tasks_str():
    return '[{"task": "text-to-image", "id": 0, "dep": [-1], "args": {"text": "Draw me a sheep." }}]'


@pytest.fixture
def complex_tasks_str():
    return '[{"task": "text-to-image", "id": 0, "dep": [-1], "args": {"text": "Generate an image of two bulls" }}, {"task": "object-detection", "id": 1, "dep": [-1], "args": {"image": "<GENERATED>-0" }}, {"task": "visual-question-answering", "id": 2, "dep": [1], "args": {"image": "<GENERATED>-1", "text": "How many horns in the image"}}, {"task": "text-generation", "id": 3, "dep": [2], "args": {"text": "<GENERATED>-2" }}]'


@pytest.fixture
def bad_tasks_str():
    return '[{"task": "text-to-image", "id": 0, "dep": [-1], "args": {"text": "Draw me a sheep." }]'


@pytest.fixture
def task():
    return Task(task="text-to-image", id=0, dep=[-1], args={"text": "Draw me a sheep."})


@pytest.fixture
def dependent_task():
    return Task(
        task="object-detection",
        id=1,
        dep=[0],
        args={"image": f"{GENERATED_TOKEN}-0"},
    )


@pytest.fixture
def outputs():
    return {0: {"inference result": {"generated image": "/images/007.png"}}}