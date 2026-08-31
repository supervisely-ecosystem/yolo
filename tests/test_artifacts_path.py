"""Tests for where the training artifacts are looked for after a training.

A repeated training in one live session finished and then lost its checkpoints:
ultralytics increments the run directory when `<project>/<name>` already exists, so the
weights landed in `output/ultralytics2` while the app kept reporting `output/ultralytics`
and the finalize step raised FileNotFoundError.

`supervisely_integration/train/main.py` cannot be imported here, it pulls in the whole
ultralytics and serve stack and talks to the instance, so its part is checked on the
parsed source. `Trainer` is loaded with a stub ultralytics, which keeps the test hermetic.

Run with:  python -m pytest tests/test_artifacts_path.py -v
"""

import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest

TRAIN_DIR = Path(__file__).parents[1] / "supervisely_integration" / "train"
MAIN = TRAIN_DIR / "main.py"
TRAINER = TRAIN_DIR / "trainer.py"


class _FakeUltralyticsTrainer:
    def __init__(self, save_dir):
        self.save_dir = save_dir


class _FakeYOLO:
    """Records what it was trained with and reports the directory it wrote to."""

    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.train_config = None
        self.callbacks = {}
        self.trainer = None

    def add_callback(self, event, func):
        self.callbacks[event] = func

    def train(self, **config):
        self.train_config = config
        # ultralytics renames the run when <project>/<name> is taken and exist_ok is not set
        name = config["name"] if config.get("exist_ok") else config["name"] + "2"
        self.trainer = _FakeUltralyticsTrainer(f"{config['project']}/{name}")
        return self.trainer


@pytest.fixture
def trainer_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=_FakeYOLO))
    spec = importlib.util.spec_from_file_location("yolo_trainer_under_test", TRAINER)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def main_tree():
    return ast.parse(MAIN.read_text(encoding="utf-8"), filename=str(MAIN))


def find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name}() not found in {MAIN}")


def test_train_returns_the_directory_the_trainer_used(trainer_module):
    trainer = trainer_module.Trainer(
        {"model": "yolo11n-seg.pt", "project": "/work/output", "name": "ultralytics"}
    )
    save_dir = trainer.train()

    assert save_dir == "/work/output/ultralytics2"  # the stub renamed it, as ultralytics would
    assert save_dir == trainer.model.trainer.save_dir


def test_train_config_keeps_the_run_in_one_directory(main_tree):
    """`exist_ok` is what stops the rename, and the tensorboard log dir points at the
    unrenamed path, so without it the logs end up watching an empty directory."""
    prepare = find_function(main_tree, "prepare_train_config")
    keys = {
        key.value
        for node in ast.walk(prepare)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant)
    }
    assert "exist_ok" in keys, "the run directory can be renamed under the app"

    values = {
        key.value: value
        for node in ast.walk(prepare)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant)
    }
    assert isinstance(values["exist_ok"], ast.Constant) and values["exist_ok"].value is True


def test_checkpoints_path_comes_from_the_trainer(main_tree):
    """Rebuilding it from train_config["name"] is what broke: the constants say
    `ultralytics` while the weights may sit in `ultralytics2`. The tensorboard log dir may
    still use the constants, it is supposed to point at the unrenamed path."""
    start = find_function(main_tree, "start_training")
    assignment = next(
        node
        for node in ast.walk(start)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "output_checkpoint_dir"
            for target in node.targets
        )
    )
    expression = ast.unparse(assignment.value)

    assert "save_dir" in expression, expression
    assert "train_config" not in expression, (
        f"the checkpoints path is rebuilt from the constants: {expression}"
    )
    assert "trainer.train()" in ast.unparse(start)


def test_exist_ok_keeps_the_run_directory(trainer_module):
    """The pair of fixes together: nothing is renamed, so the reported path is the real one."""
    trainer = trainer_module.Trainer(
        {
            "model": "yolo11n-seg.pt",
            "project": "/work/output",
            "name": "ultralytics",
            "exist_ok": True,
        }
    )
    save_dir = trainer.train()

    assert save_dir == "/work/output/ultralytics"
    assert trainer.model.train_config["exist_ok"] is True


def test_training_config_is_forwarded_verbatim(trainer_module):
    config = {
        "model": "yolo11n-seg.pt",
        "project": "/work/output",
        "name": "ultralytics",
        "exist_ok": True,
        "epochs": 1,
        "batch": 8,
    }
    trainer = trainer_module.Trainer(dict(config))
    trainer.train()

    assert trainer.model.train_config == config


def test_progress_callbacks_are_registered(trainer_module):
    trainer = trainer_module.Trainer(
        {"model": "yolo11n-seg.pt", "project": "/work/output", "name": "ultralytics"}
    )

    assert set(trainer.model.callbacks) == {
        "on_train_start",
        "on_train_epoch_start",
        "on_train_batch_end",
        "on_train_epoch_end",
        "on_train_end",
    }
