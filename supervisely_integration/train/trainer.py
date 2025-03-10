from ultralytics import YOLO

from supervisely.nn.training.loggers import train_logger


class Trainer:
    def __init__(self, train_config: dict):
        """Initialize the Trainer with a model and training configuration."""
        self.model = YOLO(train_config["model"])
        self.model.to(train_config["device"])
        self.train_config = train_config
        self.setup_callbacks()

    def on_train_start(self, trainer):
        """Callback for when training starts."""
        train_logger.train_started(total_epochs=trainer.epochs)

    def on_train_epoch_start(self, trainer):
        """Callback for the start of each training epoch."""
        total_steps = len(trainer.train_loader)
        train_logger.epoch_started(total_steps=total_steps)

    def on_train_batch_end(self, trainer):
        """Callback for the end of each training batch."""
        train_logger.step_finished()

    def on_train_epoch_end(self, trainer):
        """Callback for the end of each training epoch."""
        train_logger.epoch_finished()

    def on_train_end(self, trainer):
        """Callback for when training ends."""
        train_logger.train_finished()

    def setup_callbacks(self):
        """Register all callbacks with the model."""
        self.model.add_callback("on_train_start", self.on_train_start)
        self.model.add_callback("on_train_epoch_start", self.on_train_epoch_start)
        self.model.add_callback("on_train_batch_end", self.on_train_batch_end)
        self.model.add_callback("on_train_epoch_end", self.on_train_epoch_end)
        self.model.add_callback("on_train_end", self.on_train_end)

    def train(self):
        """Start the training process using the provided configuration."""
        self.model.train(**self.train_config)
