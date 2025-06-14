import logging
import os

import torch
import lightning
import numpy as np
from monai.inferers.inferer import SlidingWindowInfererAdapt


logger = logging.getLogger(__name__)


class PLModule(lightning.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss,
        optimizer_factory,
        prediction_threshold: float,
        scheduler_configs=None,
        evaluator=None
    ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer_factory = optimizer_factory
        self.scheduler_configs = scheduler_configs
        self.prediction_threshold = prediction_threshold
        self.rank = 0 if "LOCAL_RANK" not in os.environ else os.environ["LOCAL_RANK"]
        self.evaluator = evaluator

    def configure_optimizers(self):
        optimizer = self.optimizer_factory(params=self.parameters())

        if self.scheduler_configs is not None:
            schedulers = []
            logger.info(f"Initializing schedulers: {self.scheduler_configs}")
            for scheduler_name, scheduler_config in self.scheduler_configs.items():
                if scheduler_config is None:
                    continue    # skip empty configs during finetuning

                logger.info(f"Initializing scheduler: {scheduler_name}")
                scheduler_config["scheduler"] = scheduler_config["scheduler"](optimizer=optimizer)
                scheduler_config = dict(scheduler_config)
                schedulers.append(scheduler_config)
            return [optimizer], schedulers
        return optimizer

    def training_step(self, batch, batch_idx):
        image, mask = batch
        pred_mask = self.model(image)
        loss = self.loss(pred_mask, mask)
        self.log("train_loss", loss.item(), logger=(self.rank == 0))
        return loss

    def validation_step(self, batch, batch_idx):
        image, mask, name = batch
        pred_mask = self.model(image)
        loss = self.loss(pred_mask, mask)
        self.log("val_loss", loss.item(), logger=(self.rank == 0))

        metrics = self.evaluator.estimate_metrics(
            pred_mask.sigmoid().squeeze(), mask.squeeze(), threshold=self.prediction_threshold
        )
        for metric, value in metrics.items():
            value = value.item() if isinstance(value, (torch.Tensor, np.ndarray)) else value
            self.log(f"val_{name[0]}_{metric}", value, logger=(self.rank == 0))


class PLModuleFinetune(PLModule):
    def __init__(
        self,
        dataset_name: str = None,
        input_size: tuple = None,
        batch_size: int = None,
        num_shots: int = None,
        *args,
        **kwargs
    ):
        # Remove dataset_name from kwargs
        self.dataset_name = dataset_name
        logger.info(f"Dataset name: {self.dataset_name}")
        super().__init__(*args, **kwargs)
        self.sliding_window_inferer = SlidingWindowInfererAdapt(
            roi_size=input_size, sw_batch_size=batch_size, overlap=0.5,
        )
        self.num_shots = num_shots

    def validation_step(self, batch, batch_idx):
        image, mask = batch
        with torch.no_grad():
            pred_mask = self.sliding_window_inferer(image, self.model)
            loss = self.loss(pred_mask, mask)
            self.log(f"{self.dataset_name}_val_loss", loss.item())

            metrics = self.evaluator.estimate_metrics(
                pred_mask.sigmoid().squeeze(), mask.squeeze(), threshold=self.prediction_threshold, fast=True
            )
            for name, value in metrics.items():
                value = value.item() if isinstance(value, (torch.Tensor, np.ndarray)) else value
                self.log(f"{self.dataset_name}_val_{name}", value)
                
                if name == "dice":
                    self.log("val_DiceMetric", value)

        return loss

    def test_step(self, batch, batch_idx):
        image, mask = batch
        with torch.no_grad():
            pred_mask = self.sliding_window_inferer(image, self.model)
            loss = self.loss(pred_mask, mask)
            self.log(f"{self.dataset_name}_test_loss", loss.item())

            try:
                metrics = self.evaluator.estimate_metrics(
                    pred_mask.sigmoid().squeeze(), mask.squeeze(), threshold=self.prediction_threshold
                )
                for name, value in metrics.items():
                    value = value.item() if isinstance(value, (torch.Tensor, np.ndarray)) else value
                    self.log(f"{self.dataset_name}_test_{name}", value)

            except ValueError as e:
                if "Only one class present in y_true" in str(e):
                    logger.warning(
                        f"Skipping metrics calculation for batch {batch_idx} in test_step "
                        f"for dataset {self.dataset_name} due to: {e}"
                    )
                else:
                    # Re-raise other ValueErrors not related to this specific issue
                    raise e

        return loss