"""Connector to Pytorch Lightning"""

import os
import shutil
from pathlib import Path
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback


class OnCheckpointSavePadl(Callback):
    def __init__(self):
        super().__init__()
        self.pd_previous = []

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Adding PADL saving to the checkpointing in Pytorch Lightning. It will save both at
        `dirpath` and `best_model_path` as found in `ModelCheckpoint` callback. """
        # TODO This relies on dictionary ordering to be correct
        best_k_model_paths = list(trainer.checkpoint_callback.best_k_models)
        best_k_model_paths = [x.replace(Path(x).suffix, '') for x in best_k_model_paths]
        dirpath = trainer.checkpoint_callback.dirpath

        if len(best_k_model_paths) == 0:
            path = os.path.join(dirpath, 'model.padl')
        else:
            path = best_k_model_paths[-1] + '.padl'

        if path not in self.pd_previous:
            self.pd_previous.append(path)
            pl_module.model.pd_save(path, force_overwrite=True)

        k = len(trainer.checkpoint_callback.best_k_models) + 1 \
            if trainer.checkpoint_callback.save_top_k == -1 else trainer.checkpoint_callback.save_top_k
        del_dirpath = None
        if len(self.pd_previous) == k + 1 and k > 0:
            del_dirpath = self.pd_previous.pop(0)
        if del_dirpath is not None:
            shutil.rmtree(del_dirpath)


class PadlLightning(pl.LightningModule):
    """Connector to Pytorch Lightning

    :param padl_model: PADL transform to be trained
    :param train_data: list of training data points
    :param val_data: list of validation data points
    :param test_data: list of test data points
    :param learning_rate: learning rate
    :param kwargs: loader key word arguments for the DataLoader
    """
    def __init__(
        self,
        padl_model,
        train_data,
        val_data=None,
        test_data=None,
        learning_rate=1e-3,
        **kwargs
    ):
        super().__init__()
        self.model = padl_model
        self.train_data = train_data
        self.learning_rate = learning_rate
        self.val_data = val_data
        self.test_data = test_data
        self.loader_kwargs = kwargs

        self.model.pd_forward_device_check()

        # Set Pytorch layers as attributes from PADL model
        layers = self.model.pd_layers
        for i, layer in enumerate(layers):
            key = f'{layer.__class__.__name__}'
            counter = 0
            while hasattr(self, key):
                key = f'{layer.__class__.__name__}_{counter}'
                counter += 1
            setattr(self, key, layer)

    def forward(self, x):
        """In pytorch lightning, forward defines the prediction/inference actions"""
        return None

    def train_dataloader(self):
        """Create the train dataloader using `padl.transforms.Transform.pd_get_loader`"""
        return self.model.pd_get_loader(self.train_data, self.model.pd_preprocess, 'train',
                                        **self.loader_kwargs)

    def val_dataloader(self):
        """Create the val dataloader using `padl.transforms.Transform.pd_get_loader`
        if *self.val_data* is provided"""
        if self.val_data is not None:
            return self.model.pd_get_loader(self.val_data, self.model.pd_preprocess, 'eval',
                                            **self.loader_kwargs)
        return None

    def test_dataloader(self):
        """Create the test dataloader using `padl.transforms.Transform.pd_get_loader`
        if *self.test_data* is provided"""
        if self.test_data is not None:
            return self.model.pd_get_loader(self.test_data, self.model.pd_preprocess, 'eval',
                                            **self.loader_kwargs)
        return None

    def training_step(self, batch, batch_idx):
        """Default training step"""
        loss = self.model.pd_forward.pd_call_in_mode(batch, 'train')
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Default validation step"""
        loss = self.model.pd_forward.pd_call_in_mode(batch, 'eval')
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        """Default test step"""
        loss = self.model.pd_forward.pd_call_in_mode(batch, 'eval')
        self.log("test_loss", loss)

    def configure_callbacks(self):
        """When passing the `pl.LightingModule` to the `pl.Trainer` these callbacks are added to the
        `pl.Trainer` callbacks. If there are duplicate callbacks these take precedence over the
        `pl.Trainer` callbacks."""
        early_stop = EarlyStopping(monitor="val_loss", mode="min")
        checkpoint = ModelCheckpoint(monitor="val_loss", every_n_val_epochs=1)
        return [early_stop, checkpoint, OnCheckpointSavePadl()]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
