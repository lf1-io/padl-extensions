"""Connector to Pytorch Lightning"""

import os
from pathlib import Path
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback


# TODO Some issues to resolve
#      If best_model_path is the same between two iterations this will still overwrite it, shoudn't do that
#      We leave behind previous versions of the best model. They should be removed when best_model_path is detected.
class OnCheckpointSavePadl(Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Adding PADL saving to the checkpointing in Pytorch Lightning. It will save both at
        `dirpath` and `best_model_path` as found in `ModelCheckpoint` callback. """
        best_model_path = trainer.checkpoint_callback.best_model_path
        best_model_path = best_model_path.replace(Path(best_model_path).suffix, '')
        dirpath = trainer.checkpoint_callback.dirpath

        if best_model_path == '':
            path = os.path.join(dirpath, 'model')
        else:
            path = best_model_path

        pl_module.model.pd_save(path, force_overwrite=True)


class PADLLightning(pl.LightningModule):
    """Connector to Pytorch Lightning

    :param padl_model: PADL transform to be trained
    :param train_data: list of training data points
    :param val_data: list of validation data points
    :param test_data: list of test data points
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
        """In lightning, forward defines the prediction/inference actions"""
        return None

    def train_dataloader(self):
        """Create the train dataloader using `pd_get_loader`"""
        return self.model.pd_get_loader(self.train_data, self.model.pd_preprocess, 'train',
                                        **self.loader_kwargs)

    def val_dataloader(self):
        """Create the val dataloader using `pd_get_loader` if *self.val_data* is provided"""
        if self.val_data is not None:
            return self.model.pd_get_loader(self.val_data, self.model.pd_preprocess, 'eval',
                                            **self.loader_kwargs)
        return None

    def test_dataloader(self):
        """Create the test dataloader using `pd_get_loader` if *self.test_data* is provided"""
        if self.test_data is not None:
            return self.model.pd_get_loader(self.test_data, self.model.pd_preprocess, 'eval',
                                            **self.loader_kwargs)
        return None

    def training_step(self, batch, batch_idx):
        """Default training step"""
        loss = self.model.pd_forward.pd_call_transform(batch, 'train')
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Default validation step"""
        loss = self.model.pd_forward.pd_call_transform(batch, 'eval')
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        """Default test step"""
        loss = self.model.pd_forward.pd_call_transform(batch, 'eval')
        self.log("test_loss", loss)

    def configure_callbacks(self):
        early_stop = EarlyStopping(monitor="val_loss", mode="min")
        checkpoint = ModelCheckpoint(monitor="val_loss", every_n_val_epochs=1)
        return [early_stop, checkpoint, OnCheckpointSavePadl()]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
