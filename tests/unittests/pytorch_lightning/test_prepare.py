import pytest
import torch
import tempfile
import shutil
import os

from tests.material import utils

import padl
from padl import transform, identity, batch

from padl_ext.pytorch_lightning.prepare import LightningModule
try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
except (ImportError, ModuleNotFoundError):
    pass


@transform
class PadlEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128), torch.nn.ReLU(), torch.nn.Linear(128, 3)
        )

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding


@transform
class PadlDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(3, 128), torch.nn.ReLU(), torch.nn.Linear(128, 28 * 28)
        )

    def forward(self, x):
        decoding = self.decoder(x)
        return decoding


@transform
def padl_loss(reconstruction, original):
    return torch.nn.functional.mse_loss(reconstruction, original)


class MyModule(LightningModule):
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


@pytest.mark.skipif((not utils.check_if_module_installed('pytorch_lightning')),
                    reason="requires the torchserve and torch-model-archiver")
class TestPadlLightning:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        autoencoder = PadlEncoder() >> PadlDecoder()
        padl_training_model = (
            identity
            >> batch
            >> transform(lambda x: x.view(x.size(0), -1))
            >> autoencoder + identity
            >> padl_loss
        )
        request.cls.transform_1 = padl_training_model
        request.cls.train_data = [torch.randn([28, 28])] * 16
        request.cls.val_data = [torch.randn([28, 28])] * 8

    def test_training_from_load(self):
        dirpath = tempfile.mkdtemp()
        model_dir = os.path.join(dirpath, 'model.padl')
        padl.save(self.transform_1, model_dir)
        trainer = pl.Trainer(max_epochs=4, default_root_dir=dirpath, log_every_n_steps=2)
        padl_lightning = MyModule(model_dir, trainer, batch_size=2, num_workers=0)
        padl_lightning.fit(train_data=self.train_data, val_data=self.val_data)
        shutil.rmtree(dirpath)

    def test_reload_checkpoint(self):
        dirpath = tempfile.mkdtemp()
        dirpath = padl.value(dirpath)
        model_dir = os.path.join(dirpath, 'tmp.padl')
        padl.save(self.transform_1, model_dir)
        trainer = pl.Trainer(max_epochs=4, default_root_dir=dirpath, log_every_n_steps=2)
        padl_lightning = MyModule(model_dir, trainer, batch_size=2, num_workers=0)
        padl_lightning.fit(train_data=self.train_data, val_data=self.val_data)

        loaded_padl_lightning = padl.load(padl_lightning.best_model_path)
        loaded_padl_lightning.fit(train_data=self.train_data, val_data=self.val_data)

        shutil.rmtree(dirpath)
