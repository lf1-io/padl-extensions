import pytest
import torch

from tests.material import utils

import padl
from padl import transform, identity, batch

from padl_ext.pytorch_lightning.prepare import LightningModule, padl_data_loader
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
        request.cls.padl_lightning = MyModule(
            self.transform_1, train_data=self.train_data,
            val_data=self.val_data, batch_size=2, num_workers=0)

    def test_training(self, tmp_path):
        trainer = pl.Trainer(max_epochs=4, default_root_dir=str(tmp_path), log_every_n_steps=2)
        trainer.fit(self.padl_lightning)

    def test_training_from_load(self, tmp_path):
        model_dir = str(tmp_path / 'model.padl')
        padl.save(self.transform_1, model_dir)
        padl_lightning = MyModule(model_dir, train_data=self.train_data,
                                  val_data=self.val_data, batch_size=2, num_workers=0)
        trainer = pl.Trainer(max_epochs=4, default_root_dir=str(tmp_path), log_every_n_steps=2)
        trainer.fit(padl_lightning)

    def test_reload_checkpoint(self, tmp_path):
        trainer = pl.Trainer(max_epochs=4, default_root_dir=str(tmp_path), log_every_n_steps=2)
        trainer.fit(self.padl_lightning)
        pl_module = MyModule.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,
            padl_model=trainer.checkpoint_callback.best_model_path.replace('.ckpt', '.padl'),
            train_data=self.train_data,
        )
        trainer.fit(pl_module)

    def test_pass_dataloader_to_trainer(self, tmp_path):
        train_loader = padl_data_loader(self.train_data, self.transform_1, 'train',
                                        batch_size=2, num_workers=0)
        val_loader = padl_data_loader(self.val_data, self.transform_1, 'eval',
                                      batch_size=2, num_workers=0)
        trainer = pl.Trainer(max_epochs=4, default_root_dir=str(tmp_path), log_every_n_steps=2)
        padl_module = LightningModule(self.transform_1)
        trainer.fit(padl_module, train_loader, val_loader)
