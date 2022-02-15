import pytest
import torch

from tests.material import utils

from padl import transform, identity

from padl_ext.pytorch_lightning.prepare import DefaultPadlLightning
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


@pytest.mark.skipif((not utils.check_if_module_installed('pytorch_lightning')),
                    reason="requires the torchserve and torch-model-archiver")
def test_padl_lightning(tmp_path):
    autoencoder = PadlEncoder() >> PadlDecoder()
    padl_training_model = (
        transform(lambda x: x.view(x.size(0), -1))
        >> autoencoder + identity
        >> padl_loss
    )
    train_data = [torch.randn([28, 28])] * 16
    val_data = [torch.randn([28, 28])] * 8
    trainer = pl.Trainer(max_epochs=4, default_root_dir=str(tmp_path / 'tmp'), log_every_n_steps=2)
    padl_lightning = DefaultPadlLightning(
        padl_training_model, learning_rate=1e-3, train_data=train_data,
        val_data=val_data, batch_size=2, num_workers=0)
    trainer.fit(padl_lightning)
