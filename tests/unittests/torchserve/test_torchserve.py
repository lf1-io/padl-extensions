import pytest
import torch
import tempfile
import padl_ext.torchserve as torchserve
import padl

from subprocess import TimeoutExpired

from tests.material import utils


@pytest.fixture()
def tmp_dir():
    tmp_path = tempfile.TemporaryDirectory(dir='tests/material/')
    yield tmp_path.name
    tmp_path.cleanup()


@padl.transform
class PadlModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(36, 128), torch.nn.ReLU(), torch.nn.Linear(128, 3)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(3, 3), torch.nn.Linear(3, 10)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


@padl.transform
def postprocess(x):
    return torch.argmax(torch.nn.Softmax(dim=1)(x), dim=1)


@pytest.mark.skipif((not utils.check_if_module_installed('torchserve')) and \
                    (not utils.check_if_module_installed("torch-model-archiver")),
                    reason="requires the torchserve and torch-model-archiver")
class TestTorchServe:
    @pytest.fixture(autouse=True, scope='class')
    def init(self, request):
        request.cls.transform_1 = PadlModel() >> postprocess

    def test_torchserve(self, tmp_dir):
        save_dir = tmp_dir + '/temp.padl'
        padl.save(self.transform_1, save_dir)
        with pytest.raises(TimeoutExpired):
            torchserve.prepare_and_serve(save_dir, timeout=5)
        torchserve.stop()
