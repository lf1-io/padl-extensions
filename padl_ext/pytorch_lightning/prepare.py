"""Connector to Pytorch Lightning"""

import shutil
from pathlib import Path
import torch
import padl

import pytorch_lightning as pl


def padl_data_loader(data, padl_model, mode, **kwargs):
    """Create the :class:`~torch.utils.data.DataLoader` used by PADL models.

    This creates a :class:`~torch.utils.data.DataLoader` that can be directly passed to the
    :meth:`pytorch_lightning.Trainer.fit` function.

    Example:

    >>> from padl import transform, batch, identity
    >>> import torch
    >>> from torch.utils.data import DataLoader
    >>> import pytorch_lightning as pl
    >>> @transform
    ... class Net(torch.nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.encoder = torch.nn.Linear(16, 16)
    ...     def forward(self, x):
    ...         return self.encoder(x)
    >>> @transform
    ... def padl_loss(reconstruction, original):
    ...     return torch.nn.functional.mse_loss(reconstruction, original)
    >>> model = identity >> batch >> Net() + identity >> padl_loss
    >>> data_list = [torch.randn([16])] * 4
    >>> train_data_loader = padl_data_loader(data_list, model, 'train', batch_size=2)
    >>> isinstance(train_data_loader, DataLoader)
    True
    >>> val_data_loader = padl_data_loader(data_list, model, 'eval', batch_size=2)
    >>> isinstance(val_data_loader, DataLoader)
    True
    >>> padl_lightning = LightningModule(model)
    >>> trainer = pl.Trainer()
    >>> trainer.fit(padl_lightning, train_data_loader, val_data_loader)

    :param data: List or iterator of data points to be preprocessed by `padl_model`
    :param padl_model: PADL transform to be used in training
    :param mode: PADL mode to call the preprocess :class:`padl.transforms.Transform` in
        (can be 'infer', 'eval' or 'train')
    :param kwargs: Keyword arguments passed to the data loader
        (see :class:`~torch.utils.data.DataLoader` documentation for details).
    """
    return padl_model.pd_get_loader(data, padl_model.pd_preprocess, mode, **kwargs)


# @padl.transform
class LightningModule(pl.LightningModule):
    """PADL connector to Pytorch Lightning.

    :param padl_model: PADL transform. Can provide a string path to load a PADL transform.
    :param train_data: list of training data points
    :param val_data: list of validation data points
    :param test_data: list of test data points
    :param learning_rate: learning rate
    :param kwargs: loader key word arguments for the DataLoader
    """
    def __init__(
        self,
        padl_model,
        train_data=None,
        val_data=None,
        test_data=None,
        learning_rate=1e-4,
        inference_model=None,
        **kwargs
    ):
        super().__init__()

        if isinstance(padl_model, str):
            padl_model = padl.load(padl_model)
        elif not isinstance(padl_model, padl.transforms.Transform):
            raise TypeError('Please provide a PADL transform or a str path to load a '
                            'PADL transform')

        self.padl_model = padl_model

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.loader_kwargs = kwargs
        self.learning_rate = learning_rate
        self.pd_previous = []

        self.padl_model.pd_forward_device_check()

        # Set Pytorch layers as attributes from PADL model
        layers = self.padl_model.pd_layers
        for layer in layers:
            if layer.pd_name is not None:
                prefix = f'{layer.pd_name}'
            else:
                prefix = f'{layer.__class__.__name__}'
            key = prefix
            counter = 0
            while hasattr(self, key):
                key = f'{prefix}_{counter}'
                counter += 1
            setattr(self, key, layer)

        self.inference_model = inference_model

    def forward(self, x):
        """In pytorch lightning, forward defines the prediction/inference actions."""
        return None

    def train_dataloader(self):
        """Creates the train :class:`~torch.utils.data.DataLoader`
        using :meth:`self.padl_model.pd_get_loader` if `self.train_data` is provided.
        """
        if self.train_data is not None:
            return self.padl_model.pd_get_loader(self.train_data, self.padl_model.pd_preprocess,
                                                 'train', **self.loader_kwargs)
        return None

    def val_dataloader(self):
        """Creates the validation :class:`~torch.utils.data.DataLoader`
        using :meth:`self.padl_model.pd_get_loader` if `self.val_data` is provided.
        """
        if self.val_data is not None:
            return self.padl_model.pd_get_loader(self.val_data, self.padl_model.pd_preprocess,
                                                 'eval', **self.loader_kwargs)
        return None

    def test_dataloader(self):
        """Creates the test :class:`~torch.utils.data.DataLoader`
        using :meth:`self.padl_model.pd_get_loader` if `self.test_data` is provided.
        """
        if self.test_data is not None:
            return self.padl_model.pd_get_loader(self.test_data, self.padl_model.pd_preprocess,
                                                 'eval', **self.loader_kwargs)
        return None

    def training_step(self, batch, batch_idx):
        """Default training step overwritten from
        :meth:`pytorch_lightning.LightningModule.training_step`.

        Note: The data loader generated by :meth:`self.padl_model.pd_get_loader` returns a
            tuple of (idx, batch). We only need the batch here.

        :param batch: The output of your :class:`~torch.utils.data.DataLoader`.
            A tensor, tuple or list.
        :param batch_idx: Integer displaying index of this batch
        """
        _, batch = batch
        loss = self.padl_model.pd_forward.pd_call_in_mode(batch, 'train')
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Default validation step overwritten from
        :meth:`pytorch_lightning.LightningModule.validation_step`.

        Note: The data loader generated by :meth:`self.padl_model.pd_get_loader` returns a
            tuple of (idx, batch). We only need the batch here.

        :param batch: The output of your :class:`~torch.utils.data.DataLoader`.
            A tensor, tuple or list.
        :param batch_idx: Integer displaying index of this batch
        """
        _, batch = batch
        loss = self.padl_model.pd_forward.pd_call_in_mode(batch, 'eval')
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        """Default test step overwritten from :meth:`pytorch_lightning.LightningModule.test_step`.

        Note: The data loader generated by :meth:`self.padl_model.pd_get_loader` returns a
            tuple of (idx, batch). We only need the batch here.

        :param batch: The output of your :class:`~torch.utils.data.DataLoader`.
            A tensor, tuple or list.
        :param batch_idx: Integer displaying index of this batch
        """
        _, batch = batch
        loss = self.padl_model.pd_forward.pd_call_in_mode(batch, 'eval')
        self.log("test_loss", loss)

    def on_save_checkpoint(self, checkpoint):
        """Called by Lightning when saving a checkpoint to give you a chance to store anything
        else you might want to save.

        Adding PADL saving by implementing the
        :meth:`pytorch_lightning.LightningModule.on_save_checkpoint` callback and calling
        :meth:`self.padl_model.save`.
        This works in addition to the :class:`pytorch_lightning.callbacks.ModelCheckpoint`
        callback which will still save the pytorch lightning ckpt file.

        :param checkpoint: The full checkpoint dictionary before it gets dumped to a file.
            Implementations of this hook can insert additional data into this dictionary.
        """
        callback_state_keys = checkpoint['callbacks'].keys()
        callback_state_keys = [str(k) for k in callback_state_keys]
        checkpoint_callback_key = [k for k in callback_state_keys if 'ModelCheckpoint' in k][0]
        checkpoint_callback = checkpoint['callbacks'][checkpoint_callback_key]

        best_model_path = checkpoint_callback.get('best_model_path')
        best_model_path = best_model_path.replace(Path(best_model_path).suffix, '')

        best_k_model_paths = checkpoint_callback.get('best_k_models')
        if best_k_model_paths is not None:
            best_k_model_paths = [x.replace(Path(x).suffix, '') for x in best_k_model_paths.keys()]
        else:
            best_k_model_paths = []

        if len(best_k_model_paths) == 0:
            path = best_model_path + '.padl'
        else:
            path = best_k_model_paths[-1] + '.padl'

        if path not in self.pd_previous:
            self.pd_previous.append(path)
            self.padl_model.pd_save(path, force_overwrite=True)
            if self.inference_model is not None:
                self.inference_model.pd_save(path.replace('.padl', '.infer.padl') ,
                                             force_overwrite=True)

        if len(best_k_model_paths) == 0:
            k = 1
        else:
            k = len(best_k_model_paths)

        del_dirpath = None
        if len(self.pd_previous) == k + 1:
            del_dirpath = self.pd_previous.pop(0)
        if del_dirpath is not None:
            shutil.rmtree(del_dirpath)

        # This will get saved in the Pytorch Lightning ckpt and is needed for reloading the ckpt
        checkpoint['padl_models'] = self.pd_previous

    def configure_optimizers(self):
        """Implementation of the inherited method
        :meth:`pytorch_lightning.LightningModule.configure_optimizers`.

        A default optimizer is provided for ease of use, but feel free to overwrite for your
        specific use case. You can choose what optimizers and learning-rate schedulers to use
        in your optimization. Normally you'd need one. But in the case of GANs or similar you
        might have multiple.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    # def load_from_checkpoint(
    #     cls,
    #     checkpoint_path: Union[str, IO],
    #     map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
    #     hparams_file: Optional[str] = None,
    #     strict: bool = True,
    #     **kwargs,
    # ):
