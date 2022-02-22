try:
    from padl_ext.pytorch_lightning.prepare import LightningModule
except ImportError:
    print("Please install pytorch lightning dependencies "
          "(pip install padl-extensions[pytorch_lightning]) to use the connector")
    pass

try:
    from padl_ext.torchserve.prepare import prepare
    from padl_ext.torchserve.handler import PadlHandler
except ImportError:
    print("Please install torchserve dependencies "
          "(pip install padl-extensions[torchserve]) to use the connector")
    pass

try:
    from padl_ext.huggingface.convert import to_padl
except ImportError:
    print("Please install huggingface transformers dependencies "
          "(pip install padl-extensions[huggingface]) to use connector")
    pass
