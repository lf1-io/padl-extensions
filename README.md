![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
[![GitHub Issues](https://img.shields.io/github/issues/lf1-io/padl-extensions.svg)](https://github.com/lf1-io/padl-extensions/issues)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lf1-io/padl-extensions/)
[![LF1 on Twitter](https://badgen.net/badge/icon/twitter?icon=twitter&label)](https://twitter.com/lf1_io)

# PADL-Extensions
Trainers, monitors, connectors for PADL

**PADL-Extensions** was developed at [LF1](https://lf1.io/), an AI innovation lab based in Berlin, Germany.

## Installation
```
pip install padl-extensions
```

## Extras Installation
```
pip install padl-extensions[huggingface]
pip install padl-extensions[pytorch_lightning]
pip install padl-extensions[torchserve]
```
If you are installing locally use
```
pip install -e ".[huggingface]"
pip install -e ".[pytorch_lightning]"
pip install -e ".[torchserve]"
```

## Working with `torchserve`
If your model is stored in `checkpoints/model_dir/model.padl`, you can serve it easily with torchserve
```python
from padl_ext import torchserve
torchserve.prepare_and_serve(("examples/model_dir/model.padl")
```
Default address for inference is: http://127.0.0.1:8080
If your model needs image file for inferences, you can try: 
`curl http://127.0.0.1:8080/predictions/model -T test_image.jpg`

You can easily stop the serving with 
```python
torchserve.stop()
```
