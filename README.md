# padl-extensions
Trainers, monitors, connectors for PADL

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
from padl_ext.torchserve import prepare_and_serve
prepare_and_serve(("checkpoints/model_dir/model.padl")
```
Default address for inference is: http://127.0.0.1:8080
If your model needs image file for inferences, you can try: 
`curl http://127.0.0.1:8080/predictions/model -T test_image.jpg`
