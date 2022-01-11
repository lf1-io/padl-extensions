from transformers import pipeline
from padl import transform, IfInfer, batch, unbatch, identity
import torch


@transform
class GenericPadder:
    """Pads the last dimension of each output in *dictionary_1* to length *padding_length*

    :param padding_length: padding length
    """
    def __init__(self, padding_length):
        self.padding_length = padding_length

    def __call__(self, dictionary_1):
        for k in dictionary_1:
            shape = dictionary_1[k].shape
            dictionary_1[k] = dictionary_1[k][..., :self.padding_length]
            if shape[-1] < self.padding_length:
                z = torch.zeros(*[*shape[:-1], self.padding_length - shape[-1]])
                z = z.type(dictionary_1[k].type())
                dictionary_1[k] = torch.cat([dictionary_1[k], z], len(shape) - 1)
        return dictionary_1


@transform
def mysqueeze(preprocess_output: dict):
    """Squeeze each of the outputs in *preprocess_output*

    :param preprocess_output: dictionary of preprocessing outputs
    """
    output = {}
    # FIXME Is this meant to be dictionary_2.keys()?
    for k in preprocess_output.items():
        output[k] = preprocess_output[k][0]
    return output


@transform
def myunsqueeze(model_output: dict):
    """Unsqueeze each of the outputs in *model_output*

    :param model_output: dictionary of model outputs
    """
    return {k: model_output[k].unsqueeze(0) for k in model_output}


def to_padl(hug_task: str, padding_length: int = 20):
    """Convert Huggingface pipeline to a PADL pipeline

    :param hug_task: Huggingface task specifying the pipeline
    :param padding_length: padding length used when running `padl.Transform.eval_apply` and
        `padl.Transform.train_apply`
    """
    hug_pipeline = pipeline(hug_task)
    padder = GenericPadder(padding_length)
    return (
        transform(hug_pipeline.preprocess)
        >> mysqueeze
        >> IfInfer(identity, padder)
        >> batch
        >> transform(lambda x: list(x.values()))
        >> transform(hug_pipeline.model)
        >> transform(lambda x: dict(x))
        >> unbatch
        >> myunsqueeze
        >> transform(hug_pipeline.postprocess)
    )
