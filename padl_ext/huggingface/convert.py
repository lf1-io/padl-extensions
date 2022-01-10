from transformers import pipeline
from padl import transform, IfInfer, batch, unbatch, identity
import torch


@transform
class GenericPadder:
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
def mysqueeze(dictionary_2):
    output = {}
    for k in dictionary_2.items():
        output[k] = dictionary_2[k][0]
    return output


@transform
def myunsqueeze(model_output):
    return {k: model_output[k].unsqueeze(0) for k in model_output}


def to_padl(pl, padding_length=20):
    pl = pipeline(pl)
    padder = GenericPadder(padding_length)
    t = (
        transform(pl.preprocess)
        >> mysqueeze
        >> IfInfer(identity, padder)
        >> batch
        >> transform(lambda x: list(x.values()))
        >> transform(pl.model)
        >> transform(lambda x: dict(x))
        >> unbatch
        >> myunsqueeze
        >> transform(pl.postprocess)
    )
    return t
