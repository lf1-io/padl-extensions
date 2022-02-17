from transformers import pipeline
from padl import transform, IfInfer, batch, unbatch, identity, same
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
class SimplePadder:
    def __init__(self, padding_length):
        self.padding_length = padding_length

    def __call__(self, tensor_):
        shape = tensor_.shape
        tensor_ = tensor_[..., :self.padding_length]
        if shape[-1] < self.padding_length:
            z = torch.zeros(*[*shape[:-1], self.padding_length - shape[-1]])
            z = z.type(tensor_.type())
            tensor_ = torch.cat([tensor_, z], len(shape) - 1)
        return tensor_


@transform
def mysqueeze(dictionary_2):
    output = {}
    for k in dictionary_2:
        output[k] = dictionary_2[k][0]
    return output


@transform
def myunsqueeze(model_output):
    return {k: model_output[k].unsqueeze(0) for k in model_output}


def audio_classification_features():
    ...


def feature_extraction():
    ...


def image_classification():
    pl = pipeline('image-classification')

    @transform
    def add_logits(x):
        output = lambda: None
        output.logits = x
        return output

    t = (
        transform(pl.preprocess)
        >> mysqueeze
        >> batch
        >> same['pixel_values']
        >> transform(pl.model)
        >> transform(lambda x: x.logits)
        >> unbatch
        >> transform(lambda x: x.unsqueeze(0))
        >> add_logits
        >> transform(pl.postprocess)
    )
    return t


def text_classification(padding_length=20):
    pl = pipeline('text-classification')
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


@transform
class _ExceptionNotInfer:
    def __call__(self):
        if self.pd_stage != 'infer':
            raise Exception


@transform
class SetTransientState:
    def __init__(self, model_to_set):
        self.model_to_set = model_to_set

    def __call__(self, h):
        self.model_to_set.h = h


def text_generation(padding_length=20):
    pl = pipeline('text-generation')
    padder = SimplePadder(padding_length)
    targets = (
        transform(pl.preprocess)
        >> same['input_ids']
        >> same[0]
        >> same[1:]
        >> IfInfer(identity, padder)
    )
    trainer = (
        transform(pl.preprocess)
        >> same['input_ids']
        >> same[0]
        >> same[:-1]
        >> IfInfer(identity, padder)
        >> batch
        >> transform(pl.model)
    )
    generator = (
        transform(pl)
        >> _ExceptionNotInfer()
    )
    return trainer, targets, generator


def conditioned_text_generation(conditioner, padding_length=20):
    pl = pipeline('text-generation')
    padder = SimplePadder(padding_length)
    targets = (
            transform(pl.preprocess)
            >> same['input_ids']
            >> same[0]
            >> same[1:]
            >> IfInfer(identity, padder)
    )
    right = (
            transform(pl.preprocess)
            >> same['input_ids']
            >> same[0]
            >> same[:-1]
            >> IfInfer(identity, padder)
            >> batch
    )
    trainer = (
        conditioner / right
        >> identity / SetTransientState(pl.model)
        >> transform(pl.model)
    )
    generator = (
        conditioner
        >> SetTransientState(pl.model)
        >> transform(lambda x: None)
        >> transform(pl.model)
    )
    return trainer, targets, generator

