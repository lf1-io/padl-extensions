import json
import logging

from ts.torch_handler.base_handler import BaseHandler
from padl import load, unbatch
from padl.transforms import Transform

logger = logging.getLogger(__name__)


class PadlHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self._pd_preprocess = None
        self._pd_forward = None
        self._pd_postprocess = None

    def initialize(self, context):
        properties = context.system_properties
        model_dir = properties.get('model_dir')
        logger.warning(model_dir)
        m = load(model_dir)
        self._pd_preprocess = m.pd_preprocess
        self._pd_forward = m.pd_forward
        self._pd_postprocess = m.pd_postprocess

    def preprocess(self, data):
        data = data[0]['body']
        return self._pd_preprocess.infer_apply(data)

    def inference(self, data, *args, **kwargs):
        return self._pd_forward.infer_apply(data)

    def postprocess(self, data):
        Transform.pd_mode = 'infer'
        output = self._pd_postprocess[1:].infer_apply(unbatch(data))
        return [json.dumps(output)]
