import threading

import torch
from transformers import BitsAndBytesConfig

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path

from ..config import MODEL_PATH, MODEL_BASE


class ModelManager:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        disable_torch_init()
        model_path = MODEL_PATH
        model_name = get_model_name_from_path(model_path)
        model_base = MODEL_BASE

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            llm_int8_skip_modules=["ecg_tower", "vision_tower", "mm_projector"]
        )

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path,
            model_base,
            model_name,
            quantization_config=bnb_config,
            device_map={"": 0}
        )

    @classmethod
    def get(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
