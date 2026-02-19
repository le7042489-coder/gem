from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from io import BytesIO
from typing import Optional

from PIL import Image

from ..config import IMAGE_CACHE_MAX_ITEMS


class ImageStore:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._images = OrderedDict()
        self._images_lock = threading.Lock()
        self._max_items = max(16, int(IMAGE_CACHE_MAX_ITEMS))

    @classmethod
    def get(cls) -> "ImageStore":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def put_image(self, image: Image.Image) -> str:
        image_id = str(uuid.uuid4())
        buf = BytesIO()
        image.save(buf, format="PNG")
        payload = buf.getvalue()
        with self._images_lock:
            self._images[image_id] = payload
            self._images.move_to_end(image_id)
            while len(self._images) > self._max_items:
                self._images.popitem(last=False)
        return image_id

    def get_image(self, image_id: str) -> Optional[bytes]:
        with self._images_lock:
            payload = self._images.get(image_id)
            if payload is not None:
                self._images.move_to_end(image_id)
            return payload

