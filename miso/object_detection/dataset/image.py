import os.path

import json
from pathlib import Path
from typing import List, Union
from miso.object_detection.dataset.annotation import RectangleAnnotation


class ImageMetadata(object):
    def __init__(self,
                 path,
                 container,
                 dataset_id=0,
                 frame_id=0,
                 metadata=None):
        # Path to image within the container
        self.path = path
        # Container root directory
        self.container = container
        # Id of dataset this image belongs to
        self.dataset_id = dataset_id
        # Index of image in the dataset
        self.frame_id = frame_id
        # Annotations
        # - boxes
        self.boxes: List[RectangleAnnotation] = []

        self.metadata = metadata
        if self.metadata is None:
            self.metadata = dict()

    def has_label(self, label: Union[str, list]):
        if isinstance(label, str):
            for box in self.boxes:
                if box.label == label:
                    return True
        elif isinstance(label, list):
            for box in self.boxes:
                if box.label in label:
                    return True
        return False

    @property
    def id(self):
        return ImageMetadata.create_id(self.path, self.dataset_id, self.frame_id)

    @property
    def full_path(self):
        return os.path.join(self.container, self.path)

    @property
    def labels(self):
        labels = []
        for box in self.boxes:
            labels.append(box.label)
        label_set = set(labels)
        return list(label_set)

    @staticmethod
    def create_id(path, dataset_id=0, frame_id=0):
        return f"{dataset_id}_{frame_id}_{path}"


if __name__ == "__main__":
    image_metadata = ImageMetadata("test_filename.jpg", 0)
    rect1 = RectangleAnnotation(100, 200, 400, 600, "test", 1.0, "ross", "greg", 1, 101, 2, 45)
    rect2 = RectangleAnnotation(100, 200, 400, 600, "test", 1.0, "ross", "greg", 1, 101, 2, 45)
    image_metadata.add_annotation(rect1)
    image_metadata.add_annotation(rect2)
    image_metadata.metadata['lat'] = 19.0123
    image_metadata.metadata['lon'] = 145.01234
    print(ImageMetadata.from_dict(image_metadata.to_dict()).to_json())
    print(ImageMetadata.from_json(image_metadata.to_json()).to_json())
    print(json.dumps(image_metadata.to_via(), indent=4))
