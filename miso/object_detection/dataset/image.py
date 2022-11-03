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

    # def rescale_annotation(self, factor):
    #     for ann in self.annotations:
    #         ann.x = ann.x * factor
    #         ann.y = ann.y * factor
    #         ann.width = ann.width * factor
    #         ann.height = ann.height * factor

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

    # def time_from_filename(self, mode=0):
    #     fn = os.path.basename(self.filename)
    #     if mode == 0:
    #         fstr = '%Y%m%dT%H%M%S%fZ'
    #         dt = fn[-len(fstr)-4:-4]
    #         return datetime.strptime(dt, '%Y%m%dT%H%M%S%fZ')
    #     if mode == 1:
    #         fstr = '%Y%m%d_%H%M%S%f'
    #         dt = fn[-12-18:-12]
    #         return datetime.strptime(dt, fstr)
    #
    # def to_dict(self):
    #     d = OrderedDict()
    #     d['filename'] = self.filename
    #     d['filesize'] = self.filesize
    #     d['metadata'] = OrderedDict()
    #     if self.metadata is not None:
    #         d['metadata'] = self.metadata
    #         for k, v in d['metadata'].items():
    #             if isinstance(v, Timestamp):
    #                 d['metadata'][k] = None
    #     d['annotations'] = []
    #     for ann in self.annotations:
    #         d['annotations'].append(ann.to_dict())
    #     return d
    #
    # @classmethod
    # def from_dict(cls, d):
    #     c = cls(d['filename'], d['filesize'], d['metadata'])
    #     for ann in d['annotations']:
    #         c.add_annotation(RectangleAnnotation.from_dict(ann))
    #     return c
    #
    # def to_json(self):
    #     return json.dumps(self.to_dict(), indent=4)
    #
    # @classmethod
    # def from_json(cls, json_str):
    #     j = json.loads(json_str, object_pairs_hook=OrderedDict)
    #     return cls.from_dict(j)
    #
    # def to_via(self):
    #     """
    #     Convert to VIA format
    #     """
    #     metadata = OrderedDict()
    #     metadata['filename'] = self.filename
    #     metadata['size'] = self.filesize
    #     metadata['regions'] = list()
    #     # metadata['file_attributes'] = self.metadata
    #     for annotation in self.annotations:
    #         metadata['regions'].append(annotation.to_via())
    #     return metadata
    #
    # @classmethod
    # def from_via(cls, metadata):
    #     image_metadata = cls(metadata['filename'],
    #                          metadata['size'])
    #                          # metadata['file_attributes'])
    #     for region in metadata['regions']:
    #         image_metadata.add_annotation(RectangleAnnotation.from_via(region))
    #     return image_metadata
    #
    # def to_cvat(self):
    #     attributes = {"id": str(self.metadata["id"]),
    #                   "name": self.filename,
    #                   "width": str(self.metadata["width"]),
    #                   "height": str(self.metadata["height"])}
    #     el = etree.Element("image", attrib=attributes)
    #     return el
    #
    # @staticmethod
    # def from_cvat(el: etree):
    #     filename = el.get("name")
    #     im = UmamiImage(filename, 0)
    #     im.metadata["id"] = el.get("id")
    #     im.metadata["width"] = el.get("width")
    #     im.metadata["height"] = el.get("height")
    #     return im


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
