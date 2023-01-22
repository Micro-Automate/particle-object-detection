from typing import Dict, Union

from miso.object_detection.dataset.image import ImageMetadata
from miso.object_detection.dataset.label import Label


class Project(object):
    def __init__(self):
        self.filename = ""
        self.task_names: Dict[int, str] = dict()
        self.image_dict: Dict[str, ImageMetadata] = dict()
        self.label_dict: Dict[str, Label] = dict()

    """
    Labels
    """
    @property
    def label_names(self):
        return [label.name for label in self.label_dict.values()]

    def add_label(self, id_, name, colour):
        if name not in self.label_dict:
            self.label_dict[name] = Label(id_, name, colour)

    def keep_annotations_with_label(self, labels: Union[str, list]):
        if isinstance(labels, str):
            labels = [labels]
        for image in self.image_dict.values():
            image.boxes = [ann for ann in image.boxes if ann.label in labels]

    def remove_annotations_with_label(self, labels: Union[str, list]):
        if isinstance(labels, str):
            labels = [labels]
        for image in self.image_dict.values():
            image.boxes = [ann for ann in image.boxes if ann.label not in labels]

    def remove_annotations_below_threshold(self, threshold: float):
        for image in self.image_dict.values():
            image.boxes = [ann for ann in image.boxes if ann.score < threshold]

    def remove_unlabelled_images(self):
        self.image_dict = {k: v for k, v in self.image_dict.items() if len(v.boxes) > 0}

    def remove_labelled_images(self):
        self.image_dict = {k: v for k, v in self.image_dict.items() if len(v.boxes) == 0}

    def label_counts(self):
        counts = {k: 0 for k, v in self.label_dict.items()}
        for image in self.image_dict.values():
            for box in image.boxes:
                counts[box.label] += 1
        return counts

    def labels_in_use(self):
        labels = {}
        for image in self.image_dict.values():
            for box in image.boxes:
                if box.label in self.label_dict:
                    labels[box.label] = self.label_dict[box.label]
                elif box.label not in labels:
                    labels[box.label] = Label(None, box.label, None)
        return labels

    def update_label_dict(self):
        self.label_dict = self.labels_in_use()

    """
    Images
    """
    def add_image(self, image: ImageMetadata):
        self.image_dict[image.id] = image
        for box in image.boxes:
            if box.label not in self.label_dict:
                self.label_dict[box.label] = Label(None, box.label, None)

    def add_project(self, project: "Project"):
        for key, image in project.image_dict.items():
            if key in self.image_dict:
                raise KeyError("Image from project already exists in this project")
            else:
                self.image_dict[key] = image

        for key, label in project.label_dict.items():
            if key not in self.label_dict:
                self.label_dict[key] = label

    def box_counts(self):
        counts = {"0": 0,
                  "1-10": 0,
                  "11-100": 0,
                  "100+": 0}
        for image in self.image_dict.values():
            boxes = len(image.boxes)
            if boxes == 0:
                counts["0"] += 1
            elif boxes <= 10:
                counts["1-10"] += 1
            elif boxes <= 100:
                counts["11-100"] += 1
            else:
                counts["100+"] += 1
        return counts

    def summary(self):
        print("-" * 80)
        print("Project summary")
        print("Labels:")
        counts = self.label_counts()
        for label in self.label_dict.values():
            print(f"- {label.name} - id: {label.id}, colour: {label.colour}, count: {counts[label.name]}")
        print(f"- total boxes: {sum(counts.values())}")
        print("Image:")
        for rng, count in self.box_counts().items():
            print(f"- {rng}: {count}")
        print(f"- total images: {len(self.image_dict)}")
        print("-" * 80)
        print()



