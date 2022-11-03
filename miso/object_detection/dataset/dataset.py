import numpy as np
import torch
import torch.utils.data
from PIL import Image

from miso.object_detection.dataset.project import Project


class ObjectDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, project: Project, transforms):
        self.project = project
        self.images = list(project.image_dict.values())
        self.cls_labels = project.label_names
        self.transforms = transforms

    def __getitem__(self, idx):
        cvat_image = self.images[idx]
        img = Image.open(cvat_image.full_path).convert("RGB")

        boxes = np.asarray([box.coords for box in cvat_image.boxes])
        labels = np.asarray([self.cls_labels.index(box.label) + 1 for box in cvat_image.boxes])
        # print(labels)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])

        # Bounding box areas
        if len(boxes) > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = []

        # All instances are not crowd
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, cvat_image

    def __len__(self):
        return len(self.images)
