import os.path
from pathlib import Path

from typing import List
import copy
import numpy as np
import torch
import miso.object_detection.engine.utils as utils
import miso.object_detection.engine.transforms as T
from miso.object_detection.dataset.annotation import RectangleAnnotation
from miso.object_detection.dataset.dataset import ObjectDetectionDataset
from miso.object_detection.dataset.image import ImageMetadata
from miso.object_detection.dataset.project import Project


def infer(project: Project,
          model_path: str,
          model_labels: List[str] = None,
          threshold: float = 0.5,
          batch_size=2,
          nv: bool = False):
    if nv:
        model_labels = [label + "_NV" for label in model_labels]
    # Ensure labels
    for label in model_labels:
        project.add_label(None, label, None)

    # Load model
    model = torch.load(model_path)
    model.cuda()
    model.eval()

    # Create dataset
    project = copy.deepcopy(project)
    project.remove_labelled_images()
    dataset = ObjectDetectionDataset(project, T.Compose([T.ToTensor()]))

    # Get data loader
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4,
                                              collate_fn=utils.collate_fn)

    # New project
    project = Project()

    idx = 0
    with torch.inference_mode():
        for images, targets, metadata in data_loader:
            images_cuda = list(image.cuda() for image in images)
            results = model(images_cuda)
            for metadata, result in zip(metadata, results):
                boxes = result['boxes'][result['scores'] > threshold].cpu().numpy()
                labels = result['labels'][result['scores'] > threshold].cpu().numpy()
                for box, label in zip(boxes, labels):
                    ann = RectangleAnnotation(box[0],
                                              box[1],
                                              box[2] - box[0],
                                              box[3] - box[1],
                                              model_labels[label - 1])
                    metadata.boxes.append(ann)
                idx += 1
                project.add_image(metadata)
    return project


def infer_directory(input_dir: str,
                    model_path: str,
                    model_labels: List[str] = None,
                    threshold: float = 0.5,
                    batch_size=2):

    # Filenames
    p = Path(input_dir)
    if not p.exists():
        raise ValueError(f"Directory does not exist: {input_dir}")
    paths = p.rglob("*.*")
    filepaths = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".jpg" or suffix == ".jpeg" or suffix == ".png" or suffix == ".bmp" or suffix == ".tiff" or suffix == ".tif":
            filepaths.append(path)

    # Create project
    project = Project()
    for i, filepath in enumerate(filepaths):
        project.add_image(ImageMetadata(filepath, "/", 0, i))

    # Ensure labels
    for label in model_labels:
        project.add_label(None, label, None)

    # Load model
    model = torch.load(model_path)
    model.cuda()
    model.eval()

    # Create dataset
    project = copy.deepcopy(project)
    project.remove_labelled_images()
    dataset = ObjectDetectionDataset(project, T.Compose([T.ToTensor()]))

    # Get data loader
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4,
                                              collate_fn=utils.collate_fn)

    # New project
    project = Project()

    idx = 0
    with torch.inference_mode():
        for images, targets, metadata in data_loader:
            images_cuda = list(image.cuda() for image in images)
            results = model(images_cuda)
            for metadata, result in zip(metadata, results):
                boxes = result['boxes'][result['scores'] > threshold].cpu().numpy()
                labels = result['labels'][result['scores'] > threshold].cpu().numpy()
                for box, label in zip(boxes, labels):
                    ann = RectangleAnnotation(box[0],
                                              box[1],
                                              box[2] - box[0],
                                              box[3] - box[1],
                                              model_labels[label - 1])
                    metadata.boxes.append(ann)
                idx += 1
                project.add_image(metadata)
    return project
