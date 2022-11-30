import copy
import os
from datetime import datetime
from typing import List

import torch.onnx
import miso.object_detection.engine.utils as utils
from miso.object_detection.dataset.dataset import ObjectDetectionDataset
from miso.object_detection.dataset.project import Project
from miso.object_detection.engine.engine import train_one_epoch, evaluate
from miso.object_detection.models import get_object_detection_model
from miso.object_detection.transforms import get_transforms
from miso.shared.learning_rate_scheduler import AdaptiveLearningRateScheduler


def train(project: Project,
          labels: List[str],
          output_dir: str = None,
          name: str = None,
          batch_size=2,
          alrs_epochs=10,
          alrs_drops=4,
          alrs_startup_factor=2,
          optimiser='sgd',
          max_epochs=500):
    # Fix project
    project = copy.deepcopy(project)
    if labels is not None:
        project.keep_annotations_with_label(labels)
    project.remove_unlabelled_images()
    project.update_label_dict()
    labels = project.label_names

    print()
    print("=" * 80)

    # Directory to save in
    if output_dir is None:
        output_dir = os.getcwd()
    if name is None:
        name = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = os.path.join(output_dir, name)

    print("Object engine training")
    print(f"- output directory: {output_dir}")
    project.summary()

    # Get datasets
    dataset_train = ObjectDetectionDataset(project, get_transforms(train=True))
    dataset_test = ObjectDetectionDataset(project, get_transforms(train=False))

    # Split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset_train)).tolist()
    fraction = int(0.2 * len(dataset_train))
    dataset_train = torch.utils.data.Subset(dataset_train, indices[:-fraction])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-fraction:])

    print("Training set images")
    print(f"- total: {len(indices)}")
    print(f"- train: {len(dataset_train)}")
    print(f"- test:  {len(dataset_test)}")

    sharing_strategy = "file_system"
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)

    def set_worker_sharing_strategy(worker_id: int) -> None:
        torch.multiprocessing.set_sharing_strategy(sharing_strategy)

    # Define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(dataset_train,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=4,
                                                    collate_fn=utils.collate_fn,
                                                    worker_init_fn=set_worker_sharing_strategy)

    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=4,
                                                   collate_fn=utils.collate_fn,
                                                   worker_init_fn=set_worker_sharing_strategy)

    # Device to train on
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Training device is: {device}")

    # Get the model and move to correct device
    num_classes = len(labels) + 1
    print(f"Number of classes: {num_classes}")
    model = get_object_detection_model(num_classes)
    model.to(device)

    # Construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if optimiser == 'sgd':
        opt = torch.optim.SGD(params,
                              lr=0.005,
                              momentum=0.9,
                              weight_decay=0.0005)
    elif optimiser == 'adam':
        opt = torch.optim.Adam(params,
                               lr=0.001)
    else:
        raise ValueError("Optimiser must be one of 'sgd' or 'adam'")
    print(f"Optimiser: {optimiser}")
    # Learning rate scheduler
    lr_scheduler = AdaptiveLearningRateScheduler(opt,
                                                 factor=0.5,
                                                 nb_drops=alrs_drops,
                                                 nb_epochs=alrs_epochs,
                                                 startup_delay_factor=alrs_startup_factor)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=10,
    #                                                gamma=0.5)

    # Train
    print("=" * 80)
    for epoch in range(max_epochs):
        # Train for one epoch, printing every 10 iterations
        metrics = train_one_epoch(model, opt, data_loader_train, device, epoch, print_freq=10)
        # Evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        # Update the learning rate
        if lr_scheduler.step(epoch, metrics.loss.global_avg):
            break

    print("-" * 80)
    print(f"Training finished, {epoch + 1} epochs")
    _, stats = evaluate(model, data_loader_test, device=device)
    print("=" * 80)

    # Save the model in torch format
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model, os.path.join(output_dir, "model.pt"))

    # Save the labels
    with open(os.path.join(output_dir, "labels.txt"), 'w') as fp:
        for idx, label in enumerate(labels):
            fp.write(f"{idx + 1},{label}\n")

    # Save pycocotools stats
    stat_names = [
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]",
        "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]",
        "Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]",
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]",
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]"
    ]
    with open(os.path.join(output_dir, "results.txt"), 'w') as fp:
        for i, stat in enumerate(stats[0]):
            fp.write(f"{stat_names[i]} = {stat:.3f}\n")
