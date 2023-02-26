import os.path
from pathlib import Path

import click

from miso.object_detection.dataset.cvat.cvat_web_api import CvatTask
from miso.object_detection.dataset.project import Project
from miso.object_detection.inference import infer
from miso.object_detection.inference import infer_directory as infer_directory_fn
from miso.object_detection.training import train
from miso.object_detection.crop import crop_objects as crop_objects_fn
from miso.shared.utils import now_as_str


@click.group()
def cli():
    pass


@cli.command()
@click.option('-t',
              '--tasks',
              type=str,
              prompt='List of task ids to train on',
              help='List of task ids to train on separated by commas')
@click.option('-l',
              '--labels',
              type=str,
              default=None,
              help='List of label names to train on separated by commas')
@click.option('--batch-size',
              type=int,
              default=2,
              show_default=True,
              help='Batch size for training (reduce if getting out-of-memory errors')
@click.option("--wsl2",
              is_flag=True,
              default=False,
              help="Running this on a windows machine using WSL2 instead of docker")
@click.option('--api',
              type=str,
              default="v1",
              show_default=True,
              help='CVAT api version string, v1 or v2')
@click.option('-o',
              '--output-dir',
              type=str,
              default="/obj_det/models",
              show_default=True,
              help='Model output directory')
@click.option('-m',
              '--model',
              type=str,
              default=None,
              help='Model name')
@click.option('-d',
              '--data',
              type=str,
              default="/data",
              show_default=True,
              help='Data directory')
@click.option('--max-epochs',
              type=int,
              default="100000",
              show_default=True,
              help='Maximum number of epochs')
def train_object_detector(tasks: str,
                          labels: str,
                          batch_size: int,
                          wsl2: bool,
                          api: str,
                          output_dir: str,
                          model: str,
                          data: str,
                          max_epochs):
    tasks = [int(task.strip()) for task in tasks.split(",")]
    if labels is not None:
        labels = [label.strip() for label in labels.split(",")]

    project = Project()
    for task in tasks:
        task = CvatTask("http://cvat:8080",
                        task,
                        is_wsl2=wsl2,
                        api=api,
                        debug=True)
        task.load()
        project.add_project(task.project)

    # TODO train test split
    train(project,
          labels,
          output_dir=output_dir,
          name=model,
          batch_size=batch_size,
          max_epochs=max_epochs)


@cli.command()
@click.option('--tasks', type=str,
              prompt='List of task ids to infer on',
              help='List of task ids to infer on')
@click.option('--model-dir',
              type=str,
              default="/obj_det/models",
              show_default=True,
              help='Directory containing models')
@click.option('--model', type=str,
              prompt='Name of folder containing model',
              help='Name of folder containing model')
@click.option('--threshold', type=float, default=0.5,
              help='Detection threshold')
@click.option('--batch-size', type=int, default=2,
              help='Batch size for training (reduce if getting out-of-memory errors')
@click.option("--nv",
              is_flag=True,
              default=False,
              help="Append NV to the detected labels")
@click.option("--wsl2",
              is_flag=True,
              default=False,
              help="Running this on a windows machine using WSL2 instead of docker")
@click.option('--api',
              type=str,
              default="v1",
              show_default=True,
              help='CVAT api version string, v1 or v2')
def infer_object_detector(tasks, model_dir, model, threshold, batch_size, nv, wsl2, api):
    tasks = [int(task) for task in tasks.split(",")]
    model_path = os.path.join(model_dir, model, "model.pt")
    labels_path = os.path.join(model_dir, model, "labels.txt")

    labels = []
    with open(labels_path) as fp:
        for line in fp.readlines():
            parts = line.split(",")
            if len(parts) > 0:
                labels.append(parts[1].strip())

    for task in tasks:
        task = CvatTask("http://cvat:8080",
                        task,
                        is_wsl2=wsl2,
                        api=api,
                        debug=True)
        task.load()
        project = infer(task.project,
                        model_path,
                        labels,
                        threshold,
                        batch_size,
                        nv)
        project.summary()
        task.add_shapes(project)


@cli.command()
@click.option('--tasks', type=str,
              prompt='List of task ids to crop from',
              help='List of task ids to crop from')
@click.option('-o',
              '--output-dir',
              type=str,
              default="/obj_det/crops",
              show_default=True,
              help='Crop output directory')
@click.option("--wsl2",
              is_flag=True,
              default=False,
              help="Running this on a windows machine using WSL2 instead of docker")
@click.option('--api',
              type=str,
              default="v1",
              show_default=True,
              help='CVAT api version string, v1 or v2')
def crop_objects(tasks, output_dir, wsl2, api):
    tasks = [int(task) for task in tasks.split(",")]
    output_dir = os.path.join(output_dir, now_as_str() + "_" + "_".join([str(task) for task in tasks]))
    for task in tasks:
        task = CvatTask("http://cvat:8080",
                        task,
                        is_wsl2=wsl2,
                        api=api,
                        debug=True)
        task.load()
        crop_objects_fn(task.project, output_dir)


@cli.command()
@click.option('-i', '--input-dir', type=str,
              prompt='Name of folder containing images to infer on',
              help='Name of folder containing images to infer on')
@click.option('--model-dir',
              type=str,
              default="/obj_det/models",
              show_default=True,
              help='Directory containing models')
@click.option('--model', type=str,
              prompt='Name of folder containing model',
              help='Name of folder containing model')
@click.option('--threshold', type=float, default=0.5,
              help='Detection threshold')
@click.option('--batch-size', type=int, default=2,
              help='Batch size for training (reduce if getting out-of-memory errors')
def infer_object_detector_directory(input_dir, model_dir, model, threshold, batch_size):
    model_path = os.path.join(model_dir, model, "model.pt")
    labels_path = os.path.join(model_dir, model, "labels.txt")
    labels = []
    with open(labels_path) as fp:
        for line in fp.readlines():
            parts = line.split(",")
            if len(parts) > 0:
                labels.append(parts[1].strip())

    project = infer_directory_fn(input_dir, model_path, labels, threshold, batch_size)

    if crop:
        crops_dir = Path(input_dir).joinpath("crops")
        crops_dir.mkdir(parents=True, exist_ok=True)
        crop_objects_fn(project, str(crops_dir))


if __name__ == "__main__":
    cli()
