import os
from pathlib import Path

import click

from miso.object_detection.dataset.cvat.cvat_web_api import CvatTask
from miso.object_detection.dataset.project import Project
from miso.object_detection.inference import infer
from miso.object_detection.inference import infer_directory as infer_directory_fn
from miso.object_detection.training import train
from miso.object_detection.crop import crop_objects as crop_objects_fn
from miso.shared.utils import now_as_str

# env variable HOST_HOSTNAME must be extracted when container 
# is launched using "docker run" with "-e HOST_HOSTNAME=$(hostname -f)""
hostname = os.getenv('HOST_HOSTNAME')
port = '8080'

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
@click.option('--merge-label',
              type=str,
              default=None,
              help='Merge the labels into a single label')
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
@click.option('--alrs-epochs',
              type=int,
              default="10",
              show_default=True,
              help='Number of epochs to monitor for alrs')
@click.option('--optimiser',
              type=str,
              default="sgd",
              show_default=True,
              help='Optimiser to use')
@click.option('--hostname',
              type=str,
              default=hostname,
              show_default=True,
              help='Host hostname')
@click.option('--port',
              type=str,
              default=port,
              show_default=True,
              help='port number')
def train_object_detector(tasks: str,
                          labels: str,
                          merge_label: str,
                          batch_size: int,
                          wsl2: bool,
                          api: str,
                          output_dir: str,
                          model: str,
                          data: str,
                          max_epochs,
                          alrs_epochs,
                          optimiser,
                          hostname,
                          port):
    # Tasks and labels
    tasks = [int(task.strip()) for task in tasks.split(",")]
    if labels is not None:
        labels = [label.strip() for label in labels.split(",")]

    # Create project from tasks
    project = Project()
    for task in tasks:
        task = CvatTask(f"http://{hostname}:{port}",
                        task,
                        is_wsl2=wsl2,
                        api=api,
                        debug=True)
        task.load()
        project.add_project(task.project)

    # Merge labels if desired
    if merge_label is not None:
        for label in project.label_dict.values():
            if label.name in labels:
                project.rename_label(label.name, merge_label)
        project.update_label_dict()
        labels = [merge_label]

    # Train model
    # TODO train test split
    train(project,
          labels,
          output_dir=output_dir,
          name=model,
          batch_size=batch_size,
          max_epochs=max_epochs,
          alrs_epochs=alrs_epochs,
          optimiser=optimiser)


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
@click.option('--hostname',
              type=str,
              default=hostname,
              show_default=True,
              help='Host hostname')
@click.option('--port',
              type=str,
              default=port,
              show_default=True,
              help='port number')
def infer_object_detector(tasks, model_dir, model, threshold, batch_size, nv, wsl2, api, hostname, port):
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
        task = CvatTask(f"http://{hostname}:{port}",
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
        task = CvatTask(f"http://{hostname}:{port}",
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
@click.option('-o', '--output-dir', type=str,
              prompt='Name of folder to store results',
              help='Name of folder to store results')
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
def infer_object_detector_directory(input_dir, output_dir, model_dir, model, threshold, batch_size):
    model_path = os.path.join(model_dir, model, "model.pt")
    labels_path = os.path.join(model_dir, model, "labels.txt")
    labels = []
    with open(labels_path) as fp:
        for line in fp.readlines():
            parts = line.split(",")
            if len(parts) > 0:
                labels.append(parts[1].strip())

    project = infer_directory_fn(input_dir, model_path, labels, threshold, batch_size)

    # crops_dir = Path(input_dir).joinpath("crops")
    # crops_dir.mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    crop_objects_fn(project, output_dir, relative_to=input_dir)


if __name__ == "__main__":
    cli()
