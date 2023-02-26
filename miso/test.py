import os
from pathlib import Path

from miso.object_detection.dataset.cvat.cvat_web_api import CvatTask
from miso.object_detection.training import train
from miso.object_detection.inference import infer



def test_training():
    task = CvatTask("http://localhost:8080", 76, is_wsl2=True, debug=True, api='v2')
    task.load()
    task.project.summary()

    # train(task.project,
    #       ["Coccolith"], #, "Coccosphere", "Foraminifera"],
    #       "../training",
    #       optimiser='sgd',
    #       max_epochs=1)


def test_inference():
    task = CvatTask("http://localhost:8080", 99, is_wsl2=True, debug=True, api='v2')
    task.load()

    project = infer(task.project, "../obj_det/models/Coccolith/model.pt", ["Coccolith", "Coccosphere", "Foraminifera"], 0.5, nv=False)
    project.summary()
    task.add_shapes(project)


from miso.object_detection.inference import infer_directory as infer_directory_fn
from miso.object_detection.crop import crop_objects as crop_objects_fn

def infer_directory(input_dir, model_dir, model, threshold, batch_size, crop):
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


def test_inference_directory():
    project = infer_directory(r"/mnt/c/Users/ross.marchant/data/rapp_test",
                              "../obj_det/models/",
                              "Coccolith",
                              0.5,
                              4,
                              True)


if __name__ == "__main__":
    # test_training()
    # test_inference()
    test_inference_directory()