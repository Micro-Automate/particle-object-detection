from miso.object_detection.dataset.cvat.cvat_web_api import CvatTask
from miso.object_detection.training import train
from miso.object_detection.inference import infer


def test_training():
    task = CvatTask("http://localhost:8080", 1, is_wsl2=True, debug=True, api='v2')
    task.load()
    task.project.summary()

    train(task.project,
          ["Coccolith"], #, "Coccosphere", "Foraminifera"],
          "../training",
          optimiser='sgd',
          max_epochs=1)

def test_inference():
    task = CvatTask("http://localhost:8080", 3, is_wsl2=True, debug=True)
    task.load()
    project = infer(task.project, "../training/2022-08-16_091015/model.pt", ["Coccolith", "Coccosphere", "Foraminifera"], 0.5)
    project.summary()
    task.add_shapes(project)


if __name__ == "__main__":
    test_training()
    # test_inference()