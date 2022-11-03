from miso.object_detection.dataset.cvat.cvat_web_api import CvatTask
from miso.object_detection.training import train

task = CvatTask("http://cvat:8080", 1, debug=True)
# task = CvatTask("http://localhost:8080", 1, image_root="/mnt/docker/version-pack-data/community/docker/volumes/cvat_cvat_data/_data/data", debug=True)
task.load()

train(task.project, ["Coccolith"])

# "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth" to /home/user/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth