import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights


def get_object_detection_model(num_classes, model_name="fasterrcnn_resnet50"):
    if model_name == "fasterrcnn_resnet50":
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

