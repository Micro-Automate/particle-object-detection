import os.path

from typing import List
import copy
import numpy as np
import torch
import miso.object_detection.engine.utils as utils
import miso.object_detection.engine.transforms as T
from miso.object_detection.dataset.annotation import RectangleAnnotation
from miso.object_detection.dataset.dataset import ObjectDetectionDataset
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
                                              model_labels[label-1])
                    metadata.boxes.append(ann)
                idx += 1
                project.add_image(metadata)
    return project


                # cv2.imshow("im2", np.array(trans.ToPILImage()(display_image))[:,:, ::-1])
                # cv2.waitKey()

            # cv2.imshow("im2", np.array(trans.ToPILImage()(display_image2)))
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            # def infer(project: Project, model_path: str, threshold: float = 0.5):
            #     # Load model
            #     model = torch.load(model_path)
            #     model.cuda()
            #     model.eval()
            #
            #     # Create dataset
            #     dataset = get_dataset(project)
            #     data_loader = torch.utils.data.DataLoader(dataset,
            #                                               batch_size=2,
            #                                               shuffle=False,
            #                                               num_workers=4,
            #                                               collate_fn=utils.collate_fn)
            #
            #     # os.makedirs("output", exist_ok=True)
            #     # os.makedirs("output/images", exist_ok=True)
            #     # os.makedirs("output/crops", exist_ok=True)
            #     # cv2.destroyAllWindows()
            #     idx = 0
            #     with torch.inference_mode():
            #         for images, targets, metadata in data_loader:
            #             images_cuda = list(image.cuda() for image in images)
            #             results = model(images_cuda)
            #             for image, target, metadata, result in zip(images, targets, metadata, results):
            #
            #                 name = os.path.splitext(target["image_name"])[0]
            #                 print(name)
            #
            #                 display_image = (image * 255).to(torch.uint8)
            #
            #                 if len(target['boxes']) == 0:
            #                     boxes = result['boxes'][result['scores'] > 0.5].cpu()
            #                 else:
            #                     boxes = target['boxes'].cpu()
            #                 npim = np.array(trans.ToPILImage()(display_image))
            #
            #                 for box_idx, box in enumerate(boxes.numpy()):
            #                     box = np.round(box).astype(np.int32)
            #                     crop = npim[box[1]:box[3], box[0]:box[2], ...]
            #                     cv2.imwrite(os.path.join("output", "crops", f"{name}_{box_idx:04d}.png"), crop)
            #
            #                 display_image = draw_bounding_boxes(display_image,
            #                                                     target['boxes'],
            #                                                     colors="#aaaaaa",
            #                                                     width=6)
            #                 display_image = draw_bounding_boxes(display_image,
            #                                                     result['boxes'][result['scores'] > 0.5].cpu(),
            #                                                     colors="green",
            #                                                     width=4)
            #                 npim = np.array(trans.ToPILImage()(display_image))
            #                 cv2.imwrite(os.path.join("output", "images", f"{name}.png"), npim)
            #
            #                 idx += 1
            #
            #                 # cv2.imshow("im2", np.array(trans.ToPILImage()(display_image))[:,:, ::-1])
            #                 # cv2.waitKey()
            #
            #             # cv2.imshow("im2", np.array(trans.ToPILImage()(display_image2)))
            #             # cv2.waitKey()
            #             # cv2.destroyAllWindows()