from pathlib import Path
import os
import skimage.io as skio
from tqdm import tqdm
from miso.object_detection.dataset.project import Project
from miso.shared.utils import now_as_str


def crop_objects(project: Project, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    for image in tqdm(project.image_dict.values()):
        if len(image.boxes) == 0:
            continue
        im = skio.imread(image.full_path)
        for box in image.boxes:
            label_dir = os.path.join(output_dir, f"{image.dataset_id} - {project.task_names[image.dataset_id]}", box.label)
            os.makedirs(label_dir, exist_ok=True)
            c = box.coords_int
            s = box.bounds
            crop = im[c[1]:c[3], c[0]:c[2], ...]
            path = Path(image.full_path)
            filename = f"{path.stem}_{s[0]:.0f}_{s[1]:.0f}_{s[2]:.0f}_{s[3]:.0f}{path.suffix}"
            skio.imsave(os.path.join(label_dir, filename), crop, check_contrast=False)

