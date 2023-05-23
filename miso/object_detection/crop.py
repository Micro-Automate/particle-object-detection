from pathlib import Path
import os
import skimage.io as skio
from tqdm import tqdm
from miso.object_detection.dataset.project import Project
from miso.shared.utils import now_as_str


def crop_objects(project: Project, output_dir: str, relative_to=None):
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir)

    for image in tqdm(project.image_dict.values()):
        if len(image.boxes) == 0:
            continue
        im = skio.imread(image.full_path)
        for box in image.boxes:
            if relative_to is not None:
                label_path = output_path / Path(image.full_path).relative_to(relative_to).parent / box.label
            elif len(project.task_names) > 0:
                label_path = output_path / f"{image.dataset_id} - {project.task_names[image.dataset_id]}" / box.label
                # label_dir = os.path.join(output_dir, f"{image.dataset_id} - {project.task_names[image.dataset_id]}", box.label)
            else:
                label_path = output_path / box.label
                # label_dir = os.path.join(output_dir, box.label)
            label_path.mkdir(parents=True, exist_ok=True)
            # os.makedirs(label_dir, exist_ok=True)
            c = box.coords_int
            s = box.bounds
            crop = im[c[1]:c[3], c[0]:c[2], ...]
            path = Path(image.full_path)
            filename = f"{path.stem}_{s[0]:.0f}_{s[1]:.0f}_{s[2]:.0f}_{s[3]:.0f}{path.suffix}"
            skio.imsave(os.path.join(str(label_path), filename), crop, check_contrast=False)

