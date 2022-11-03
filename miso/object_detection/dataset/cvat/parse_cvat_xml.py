import os
from glob import glob
from typing import List
import numpy as np
import xmltodict


class CvatProject():
    def __init__(self):
        self.cls_labels = []
        self.images: List[CvatImage] = []
        self.xml_files = []

    def parse_api(self, task_id, image_paths=["/mnt/wsl/docker-desktop-data/version-pack-data/community/docker/volumes/cvat_cvat_data/_data/data"]):
        pass

    def parse_xml(self, path):
        multiple_projects = False
        xml_files = []
        # Path is a folder
        if os.path.isdir(path):
            xml_file = os.path.join(path, "annotations.xml")
            if os.path.exists(xml_file):
                xml_files.append(xml_file)
            else:
                folders = sorted(glob(os.path.join(path, "*")))
                print(folders)
                for folder in folders:
                    xml_file = os.path.join(folder, "annotations.xml")
                    if os.path.exists(xml_file):
                        xml_files.append(xml_file)
        elif path.endswith("annotations.xml"):
            xml_files.append(path)
        if len(xml_files) == 0:
            raise ValueError("No annotations.xml files found!")
        self.xml_files = xml_files

        labels = set()
        images = []
        for xml_file in xml_files:
            labels.update(CvatProject._parse_labels(xml_file))
            images.extend(CvatProject._parse_images(xml_file))

        self.cls_labels = sorted(list(labels))
        self.images = images

    @staticmethod
    def _parse_labels(xml_file):
        labels = []
        with open(xml_file, "r") as fp:
            xml = fp.read()
            obj = xmltodict.parse(xml)

            # Get labels
            for label in obj['annotations']['meta']['task']['labels']['label']:
                labels.append(label['name'])
        return labels

    @staticmethod
    def _parse_images(xml_file):
        folder = os.path.join(os.path.dirname(xml_file), "images")
        with open(xml_file, "r") as fp:
            xml = fp.read()
            obj = xmltodict.parse(xml)
            images = []
            for image in obj['annotations']['image']:
                cvat_image = CvatImage(image['@name'], folder)
                if "box" in image:
                    for box in image['box']:
                        cvat_image.boxes.append(CvatBox(box['@xtl'], box['@ytl'], box['@xbr'], box['@ybr'], box['@label']))
                images.append(cvat_image)
        return images

    def remove_unused_labels(self):
        labels = set()
        for image in self.images:
            labels.update({box.label for box in image.boxes})
        self.cls_labels = sorted(list(labels))

    def remove_unlabelled_images(self):
        images = []
        for image in self.images:
            if len(image.boxes) > 0:
                images.append(image)
        self.images = images

    def keep_labels(self, labels):
        print(labels)
        self.cls_labels = sorted(list(set(self.cls_labels).intersection(set(labels))))
        for image in self.images:
            boxes = []
            for box in image.boxes:
                if box.label in self.cls_labels:
                    boxes.append(box)
            image.boxes = boxes

    def count_labels(self):
        count = {k: 0 for k in self.cls_labels}
        for image in self.images:
            for box in image.boxes:
                count[box.label] += 1
        return count

    def summary(self, verbosity=0):
        print()
        print("=" * 80)
        print(f"Cvat Project")
        for xml_file in self.xml_files:
            print(f"- {xml_file}")
        print("-" * 80)
        print("- labels:")
        counts = self.count_labels()
        for cls_label in self.cls_labels:
            print(f"  - {cls_label} ({counts[cls_label]})")
        print(f"- images: {len(self.images)}")
        if verbosity > 0:
            for image in self.images:
                print(f"  - {image.path}")
                if verbosity > 1:
                    for box in image.boxes:
                        print(f"    - {box.label}: {box.x0} {box.x1} {box.x1} {box.y1}")
        print("=" * 80)

class CvatBox():
    def __init__(self, x0, y0, x1, y1, label):
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.label = label

    def to_array(self):
        return np.asarray([self.x0, self.y0, self.x1, self.y1])


class CvatImage():
    def __init__(self, name, folder):
        self.folder = folder
        self.name = name
        self.path = os.path.join(self.folder, self.name)
        self.boxes: List[CvatBox] = []

if __name__ == "__main__":
    path = r"C:\Users\ross.marchant\data\RAPP"
    path = path.replace("C:\\", "/mnt/c/")
    path = path.replace("\\", "/")
    print(path)

    project = CvatProject(path)
    project.summary(verbosity=0)

    project.remove_unused_labels()
    project.keep_labels(["Coccolith"])
    project.summary(verbosity=0)
