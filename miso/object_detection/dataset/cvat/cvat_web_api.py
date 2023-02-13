import os
import json
from pathlib import Path
from time import sleep
from typing import List
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
from miso.object_detection.dataset.annotation import RectangleAnnotation
from miso.object_detection.dataset.image import ImageMetadata
from miso.object_detection.dataset.label import Label
from miso.object_detection.dataset.project import Project


# ----------------------------------------------------------------------------
# Python objects that mirror the JSON objects in the CVAT REST API
#
# 'minimal' function is used to create an instance with the minimum number of
# fields needed for PATCH and PUT.
#
# Fields with None will be removed when serialised
# ----------------------------------------------------------------------------
class CvatJsonSerializable:
    @staticmethod
    def del_none(d):
        """
        Delete keys with the value ``None`` in a dictionary, recursively.

        This alters the input so you may wish to ``copy`` the dict first.
        """
        for key, value in list(d.items()):
            if value is None:
                del d[key]
            elif isinstance(value, dict):
                CvatJsonSerializable.del_none(value)
        return d  # For convenience

    def to_json(self):
        return json.dumps(self, default=lambda o: CvatJsonSerializable.del_none(o.__dict__), sort_keys=True, indent=4)


class CvatLabeledShape(CvatJsonSerializable):
    def __init__(self):
        """
        Shape (polygon, box, etc) with a label
        """
        super(CvatLabeledShape, self).__init__()
        self.type: str = None
        self.occluded: bool = None
        self.z_order: int = None
        self.points: List[int] = []
        self.id: int = None
        self.frame: int = None
        self.label_id: int = None
        self.group: int = None
        self.attributes = []

    @staticmethod
    def minimal(type, occluded, points, frame, label_id, group):
        obj = CvatTrackedShape()
        obj.type = type
        obj.occluded = occluded
        obj.points = points
        obj.frame = frame
        obj.label_id = label_id
        obj.group = group
        return obj

# TODO change to PatchedProject
class CvatLabeledShapeSet(CvatJsonSerializable):
    def __init__(self):
        """
        Container for labeled shapes
        """
        super(CvatLabeledShapeSet, self).__init__()
        self.shapes: List[CvatLabeledShape]


class CvatTrackedShape(CvatJsonSerializable):
    def __init__(self):
        """
        Shape that is part of a track (sequence of shapes)
        """
        super(CvatTrackedShape, self).__init__()
        self.type: str = None
        self.occluded: bool = None
        self.z_order: int = None
        self.points: List[int] = []
        self.id: int = None
        self.frame: int = None
        self.outside: bool = None
        self.attributes = []

    @staticmethod
    def minimal(type, occluded, points, frame, outside):
        obj = CvatTrackedShape()
        obj.type = type
        obj.occluded = occluded
        obj.points = points
        obj.frame = frame
        obj.outside = outside
        return obj

class CvatLabeledTrack(CvatJsonSerializable):
    def __init__(self):
        """
        Container for tracked shapes
        """
        super(CvatLabeledTrack, self).__init__()
        self.id: int = None
        self.frame: int = None
        self.label_id: int = None
        self.group: int = None
        self.source: str = None
        self.shapes: List[CvatTrackedShape] = []
        self.attributes = []

    @staticmethod
    def minimal(frame, label_id, group, shapes):
        obj = CvatLabeledTrack()
        obj.frame = frame
        obj.label_id = label_id
        obj.group = group
        obj.shapes = shapes
        return obj


class CvatLabeledData(CvatJsonSerializable):
    def __init__(self):
        """
        Container for tracks
        """
        super(CvatLabeledData, self).__init__()
        self.version: int = None
        self.tags = []  # To add when needed
        self.shapes: List[CvatLabeledShape] = []  # To add when needed
        self.tracks: List[CvatLabeledTrack] = []

    @staticmethod
    def minimal(version, tracks=None, shapes=None, tags=None):
        obj = CvatLabeledData()
        obj.version = version
        if tracks is not None:
            obj.tracks = tracks
        if shapes is not None:
            obj.shapes = shapes
        if tags is not None:
            obj.tags = tags
        return obj


class CvatPatchedLabel(CvatJsonSerializable):
    def __init__(self):
        super(CvatPatchedLabel, self).__init__()
        self.id: int = None
        self.name: str = None
        self.color: str = None


# ----------------------------------------------------------------------------
# Classes to load CVAT jobs, tasks and projects
# ----------------------------------------------------------------------------
class CvatJob(object):
    def __init__(self, id, task_id, start_frame, stop_frame):
        self.id = id
        self.task_id = task_id
        self.start_frame = start_frame
        self.stop_frame = stop_frame


class CvatTask(object):
    def __init__(self,
                 server: str,
                 task_id: int,
                 image_root=None,
                 is_wsl2=False,
                 api="v1",
                 debug=True):
        self.server = server
        self.task_id = task_id
        self.project_id = None
        self.data_location = None
        self.name = None
        self.label_dict_by_name = dict()
        self.debug = debug
        self.frames = []
        self.tracks = dict()
        if api == 'v1':
            self.api = "api/v1"
        elif api == 'v2':
            self.api = "api"
        else:
            raise ValueError("api parameter must be 'v1' or 'v2'")
        self.image_root = image_root
        self.is_wsl2 = is_wsl2
        self.project = Project()

    def load(self):
        if self.debug:
            print("=" * 80)
            print(f"Loading CVAT task {self.task_id} - {self.name}")
            print("-" * 80)
        self._get_metadata()
        self._get_frames()
        self._get_annotations()
        self._create_project()
        if self.debug:
            print("=" * 80)

    def _create_project(self):
        # Creates a project from this task with interpolation between the key frames
        # self.project.metadata["name"] = self.task_name
        # self.project.metadata["id"] = self.id
        self.project.task_names[self.task_id] = self.name
        if self.debug:
            print("-" * 80)
            print("Creating project")
            print("- labels:")
        for key, label in self.label_dict_by_name.items():
            self.project.add_label(label['id'], label['name'], label['color'])
            if self.debug:
                print(f"  - {label['name']} ({label['color']})")
        if self.debug:
            print(f"- {len(self.frames)} images")
        frame_keys = []
        for idx, frame in enumerate(self.frames):
            if os.path.exists(os.path.join(self.image_root, frame)):
                image = ImageMetadata(frame, self.image_root, self.task_id, idx)
            elif os.path.exists(os.path.join("/home/django/share", frame)):
                image = ImageMetadata(frame, "/home/django/share", self.task_id, idx)
            else:
                print(f"Image {frame} could not be found.")
                continue
            self.project.add_image(image)
            frame_keys.append(image.id)
        for track in self.tracks:
            seq_id = track['id']
            seq_len = len(track['shapes']) - 1
            seq_idx = 0
            label = self.label_dict_by_name[track['label_id']]["name"]
            last_frame_idx = None
            last_p = None
            for i, shape in enumerate(track['shapes']):
                frame_idx = shape['frame']
                if shape['type'] == 'rectangle' and shape['outside'] is False:
                    p = np.asarray(shape['points'])
                    # Interpolate
                    if last_frame_idx is not None:
                        if frame_idx - last_frame_idx > 1:
                            for idx in range(last_frame_idx + 1, frame_idx):
                                step = (idx - last_frame_idx) / (frame_idx - last_frame_idx)
                                proj_p = last_p + (p - last_p) * step
                                image = self.project.image_dict[frame_keys[frame_idx]]
                                image.boxes.append(RectangleAnnotation(x=proj_p[0],
                                                                       y=proj_p[1],
                                                                       width=proj_p[2] - proj_p[0],
                                                                       height=proj_p[3] - proj_p[1],
                                                                       label=label,
                                                                       track_id=seq_id,
                                                                       track_len=seq_len,
                                                                       track_idx=seq_idx,
                                                                       frame_id=idx))
                                seq_idx += 1
                    # Current frame
                    image = self.project.image_dict[frame_keys[frame_idx]]
                    image.boxes.append(RectangleAnnotation(x=p[0],
                                                           y=p[1],
                                                           width=p[2] - p[0],
                                                           height=p[3] - p[1],
                                                           label=label,
                                                           track_id=seq_id,
                                                           track_len=seq_len,
                                                           track_idx=seq_idx,
                                                           frame_id=frame_idx))
                    last_frame_idx = frame_idx
                    last_p = p
                    seq_idx += 1

                elif shape['type'] == 'polygon' and shape['outside'] is False:
                    pass

        for shape in self.shapes:
            seq_id = shape['id']
            seq_len = 1
            seq_idx = 0
            label = self.label_dict_by_id[shape['label_id']]["name"]
            frame_idx = shape['frame']
            # print(shape['group'])
            p = np.asarray(shape['points'])
            if shape['type'] == 'rectangle':
                image = self.project.image_dict[frame_keys[frame_idx]]
                image.boxes.append(RectangleAnnotation(x=p[0],
                                                       y=p[1],
                                                       width=p[2] - p[0],
                                                       height=p[3] - p[1],
                                                       label=label,
                                                       track_id=seq_id,
                                                       track_len=seq_len,
                                                       track_idx=seq_idx))
                seq_idx += 1

            elif shape['type'] == 'polygon' and shape['outside'] is False:
                pass

        # if self.debug:
        #     print("- label count:")
        #     for k, v in self.project.label_count().items():
        #         print(f"   {k}: {v}")
        #     print("- sequence count:")
        #     for k, v in self.project.sequence_count().items():
        #         print(f"   {k}: {v}")

    def _get_metadata(self):
        url = f"{self.server}/{self.api}/tasks/{self.task_id}"
        if self.debug:
            print(f"Fetching task metadata from {url}...")
        data = requests.get(url, auth=HTTPBasicAuth('admin', 'admin')).json()
        self.name = data['name']
        self.project_id = data['project_id']
        self.label_dict_by_name = {label["name"]: label for label in data['labels']}
        self.label_dict_by_id = {label["id"]: label for label in data['labels']}
        self.data_location = data['data']
        print(f"- Data location: {self.data_location}")
        if self.image_root is None:
            if self.is_wsl2:
                self.image_root = f"/mnt/w/version-pack-data/community/docker/volumes/cvat_cvat_data/_data/data/{self.data_location}/raw"
            else:
                self.image_root = f"/home/django/data/data/{self.data_location}/raw"
        else:
            self.image_root = os.path.join(self.image_root, str(self.data_location), "raw")
        if self.debug:
            print(f"- Name: {self.name}")
            print("- Labels:")
            for key, label in self.label_dict_by_name.items():
                print(f"  - {label['name']}")

    def _get_frames(self):
        url = f"{self.server}/{self.api}/tasks/{self.task_id}/data/meta"
        if self.debug:
            print(f"Fetching task frames from {url}...")
        data = requests.get(url, auth=HTTPBasicAuth('admin', 'admin')).json()
        frames = [frame['name'] for frame in data["frames"]]
        if self.debug:
            print(f"- {len(frames)} frames")
        self.frames = frames

    def _get_annotations(self):
        url = f"{self.server}/{self.api}/tasks/{self.task_id}/annotations"
        if self.debug:
            print(f"Fetching task tracks from {url}...")
        data = requests.get(url, auth=HTTPBasicAuth('admin', 'admin')).json()
        self.shapes = data['shapes']
        self.tracks = data['tracks']
        if self.debug:
            print(f"- {len(self.shapes)} shapes")
            print(f"- {len(self.tracks)} tracks")

    def add_missing_labels(self, project: Project):
        # Make sure we have the most up to date labels
        self._get_metadata()
        # Check if there are new labels required
        cvat_labels = [label["name"] for key, label in self.label_dict_by_name.items()]
        new_labels = [label for label in project.label_dict.values() if label.name not in cvat_labels]
        # Add them to the project if there are
        if len(new_labels) > 0:
            self.add_labels(new_labels)
        # Load the new labels
        self._get_metadata()

    def add_labels(self, labels: List[Label]):
        url = f"{self.server}/{self.api}/projects/{self.project_id}"
        if self.debug:
            print(f"Modifying labels...")
        cvat_labels = []
        for label in labels:
            # if label.name not in self.label_dict_by_name.keys():
            label_dict = {"name": label.name}
            if label.colour is not None:
                label_dict["color"] = label.colour
            if label.id is not None:
                label_dict["id"] = label.id
            cvat_labels.append(label_dict)
        cvat_labels = {"labels": cvat_labels}
        response = requests.patch(url,
                                  data=json.dumps(cvat_labels),
                                  auth=HTTPBasicAuth('admin', 'admin'),
                                  headers={'Content-Type': "application/json"})
        print(f"Add labels result: {response.status_code}")

    def add_shapes(self, project: Project):
        self.add_missing_labels(project)
        shapes = []
        # Make sure we have the most up-to-date labels
        self._get_metadata()
        for key, image in project.image_dict.items():
            for box in image.boxes:
                # print(list(box.coords))
                shape = CvatLabeledShape.minimal("rectangle",
                                                 False,
                                                 list(box.coords_int),
                                                 image.frame_id,
                                                 self.label_dict_by_name[box.label]["id"],
                                                 0)
                shapes.append(shape)
        labeled_data = CvatLabeledData.minimal(0, shapes=shapes)

        url = f"{self.server}/{self.api}/tasks/{self.task_id}/annotations?action=create"
        if self.debug:
            print(f"Creating shapes for task...")
        content = labeled_data.to_json()
        response = requests.patch(url,
                                  data=content,
                                  auth=HTTPBasicAuth('admin', 'admin'),
                                  headers={'Content-Type': "application/json"})
        print(f"Add shapes result: {response.status_code}")


class CvatProject(object):
    def __init__(self, server, project_id, api_version="api/v1", debug=True):
        self.server = server
        self.project_id = project_id
        self.api = api_version
        self.debug = debug

        self.labels = list()
        self.id_to_label_dict = None
        self.label_to_id_dict = None

        self.tasks = dict()
        self.task_to_id_dict = dict()
        self.id_to_task_dict = dict()

    def load(self):
        if self.debug:
            print("=" * 80)
            print(f"Loading CVAT project {self.project_id}")
        self._get_metadata()

    def load_task(self, id):
        if id in self.tasks:
            return self.tasks[id]
        else:
            name = self.id_to_task_dict[id]
            task = CvatTask(self.server, id, name, debug=True)
            task.load()
            self.tasks[id] = task
            return task

    def load_task_by_name(self, name):
        id = self.task_to_id_dict[name]
        return self.load_task(id)

    def create_task(self, task_name, filenames):
        url = f"{self.server}/{self.api}/tasks"
        if self.debug:
            print(f"Creating new task {task_name}...")
        content = {"project_id": self.project_id, "name": task_name}
        response = requests.post(url, json=content, auth=HTTPBasicAuth('admin', 'admin'))
        print(response)
        id = response.json()["id"]

        url = f"{self.server}/{self.api}/tasks/{id}/data"
        if self.debug:
            print(f"Creating files for task {task_name}...")
        content = {"chunk_size": 4,
                   "image_quality": 70,
                   "client_files": [],
                   "server_files": filenames,
                   "remote_files": [],
                   "use_zip_chunks": False,
                   "use_cache": True}
        response = requests.post(url, json=content, auth=HTTPBasicAuth('admin', 'admin'))
        print(response.status_code)
        # print(response.json())
        # Sleep because CVAT sucks
        print("Sleeping to wait for images to be added to project")
        sleep(15)
        self._get_metadata()

    def create_annotations(self, task_name, annotations: CvatLabeledData, overwrite=True):
        url = f"{self.server}/{self.api}/tasks/{self.task_to_id_dict[task_name]}/annotations?action=create"
        if self.debug:
            print(f"Creating annotations for task {task_name}...")
        content = annotations.to_json()
        with open("/tmp/content.json", "w") as file:
            file.write(content)
        if overwrite:
            response = requests.put(url,
                                    data=content,
                                    auth=HTTPBasicAuth('admin', 'admin'),
                                    headers={'Content-Type': "application/json"})
        else:
            response = requests.patch(url,
                                      data=content,
                                      auth=HTTPBasicAuth('admin', 'admin'),
                                      headers={'Content-Type': "application/json"})
        print(response)

    def modify_labels(self, labels: List[Label]):
        url = f"{self.server}/{self.api}/projects/{self.project_id}"
        if self.debug:
            print(f"Modifying labels...")
        cvat_labels = []
        for label in labels:
            label_dict = {"name": label.name}
            if label.colour is not None:
                label_dict["color"] = label.colour
            if label.id is not None:
                label_dict["id"] = label.id
        cvat_labels = {"labels": cvat_labels}
        response = requests.patch(url,
                                  data=json.dumps(cvat_labels),
                                  auth=HTTPBasicAuth('admin', 'admin'),
                                  headers={'Content-Type': "application/json"})
        print(response)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Hack because some CVAT projects use the 'jpg_web' directory, which is half-size images
    # We replace the filename with the 'jpg' directory which has the full-size images
    # 2021-01-27 - removed as it requires extra work to undo the process to fix annotations before upload
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # def _convert_size(self, task):
    #     for key, im in task.project.image_dict.items():
    #         if 'jpg_web' in im.filename:
    #             im.filename = im.filename.replace('jpg_web', 'jpg')
    #             for ann in im.annotations:
    #                 ann.x *= 2
    #                 ann.y *= 2
    #                 ann.width *= 2
    #                 ann.height *= 2

    def _get_metadata(self):
        url = f"{self.server}/{self.api}/projects/{self.project_id}"
        if self.debug:
            print(f"Fetching project metadata from {url}...")
        data = requests.get(url, auth=HTTPBasicAuth('admin', 'admin')).json()
        print(data)
        self.id_to_label_dict = {label['id']: label['name'] for label in data["labels"]}
        self.label_to_id_dict = {label['name']: label['id'] for label in data["labels"]}
        # self.id_to_task_dict = {task['id']: task['name'] for task in data['tasks']}
        # self.task_to_id_dict = {task['name']: task['id'] for task in data['tasks']}
        self.metadata = data
        if self.debug:
            print("Tasks:")
            for key, val in self.id_to_task_dict.items():
                print(f" - {key:3d}: {val}")

    def export(self, output_dir, remote_path, local_path, cloud_path):
        if self.debug:
            print('-' * 80)
            print(f'Exporting tasks as project files to {output_dir}')
        for task in self.tasks:
            if self.debug:
                print(f" - {task.task_name}")
            path = os.path.join(output_dir, task.task_name)
            os.makedirs(path, exist_ok=True)
            project = task.project
            project.update_image_directory_start(remote_path, local_path)
            project.save_as(os.path.join(path, f"{task.task_name}_local.json"))
            project.update_image_directory_start(local_path, cloud_path)
            project.save_as(os.path.join(path, f"{task.task_name}_cloud_via.json"), format='via')


class Cvat(object):
    def __init__(self, server, debug=True):
        super(Cvat, self).__init__()
        self.server = server
        self.debug = debug
        self.id_to_project_dict = None
        self.project_to_id_dict = None
        self.projects = dict()

    def load(self):
        if self.debug:
            print("=" * 80)
            print(f"Loading CVAT project list")
        self._get_metadata()

    def load_project(self, id):
        if id in self.projects:
            return self.projects[id]
        else:
            project = CvatProject(self.server, id, convert_half_size=True, debug=True)
            project.load()
            self.projects[id] = project
            return project

    def load_project_by_name(self, name):
        id = self.project_to_id_dict[name]
        return self.load_project(id)

    def load_task(self, project_id, task_id):
        project = self.load_project(project_id)
        task = project.load_task(task_id)
        return task

    def load_task_by_name(self, project_name, task_name):
        project = self.load_project_by_name(project_name)
        task = project.load_task_by_name(task_name)
        return task

    def load_task_by_code(self, code):
        parts = code.split("@")
        project_name = parts[1]
        task_name = parts[0]
        task = self.load_task_by_name(project_name, task_name)
        return task

    def _get_metadata(self):
        url = f"{self.server}/api/v1/projects?names_only=true"
        if self.debug:
            print(f"Fetching projects from {url}...")
        data = requests.get(url, auth=HTTPBasicAuth('admin', 'admin')).json()
        self.id_to_project_dict = {project['id']: project['name'] for project in data["results"]}
        self.project_to_id_dict = {project['name']: project['id'] for project in data["results"]}
        if self.debug:
            print("Projects:")
            for key, val in self.id_to_project_dict.items():
                print(f" - {key:3d}: {val}")


def create_task_annotations_patch(project: Project, label_to_id_dict: dict):
    track_dict = project.sequence_dict()
    cvat_tracks = []
    for id, track in track_dict.items():
        if len(track) > 0:
            cvat_track = CvatLabeledTrack.minimal(frame=track[0].frame_id,
                                                  label_id=label_to_id_dict[track[0].label],
                                                  group=0,
                                                  shapes=[])
            # Actual shapes
            for ann in track:
                cvat_tracked_shape = CvatTrackedShape.minimal(type="rectangle",
                                                              occluded=False,
                                                              points=[ann.x, ann.y, ann.x + ann.width,
                                                                      ann.y + ann.height],
                                                              frame=ann.frame_id,
                                                              outside=False)
                cvat_track.shapes.append(cvat_tracked_shape)
            # Dummy closing shape
            cvat_tracked_shape = CvatTrackedShape.minimal(type="rectangle",
                                                          occluded=False,
                                                          points=[ann.x, ann.y, ann.x + ann.width, ann.y + ann.height],
                                                          frame=ann.frame_id + 1,
                                                          outside=True)
            cvat_track.shapes.append(cvat_tracked_shape)
            cvat_tracks.append(cvat_track)
    return CvatLabeledData.minimal(0, cvat_tracks)


if __name__ == "__main__":
    task = CvatTask("http://localhost:8080", 1, debug=True)
    task.load()
    task.project.add_label(None, "Coccolith_NV", "#ff0000")
    task.add_missing_labels(task.project)
    # cvat.load()
    #
    # tasks_list = ["20200717_lamont_1_v4@Heron Island July 2020",
    #               "20200716_fitzroy_ff_v4@Heron Island July 2020"]
    #
    # for task in tasks_list:
    #     task = cvat.load_task_by_code(task)
    #     project = task.project
    #
    #     project.summary()

# def create_task_annotations_patch(project: Project, label_to_id_dict: dict):
#     track_dict = project.sequence_dict()
#     cvat_tracks = []
#     for id, track in track_dict.items():
#         if len(track) > 0:
#             cvat_track = CVATLabeledTrack.minimal(frame=track[0].frame_id,
#                                                   label_id=label_to_id_dict[track[0].label],
#                                                   group=0,
#                                                   shapes=[])
#             # Actual shapes
#             for ann in track:
#                 cvat_tracked_shape = CVATTrackedShape.minimal(type="rectangle",
#                                                               occluded=False,
#                                                               points=[ann.x, ann.y, ann.x + ann.width,
#                                                                       ann.y + ann.height],
#                                                               frame=ann.frame_id,
#                                                               outside=False)
#                 cvat_track.shapes.append(cvat_tracked_shape)
#             # Dummy closing shape
#             cvat_tracked_shape = CVATTrackedShape.minimal(type="rectangle",
#                                                           occluded=False,
#                                                           points=[ann.x, ann.y, ann.x + ann.width, ann.y + ann.height],
#                                                           frame=ann.frame_id + 1,
#                                                           outside=True)
#             cvat_track.shapes.append(cvat_tracked_shape)
#             cvat_tracks.append(cvat_track)
#     return CVATLabeledData.minimal(0, cvat_tracks)
