import os
from collections import OrderedDict
from glob import glob
from tqdm import tqdm
from visage.project.project import Project
from pathlib import Path


def ls_dir(path):
    return sorted([d for d in glob(os.path.join(path, "*")) if os.path.isdir(d)])


def ls(path, ext="*"):
    dir = sorted(glob(os.path.join(path, ext)))
    if isinstance(dir, str):
        dir = [dir]
    return dir


def combine_projects(base_dir, ext="*.json"):
    # Get latest projects in director
    project_files = latest_projects(base_dir, ext)
    # New project to save
    main_project = Project()
    # Add all projects
    for project_file in tqdm(project_files):
        main_project.add_project(Project.load(project_file))
    return main_project


def latest_projects(base_dir, ext="*.json"):
    # All the project folders
    dirs = ls_dir(base_dir)
    # List
    projects = dict()
    for dir in dirs:
        # Latest project in folder
        project = ls(dir, ext=ext)[-1]
        # Add to list
        projects[str(Path(dir).stem)] = project
    return projects


def latest_project(base_dir, ext="*.json"):
    # Latest project in folder
    return ls(base_dir, ext=ext)[-1]
