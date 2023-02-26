# Object Detection Tools for Microfossil Particles

This project contains code and a docker image for training object detectors using annotations created using CVAT

**RAPP USERS: If doing this from home, connect to the osupytheas VPN first**

## Server Installation

The following instructions are for installation on a server, or on your local machine if you want to run CVAT locally.

**RAPP USERS: please skip this section, installation has been completed on the server already**

### 1. Install CVAT

CVAT is an annotation software developed by Intel. Install CVAT using the default settings.

### 2. Install Docker

The best way to use this software is via a docker image. The docker image contains all the software dependencies and code so that you do not have to worry about configuring libraries or other dependencies. Install Docker Desktop for Windows or MacOS, or normal docker for Linux.

### 3. Install NVIDIA container toolkit

This is required so that you can use the GPU from docker

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

### 4. Create / pull docker image

Pull the docker image from the docker container repository, this will take a while the first time:

```shell
docker pull ghcr.io/microfossil/miso:latest
```

Alternatively, you can build it from source

```shell
git clone https://github.com/microfossil/particle-object-detection.git
cd particle-object-detection
docker build -t ghcr.io/microfossil/miso:latest . 
```

## Add yourself to the list of docker users

This must be completed by someone with sudo access.

**RAPP USERS: Please ask the system administrator to add you**

### 1. Create docker group

```shell
sudo groupadd docker
```

### 2. Add your username to docker group

```shell
sudo usermod -aG docker $USER
```

### 3. Login again

Logout from the server / local machine and log back in again

## Training set

**RAPP USERS: CVAT can be accessed from https://rapp.osupytheas.fr:8080**

Object detector models learn to place a bounding box (rectangle) around objects in an image. 

To train the detector we must first create a good training set. Usage of CVAT is covered in depth on their website but a quick outline is:

### 1. Create a project in CVAT

A CVAT project has a set of labels (detection classes) and contains "tasks". A CVAT task is a collection of images. The project labels are used for all tasks. 

For example, one might create a project for coccolith detection with labels for different species of coccoliths. To this project one might add a task for each set of images for each sample. 

### 2. Annotate the images

Draw bounding boxes around all the objects in the images correspond to a label.

You do not have to annotate every image. For example, you could label 10% of the images, train the object detector on those images, and then used the trained model to help label the remaining images.

However, it is very important that ALL objects corresponding to a label are annotated in a single image. If you do not draw a bounding box around an object that should be labelled, that will send a signal during training that the object is NOT that label, i.e. it is background.

An image may be left unannotated, as images with no annotations will not be used in training.

### 3. Validate the images

Review the annotated images to make sure nothing was missed.

You also want to ensure that enough examples of each class are present in the training set. Don't worry if this is not possible, at training time you can choose which labels you wish to train on.

## Training

**RAPP USERS: Open a terminal and connect to the RAPP server with `ssh -X USERNAME@rapp.osupytheas.fr`. User the terminal to perform these commands.**

### 1. Update

Make sure you have the latest version of the docker image

```shell
docker pull ghcr.io/microfossil/miso:latest
```

### 2. Create a directory to store models

If not done already, create a directory to store trained models and cropped images

```shell
mkdir ~/obj_det
```

### 3. Start the docker image

The following will start the image and set the directory where models will be stored to the `~/obj_det` directory. To save the models in another directory, replace `/home/$USER/obj_det` with that directory.

```shell
docker run --rm --shm-size 16G --gpus all --net=cvat_cvat -v /home/$USER/obj_det:/obj_det --volumes-from cvat -it ghcr.io/microfossil/miso:latest /bin/bash
```

A shell prompt should appear.

### 4. Get task numbers

Make a list of the numbers of all the CVAT tasks that you wish to use in training. The number is written to the left of the task name in the tasks list. Tasks do not need to be from the same project.

### 5. Get labels

Make a list of all the labels that you wish to train on. The labels available are at the top of the page for a project.

### 6. Launch training

Configure the following command to launch training. 

```shell
python -m miso.cli train-object-detector --tasks "15,16,18" --labels "Coccolith,Coccosphere" --model "Coccoliths" --batch-size 4 --api "v1" --max-epochs 1000 
```

Parameters:

* tasks: List of numbers of the tasks to train on
* labels: List of labels to train on. _Omit labels if you want to train on all labels with an annotation_
* model: The name of the output model. If the same name is used, the model will be overwritten. If omitted, the current date and time will be used
* batch-size: Number of images in a batch.
* api: The CVAT api version, either "v1" or "v2" depending on which version CVAT is installed. To check, go to the CVAT site and enter "api/swagger" after the address, e.g.: `http://localhost:8080/api/swagger`. If it says "CVAT REST API 1.0" then use "v1", if it says "CVAT REST API 2.0" then use "v2".
* max-epochs: The maximum number of epochs to train on. Training will be stopped when the accuracy is no longer improving, or when this number of epochs is reached.

E.g. the above command trains a model to detect "Coccolith" and "Coccosphere" using the images from tasks 15, 16, and 18

Training will go faster with a larger batch size. The maximum batch size is limited by the GPU memory. If your GPU has large memory or your images are less than 1000 x 1000 try increasing the batch size. 

### 6. Results

The trained model will be store in your home directory at `~/obj_det/models/MODEL_NAME` where `MODEL_NAME` is the name of the model.

Inside are three files:

* `model.pt`: The trained pytorch model
* `labels.txt`: A list of the labels that the model can predict
* `results.txt`: The performance of the model on the COCO metrics

## Inference

### 1. Add images / choose tasks

Add images you wish to classify to a new task or tasks. Alternatively, choose a task with unannotated images.

**Inference will only be performed on images with no annotations.**

### 2. Choose the model

Choose the model you with to use for inference. The models are in `~/obj_det/models`. The name of the model is the name of the directory.

### 3. Run inference

Configure the following command to launch training. 

```shell
python -m miso.cli infer-object-detector --tasks "15,16,18" --model "Coccoliths" --threshold 0.5 --nv --api "v1" 
```

Parameters:

* tasks: List of numbers of the tasks to train on
* model: The name of the model to use for inference
* threshold: Detection threshold (0 - 1). Choose a lower value to have more detections, but with more errors, or larger value for less, more accurate detections
* nv: This option will add "_NV" to the labels for detection. Omit if you do not want the labels to have "_NV" at the end.
* api: The CVAT api version, either "v1" or "v2" depending on which version CVAT is installed. To check, go to the CVAT site and enter "api/swagger" after the address, e.g.: `http://localhost:8080/api/swagger`. If it says "CVAT REST API 1.0" then use "v1", if it says "CVAT REST API 2.0" then use "v2".
* batch-size: Number of images in a batch (default 2)

E.g. the above command uses the model called "Coccolith" to perform inference on tasks 15, 16 and 18

The unannotated images in the tasks will be now be labelled.

**The images will use labels from training but with `_NV` appended. NV stands for "not validated"**

### 4. Validate + add missing

Open each task and validate the detections. 

**If NV option used:**

If the detection is correct, change the label from the _NV version to the original label. E.g. if a object with label "Coccolith_NV" is correct, change the label to "Coccolith". You can use the keyboard shortcuts to make this quick. In the labels tab in the CVAT annotator view, each label has a number next to it. Pressing Ctrl + this number (e.g. Ctrl+1) will set the label of selected object to this number.

Note: You do not need to delete the incorrect detections. Leave them as _NV as they will be deleted all at once in the next step

**If NV option not used**

If the detection is incorrect, delete the annotation.

Review the image and annotate any missed objects.

****

### 5. Remove _NV (if NV option used)

Once all the detection have been validated, we can remove the _NV labels. Go to the project page for each task and delete the _NV labels. This will also remove all the _NV annotations.

### 6. Retrain the model

Now that you have added more annotations, retrain the model. Use the same name for the model if you want to overwrite it in the model directory.

## Crop

### 1. Choose tasks

Choose the tasks from which to crop the images

### 2. Run crop

Configure the following command and run to start the cropping

```shell
python -m miso.cli crop-objects --tasks "15,16,18" --api "v1"
```

## Inference and crop of images

This function is for inferring on images and not CVAT tasks

### 1. Add images

Add images to a directory inside the `obj_det` directory, e.g. `~/obj_det/images/dataset1`

### 2. Choose the model

Choose the model you with to use for inference. The models are in `~/obj_det/models`. The name of the model is the name of the directory.

### 3. Run inference

Configure the following command to launch training. 

Note that the input directory is the internal directory in the docker container starting from `/obj_det`, e.g. `~/obj_det/images/dataset1` (which expands to `/home/USERNAME/obj_det/images/dataset1`) would be just `/obj_det/images/dataset1`.

```shell
python -m miso.cli infer-object-detector-directory --input-dir "/obj_det/images/dataset1" --model "Coccoliths" --threshold 0.5
```

Inference will be performed and the crops saved inside a `crops` directory in the images directory.

Parameters:

* input-dir: Directory of images
* model: The name of the model to use for inference
* threshold: Detection threshold (0 - 1). Choose a lower value to have more detections, but with more errors, or larger value for less, more accurate detections
* batch-size: Number of images in a batch (default 2)

# Troubleshooting

## Bad file descriptor

```python
Traceback (most recent call last):
  File "/app/miniconda/lib/python3.9/multiprocessing/resource_sharer.py", line 145, in _serve
    send(conn, destination_pid)
  File "/app/miniconda/lib/python3.9/multiprocessing/resource_sharer.py", line 50, in send
    reduction.send_handle(conn, new_fd, pid)
  File "/app/miniconda/lib/python3.9/multiprocessing/reduction.py", line 184, in send_handle
    sendfds(s, [handle])
  File "/app/miniconda/lib/python3.9/multiprocessing/reduction.py", line 149, in sendfds
    sock.sendmsg([msg], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds)])
OSError: [Errno 9] Bad file descriptor

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/app/miniconda/lib/python3.9/multiprocessing/resource_sharer.py", line 147, in _serve
    close()
  File "/app/miniconda/lib/python3.9/multiprocessing/resource_sharer.py", line 52, in close
    os.close(new_fd)
OSError: [Errno 9] Bad file descriptor
```

Increase the amount of memory