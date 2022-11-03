FROM ghcr.io/microfossil/pytorch-base-image:latest

# Preload weights
RUN mkdir -p /root/.cache/torch/hub/checkpoints
RUN curl -sLo /root/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"

# Directories to store
RUN mkdir /models
RUN mkdir /data
RUN mkdir /obj_det

# Install this library
WORKDIR /app
ADD miso /app/miso
COPY setup.py .
RUN pip install -e .

WORKDIR /obj_det

RUN bash -ic 'history -s python -m miso.cli crop-objects --tasks \"1,2\" --api \"v1\"'
RUN bash -ic 'history -s python -m miso.cli infer-object-detector --tasks \"1,2\" --model Coccoliths --threshold 0.5 --nv --api \"v1\"'
RUN bash -ic 'history -s python -m miso.cli train-object-detector --tasks \"1,2\" --labels \"Coccolith,Coccosphere\" --model \"Coccoliths\" --batch-size 8 --api \"v1\"'

