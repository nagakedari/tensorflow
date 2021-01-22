# FROM tensorflow/tensorflow:latest-devel
# FROM tensorflow/tensorflow:devel-gpu
FROM nvidia/cuda:10.2-base
FROM tensorflow/tensorflow:devel-gpu
FROM python:3
CMD nvidia-smi
CMD tensorflow/tensorflow:devel-gpu --gpus all
WORKDIR /opt/app
# RUN pip3 install tensorflow
# RUN pip3 install  tensorflow-gpu
COPY . /opt/app
CMD ["python", "/opt/app/image_classification.py"]