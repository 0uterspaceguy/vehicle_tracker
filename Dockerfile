FROM nvcr.io/nvidia/tensorrt:23.01-py3

RUN apt-get update -y && apt-get install ffmpeg libsm6 libxext6 -y

COPY ./requirements.txt /workspace/requirements.txt
RUN pip install -r requirements.txt

WORKDIR /workspace

COPY ./* /workspace/
