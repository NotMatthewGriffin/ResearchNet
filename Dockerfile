FROM nvcr.io/nvidia/pytorch:19.02-py3
RUN ["apt", "update"]
RUN ["apt","--yes", "install","libsm6", "libxext6", "libxrender1", "libglib2.0-0"]
RUN ["pip", "install", "-U", "pip"]
RUN ["pip", "install", "opencv-contrib-python"]
RUN ["conda", "install", "pytorch", "torchvision", "cudatoolkit=9.0", "-c", "pytorch"]

CMD ["/bin/bash"]
