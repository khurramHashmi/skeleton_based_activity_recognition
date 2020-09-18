FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
LABEL repository="Skeleton Activity"

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   python3 \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    mkl \
    torch

WORKDIR /workspace

COPY ./src ./
RUN cd skeleton_based_acitvity_recognition/
RUN pip install --no-cache-dir -r requirements.txt 
 
CMD ["/bin/bash"]