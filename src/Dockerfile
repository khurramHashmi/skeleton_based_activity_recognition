FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
LABEL repository="Skeleton Activity"

RUN apt-get update && apt update && \
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
#RUN apt install -y libgl1-mesa-glx

WORKDIR /workspace

COPY ./src ./

RUN apt update && apt install wget
RUN pip install --no-cache-dir -r requirement.txt 
RUN wget https://cloud.dfki.de/owncloud/index.php/s/r36KpEyCZe3FZYX/download
RUN mv download data_rgb.tar.gz
RUN tar -xzf data_rgb.tar.gz 
RUN wget https://cloud.dfki.de/owncloud/index.php/s/N8PCBYyDBXHFoHK/download
RUN mv download xsub_train_rgb.csv
RUN wget https://cloud.dfki.de/owncloud/index.php/s/k6A2EB4pDtsJCqW/download
RUN mv download xsub_val_rgb.csv

 
CMD ["/bin/bash"]
