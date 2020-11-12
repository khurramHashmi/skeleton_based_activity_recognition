FROM tensorflow/tensorflow:1.14.0-gpu-py3 AS base
RUN ls
RUN mkdir /home/src

COPY requirements.txt /home/src
WORKDIR /home/src
RUN ls

# FROM python:3.6.9
RUN pip install -r requirements.txt
