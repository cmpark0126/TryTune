From python:3.10.12-slim

WORKDIR /home

RUN apt-get update && apt-get -y install git libgl1-mesa-glx libglib2.0-0

RUN git clone https://github.com/cmpark0126/trytune-py
RUN cd trytune-py 
RUN pip install -r requirements.txt && pip install tritonclient[http] && pip install .

WORKDIR /home/trytune-py
