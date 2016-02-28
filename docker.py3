#!/bin/bash

docker build -t dhorta/limix-qep.py3 -f Dockerfile.py3 . && docker run -it dhorta/limix-qep.py3
