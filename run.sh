#!/bin/bash
export LD_LIBRARY_PATH = /home/semionn/Documents/caffe/build/lib:/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH
python deep_style.py --subject images/margrethe2.jpg --style images/starry_night2.jpg
