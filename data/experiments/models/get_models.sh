#!/bin/bash -l
# Use this script to download caffe-models used in this project

VGG_16=http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
echo 'Download VGG-16 model trained for ILSVRC-2014'
wget $VGG_16 -O data/experiments/models/VGG_ILSVRC_16_layers.caffemodel
