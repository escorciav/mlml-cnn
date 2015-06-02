# Helper

This folder is to keep track on the experiments performed with the ESP-GAME
dataset.

**General facts:**

- Use training set for finetuning fc-layers of VGG16 to predict a multi-label
vector.

- Replace final FC-1000 layer of VGG16 by FC-268 layer.

## List

### aux

Allocate common information of the dataset such as txt-source, 
label-matrices (using standard format) and caffe-prototxt templates

### 00

**Description:**

- Use l2-norm as loss layer
