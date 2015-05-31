from caffe.proto import caffe_pb2

import create_proto as cp
from create_proto import layers as L, params as P, NetSpec
from create_proto import conv_relu, fc_relu, max_pool

# helper function for common structures

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad, group=group)
    return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout):
    fc = L.InnerProduct(bottom, num_output=nout)
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def alexnet(lmdb, batch_size=256, include_acc=False):
    net = NetSpec()
    net.data, net.label = L.Data(source=lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
        transform_param=dict(crop_size=227, mean_value=[104, 117, 123], mirror=True))

    # the net itself
    net.conv1, net.relu1 = conv_relu(net.data, 11, 96, stride=4)
    net.pool1 = max_pool(net.relu1, 3, stride=2)
    net.norm1 = L.LRN(net.pool1, local_size=5, alpha=1e-4, beta=0.75)
    net.conv2, net.relu2 = conv_relu(net.norm1, 5, 256, pad=2, group=2)
    net.pool2 = max_pool(net.relu2, 3, stride=2)
    net.norm2 = L.LRN(net.pool2, local_size=5, alpha=1e-4, beta=0.75)
    net.conv3, net.relu3 = conv_relu(net.norm2, 3, 384, pad=1)
    net.conv4, net.relu4 = conv_relu(net.relu3, 3, 384, pad=1, group=2)
    net.conv5, net.relu5 = conv_relu(net.relu4, 3, 256, pad=1, group=2)
    net.pool5 = max_pool(net.relu5, 3, stride=2)
    net.fc6, net.relu6 = fc_relu(net.pool5, 4096)
    net.drop6 = L.Dropout(net.relu6, in_place=True)
    net.fc7, net.relu7 = fc_relu(net.drop6, 4096)
    net.drop7 = L.Dropout(net.relu7, in_place=True)
    net.fc8 = L.InnerProduct(net.drop7, num_output=1000)
    net.loss = L.SoftmaxWithLoss(net.fc8, net.label)

    if include_acc:
        net.acc = L.Accuracy(net.fc8, net.label)
    return net.to_proto()

def make_net():
    cp.dump_proto('train.prototxt', alexnet('/path/to/caffe-train-lmdb'))
    cp.dump_proto('test.prototxt', alexnet('/path/to/caffe-val-lmdb'))

if __name__ == '__main__':
    make_net()
