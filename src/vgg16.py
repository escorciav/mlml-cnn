from caffe.proto import caffe_pb2

import create_proto as cp
from create_proto import layers as L, params as P, NetSpec
from create_proto import conv_relu, fc_relu, max_pool

def add_data_blob(net, batch_size=256, **kwargs):
    """Setup data layer, input domain, of a network specification"""
    if kwargs['input'] == 'image_data_no_label':
        img_src, img_root  = kwargs['img_src'], kwargs['img_root']
        img_transf = kwargs['img_transf']
        net.data, net.dummy = L.ImageData(source=img_src, ntop=2,
            batch_size=batch_size, transform_param=img_transf)
        net.silence = L.Silence(net.dummy)
    else:
        raise ValueError, 'No information about input of the network'
    return net

def add_loss_layer(net, **kwargs):
    """Setup loss layer of a network specification"""
    if kwargs['loss'] == 'l2-norm':
        net.loss = L.EuclideanLoss(net.fc8, net.label)
    else:
        raise ValueError, 'No information about loss function of the network'
    return net

def vgg16_core(net, n_output=1000, **kwargs):
    """Update a caffe:network with the VGG-16 architecture

    Parameters
    ----------
    net : NetSpec instance
        Network specifications with a blob called "data"

    Returns
    -------
    net : NetSpec instance
        updated network. This is an in-place change.

    """
    net.conv1_1, net.relu1_1 = conv_relu(net.data, 3, 64, pad=1)
    net.conv1_2, net.relu1_2 = conv_relu(net.relu1_1, 3, 64, pad=1)
    net.pool1 = max_pool(net.relu1_2, 2, stride=2)
    net.conv2_1, net.relu2_1 = conv_relu(net.pool1, 3, 128, pad=1)
    net.conv2_2, net.relu2_2 = conv_relu(net.relu2_1, 3, 128, pad=1)
    net.pool2 = max_pool(net.relu2_2, 2, stride=2)
    net.conv3_1, net.relu3_1 = conv_relu(net.pool2, 3, 256, pad=1)
    net.conv3_2, net.relu3_2 = conv_relu(net.relu3_1, 3, 256, pad=1)
    net.conv3_3, net.relu3_3 = conv_relu(net.relu3_2, 3, 256, pad=1)
    net.pool3 = max_pool(net.relu3_3, 2, stride=2)
    net.conv4_1, net.relu4_1 = conv_relu(net.pool3, 3, 512, pad=1)
    net.conv4_2, net.relu4_2 = conv_relu(net.relu4_1, 3, 512, pad=1)
    net.conv4_3, net.relu4_3 = conv_relu(net.relu4_2, 3, 512, pad=1)
    net.pool4 = max_pool(net.relu4_3, 2, stride=2)
    net.conv5_1, net.relu5_1 = conv_relu(net.pool4, 3, 512, pad=1)
    net.conv5_2, net.relu5_2 = conv_relu(net.relu5_1, 3, 512, pad=1)
    net.conv5_3, net.relu5_3 = conv_relu(net.relu5_2, 3, 512, pad=1)
    net.pool5 = max_pool(net.relu5_3, 2, stride=2)
    net.fc6, net.relu6 = fc_relu(net.pool5, 4096)
    net.drop6 = L.Dropout(net.relu6, in_place=True)
    net.fc7, net.relu7 = fc_relu(net.drop6, 4096)
    net.drop7 = L.Dropout(net.relu7, in_place=True)
    net.fc8 = L.InnerProduct(net.drop7, num_output=n_output)
    return net

def vgg16_multilabel_hdf5(filename, **kwargs):
    """Return a string to dump the prototxt of a vgg16 arch cnn for multilabel
    task using hdf5-layer for labels
    """
    net = NetSpec()
    # Setup blob for input
    net = add_data_blob(net, **kwargs)
    # Add vgg16 core layers
    net = vgg16_core(net, **kwargs)
    # Add target layer
    net.label = L.HDF5Data(source=kwargs['label_src'],
        batch_size=kwargs['batch_size'])
    # Add loss layer
    net = add_loss_layer(net, **kwargs)
    # Save network as protoxt
    if filename is not None:
        cp.dump_proto(filename, net.to_proto())
    return net

