"""
An example for the pipeline pf ANN to SNN Conversion Framework.
"""
from Train_ANN import train_ann, initialize_weights
from Test_SNN import anntosnn
from sklearn.model_selection import train_test_split
import scipy.io as scio
from model_2a import LENet
import torch

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


def framework_pipeline(train_x, train_y, test_x, test_y, epoch=200, batch=64, T=100):
    """
        ANN to SNN Conversion framework
    input:
        train_x, test_x (float): Train and test data, shape as: samples*length*ch (samples*ch*length).
        train_y, test_y (int): Train and test label, shape as: samples, ie.: [0, 1, 1, 0, ..., 2].
        epoch (int): Total train and test epoch
        batch (int): Batch size
        T (int): Time step for SNN
    output:
        None
    """
    ann_model = LENet(classes_num=4).to(device)
    ann_model.apply(initialize_weights)

    train_acc, test_acc, model_trained = train_ann(ann_model, train_x, train_y, test_x, test_y, ep=epoch, batch=batch)
    max_norm_acc = anntosnn(model_trained, train_x, train_y, test_x, test_y, batch=batch, T=T)

    print('\n')
    print('ANN accuracy: Test: %.4f%%' % (test_acc * 100))
    print('SNN accuracy: max_norm: %.4f%%' % (max_norm_acc[-1] * 100))


# Getting real samples
file = scio.loadmat(
    'D:/Constructing Lightweight and Efficient Spiking Neural Networks for EEG-based Motor Imagery '
    'Classification/s07.mat')
all_data = file['all_data']
all_label = file['all_label'].T

datasetX = torch.tensor(all_data, dtype=torch.float32)
datasetY = torch.tensor(all_label, dtype=torch.int64)
train_data, test_data, train_label, test_label = train_test_split(datasetX, datasetY, test_size=0.2, shuffle=True,
                                                                  random_state=0)

# ANN to SNN Conversion
framework_pipeline(train_data, train_label, test_data, test_label, epoch=500, batch=64, T=100)
