from spikingjelly.activation_based import ann2snn
import torch
import numpy as np
from Data_loader import data_loader

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


def val_snn(Dec, test_loader, T=None):
    Dec.eval().to(device)
    correct = 0
    total = 0
    if T is not None:
        corrects = np.zeros(T)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if T is None:
                outputs = Dec(inputs)
                correct += (outputs.argmax(dim=1) == targets.to(device)).float().sum().item()
            else:
                for m in Dec.modules():
                    if hasattr(m, 'reset'):
                        m.reset()
                for t in range(T):
                    if t == 0:
                        outputs = Dec(inputs)
                    else:
                        outputs += Dec(inputs)
                    corrects[t] += (outputs.argmax(dim=1) == targets.to(device)).float().sum().item()
            total += targets.shape[0]
    return correct / total if T is None else corrects / total


def anntosnn(ann_model, train_x, train_y, test_x, test_y, batch=64, T=None):
    # Define data loader
    train_loader = data_loader(train_x, train_y, batch=batch)
    test_loader = data_loader(test_x, test_y, batch=batch)

    print('---------------------------------------------')
    print('Converting using MaxNorm')
    model_converter = ann2snn.Converter(mode='max', dataloader=train_loader)
    snn_model = model_converter(ann_model)
    mode_max_accs = val_snn(snn_model, test_loader, T=T)

    return mode_max_accs
