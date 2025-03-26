import torch
import torchvision
from torchvision.transforms import v2

import hydra
import torch.nn as nn
from torchinfo import summary

from train import train_model
from models.reskanet import reskalnet_18x32p

from models import SimpleConvKALN, SimpleFastConvKAN, SimpleConvKAN, SimpleConv, EightSimpleConvKALN, \
    EightSimpleFastConvKAN, EightSimpleConvKAN, EightSimpleConv, SimpleConvKACN, EightSimpleConvKACN, \
    SimpleConvKAGN, EightSimpleConvKAGN, SimpleConvWavKAN, EightSimpleConvWavKAN


def get_data():
        transforms_train = v2.Compose([
            v2.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
            v2.ToTensor(),
            v2.Normalize((0.5,), (0.5,))
        ])

        transforms_val = v2.Compose([
            v2.ToTensor(),
            v2.Normalize((0.5,), (0.5,))
        ])
        trainset_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transforms_train)
        # Load and transform the MNIST validation dataset
        valset_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transforms_val)
        # Create DataLoaders for training and validation datasets
        return trainset_dataset, valset_dataset


@hydra.main(version_base=None, config_path="./configs/", config_name="mnist-kacn.yaml")
def main(cfg):
    model = SimpleConvKACN([8 * 4, 16 * 4, 32 * 4, 64 * 4], num_classes=10, input_channels=1,
                          degree=cfg.model.degree, groups=cfg.model.groups, dropout=cfg.model.dropout, dropout_linear=cfg.model.dropout_linear,
                          l1_penalty=cfg.model.l1_activation_penalty, degree_out=cfg.model.degree_out)
    summary(model, [1, 1, 28, 28], device='cpu')
    print('END OF SUMMARY DECRICPTION, DATALOAD...')
    dataset_train, dataset_test = get_data()
    loss_func = nn.CrossEntropyLoss(label_smoothing=cfg.loss.label_smoothing)

    train_model(model, dataset_train, dataset_test, loss_func, cfg)


if __name__ == '__main__':
    main()
