import models
import os
import torch
import json
import consts_noam as consts
from training_loop import set_seed
from data_loader import ImagenetteDataset, Rescale, RandomCrop,ToTensor
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import glob

def freeze_encoder_init_last_fc(encoder):
    # freeze all layers but the last fc
    for name, param in encoder.named_parameters():
        if name not in ['fc1.weight', 'fc1.bias']:
            param.requires_grad = False
    # init the fc layer
    encoder.fc1.weight.data.normal_(mean=0.0, std=0.01)
    encoder.fc1.bias.data.zero_()
    return encoder


def load_model(dir,filename=None):
    """

    :param dir: directory path where pt files and json configuration files exists
    :param filename: the name of the json and pt file with out the suffix
    :return: pre trained model that was loaded from a pt file and the relevant configuration dictionary that was loaded from a json file
    """
    if filename is None: #if filename is none get the latest file
        file_type = '/*pt'
        files = glob.glob(dir + file_type)
        filename = max(files, key=os.path.getctime)
        json_type = '/*json'
        json_files = glob.glob(dir + json_type)
        json_filename = max(json_files, key=os.path.getctime)
    else:
        json_filename = filename +'.json'

    with open(json_filename, "r") as fp:
        config = json.load(fp)

    model = models.Encoder(config['encoder_output_dim']).double()
    model.load_state_dict(torch.load(filename))
    return model, config

def fine_tune(model, config, epochs,lr, momentum):
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    model.eval()
    for epoch in range(epochs):
        print(f'start epoch {epoch}')
        for minibatch in train_loader:

            minibatch = minibatch.double()
            optimizer.zero_grad()

            output = model(minibatch)
            loss = loss_fn()

if __name__ == '__main__':
    debug = True
    pre_trained_model, config = load_model(dir = consts.SAVED_ENCODERS_DIR)

    set_seed(config['seed'])

    imagenette_dataset = ImagenetteDataset(csv_file=consts.csv_filename,
                                           root_dir=consts.image_dir,
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()
                                           ]),
                                           debug = debug)

    train_loader = DataLoader(imagenette_dataset, batch_size=config['pretraining_batch_size'])

    imagenette_dataset_validation = ImagenetteDataset(csv_file=consts.validation_filename,
                                           root_dir=consts.image_dir_validation,
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()
                                           ]),
                                            debug = debug)
    validation_loader = DataLoader(imagenette_dataset_validation, batch_size=config['pretraining_batch_size'])

