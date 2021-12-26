import models
import os
import torch
import json
from training_loop import set_seed
from data_loader import ImagenetteDataset, Rescale, RandomCrop, ToTensor
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import glob
import consts


def freeze_encoder_init_last_fc(encoder):
    # freeze all layers but the last fc
    for name, param in encoder.named_parameters():
        if name not in ['fc1.weight', 'fc1.bias']:
            param.requires_grad = False
    # init the fc layer
    encoder.fc1.weight.data.normal_(mean=0.0, std=0.01)
    encoder.fc1.bias.data.zero_()
    return encoder


def add_classification_layers(model,hidden_size,num_of_labels):
    model.fc1 = nn.Linear(hidden_size, num_of_labels)
    model.non_linear_func = nn.Softmax()
    print(model)
    return model



def load_model(dir, filename=None):
    """
    :param dir: directory path where pt files and json configuration files exists
    :param filename: the name of the json and pt file with out the suffix
    :return: pre trained model that was loaded from a pt file and the relevant configuration dictionary that was loaded
     from a json file
    """
    if filename is None: # If filename is none get the latest file
        file_type = '/*pt'
        files = glob.glob(dir + file_type)
        filename = max(files, key=os.path.getctime)
        json_type = '/*json'
        json_files = glob.glob(dir + json_type)
        json_filename = max(json_files, key=os.path.getctime)
    else:
        json_filename = filename + '.json'

    with open(json_filename, "r") as fp:
        config = json.load(fp)

    model = models.Encoder(config['encoder_output_dim']).double()
    model.load_state_dict(torch.load(filename))
    return model, config


def fine_tune(model,train_loader, config, epochs,lr, momentum):
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    model = freeze_encoder_init_last_fc(model)
    model = add_classification_layers(model, hidden_size=consts.HIDDEN_REPRESENTATION_DIM ,num_of_labels=consts.NUM_OF_CLASSES)
    model = model.double()
    acc = []
    loss = []
    for epoch in range(epochs):
        print(f'start epoch {epoch}')
        for minibatch, lables in train_loader:
            minibatch = minibatch.double()
            optimizer.zero_grad()

            output = model(minibatch) #output shape is [batch_size,number_of_classes]
            loss_minibatch = loss_fn(output,torch.nn.functional.one_hot(lables.to(torch.int64), num_classes=consts.NUM_OF_CLASSES).to(float))
            preds = torch.argmax(output, dim =1)
            acc1 = torch.eq(preds, lables).sum().float().item() / preds.shape[0]
            acc.append(acc1)
            loss.append(loss_minibatch)

        avg_acc = sum(acc) / len(acc)
        avg_loss  = sum(loss) / len(loss)
        print(f'epoch = {epoch} avg_acc = {avg_acc} avg_loss = {avg_loss}')
        return model


if __name__ == '__main__':
    debug = True
    epochs = 2 if debug else 100
    lr = 0.01
    momentum = 0.9

    pre_trained_model, config = load_model(dir=consts.SAVED_ENCODERS_DIR)

    set_seed(config['seed'])

    imagenette_dataset = ImagenetteDataset(csv_file=consts.csv_filename,
                                           root_dir=consts.image_dir,
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()
                                           ]),
                                           labels=True,
                                           debug=debug)

    train_loader = DataLoader(imagenette_dataset, batch_size=config['pretraining_batch_size'])

    imagenette_dataset_validation = ImagenetteDataset(csv_file=consts.validation_filename,
                                                      root_dir=consts.image_dir_validation,
                                                      transform=transforms.Compose([
                                                          Rescale(256),
                                                          RandomCrop(224),
                                                          ToTensor()
                                                      ]),
                                                      labels=True,
                                                      debug=debug)

    validation_loader = DataLoader(imagenette_dataset_validation, batch_size=config['pretraining_batch_size'])

    fine_tuned_model = fine_tune(pre_trained_model,train_loader,config,epochs,lr,momentum)

