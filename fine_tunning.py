import data_loader
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
import argparse
import wandb

# TODO: Evaluate the fine-tuned model on Imagenette verification set. Top-1 accuracy should be > 0.85.

parser = argparse.ArgumentParser(
    description='Process flags for unsupervised pre-training with MoCo.')
parser.add_argument('--debug',
                    type=bool,
                    default=True,
                    required=False,
                    help="Whether or not to run fine-tuning in debug mode. In debug mode, the model learns over "
                         "a subset of the original dataset.")
parser.add_argument('--fine_tuning_epochs',
                    type=int,
                    default=100,
                    required=False,
                    help='The number of epochs used during fine-tuning.')
parser.add_argument('--fine_tuning_learning_rate',
                    type=float,
                    default=1e-2,
                    help='The initial learning rate used during fine-tuning.')
parser.add_argument('--fine_tuning_momentum',
                    type=float,
                    default=9e-1,
                    help='The momentum value during fine-tuning.')
parser.add_argument('--fine_tuning_batch_size',
                    type=int,
                    default=64,
                    help='The mini-batch size used during fine-tuning with MoCo.')


def freeze_encoder_init_last_fc(encoder):
    """
    :param encoder: A pre-trained MoCo encoder Pytorch model. Typically, this is a variant of Resnet.
    :return: A frozen version of the inputted encoder model, and with the final fully connected layer reset.
    """
    # freeze all layers but the last fc
    for name, param in encoder.named_parameters():
        if name not in ['fc1.weight', 'fc1.bias']:
            param.requires_grad = False
    # init the fc layer
    encoder.fc1.weight.data.normal_(mean=0.0, std=0.01)
    encoder.fc1.bias.data.zero_()
    return encoder


def add_classification_layers(model, hidden_size, num_of_labels, debug=False):
    """

    :param model: A pre-trained MoCo encoder Pytorch model. Typically, this is a variant of Resnet.
    :param hidden_size: The size of the hidden layer of our encoder. In Resnet for example, the default hidden dimension
        has size 2048.
    :param num_of_labels: The number of labels in the dataset we are fine-tuning our model on.
    :param debug: Whether or not we are running fine-tuning in a lightweight debug mode.
    :return: The original model, but with an additional linear layer followed by a softmax layer.
    """
    model.fc1 = nn.Linear(hidden_size, num_of_labels)
    model.non_linear_func = nn.Softmax(dim=1)
    if debug:
        print(model)
    return model


def load_model(dir, filename=None):
    """
    :param dir: directory path where pt files and json configuration files exists
    :param filename: the name of the json and pt file with out the suffix
    :return: pre trained model that was loaded from a pt file and the relevant configuration dictionary that was loaded
     from a json file
    """
    if filename is None:  # If filename is none get the latest file
        file_type = '/*pt'
        files = glob.glob(dir + file_type)
        filename = max(files, key=os.path.getctime)
        json_type = '/*json'
        json_files = glob.glob(dir + json_type)
        json_filename = max(json_files, key=os.path.getctime)
    else:
        json_filename = filename + consts.MODEL_CONFIGURATION_FILE_ENCODING

    with open(json_filename, "r") as fp:
        config = json.load(fp)

    model = models.Encoder(config['encoder_output_dim']).double()
    model.load_state_dict(torch.load(filename))
    return model, config


def fine_tune(model, train_loader, epochs, lr, momentum):
    """
    :param model: A pre-trained MoCo encoder Pytorch model. Typically, this is a variant of Resnet.
    :param train_loader: `torch.utils.data.DataLoader` instance that provides batches of training data for MoCo to
        fine-tune on.
    :param epochs: The number of training epochs used during fine-tuning.
    :param lr: The initial learning rate used during fine-tuning.
    :param momentum: The momentum value used by the optimizer during fine-tuning.
    :return: The fine-tuned model. Note that the outputted model has a new classifier head relative to the input model.
        The new classifier head of the model predicts imagenette labels.
    """
    wandb.watch(model)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    model = freeze_encoder_init_last_fc(model)
    model = add_classification_layers(model, hidden_size=consts.HIDDEN_REPRESENTATION_DIM,
                                      num_of_labels=consts.NUM_OF_CLASSES)
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
            wandb.log({"mini-batch loss": loss_minibatch,
                       "mini-batch accuracy@1": acc1})
            acc.append(acc1)
            loss.append(loss_minibatch)

        avg_acc = sum(acc) / len(acc)
        avg_loss = sum(loss) / len(loss)
        wandb.log({'epoch loss': avg_loss,
                   'epoch accuracy': avg_acc})
        print(f'epoch = {epoch} avg_acc = {avg_acc} avg_loss = {avg_loss}')
        return model


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    debug = args.debug
    epochs = 2 if debug else args.fine_tuning_epochs
    lr = args.fine_tuning_learning_rate
    momentum = args.fine_tuning_momentum,
    if not os.path.exists(consts.validation_filename):
        data_loader.create_csv_file(dir=consts.image_dir_validation, filename=consts.validation_filename)
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

    train_loader = DataLoader(imagenette_dataset, batch_size=args.fine_tuning_batch_size)

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

    # Note: momentum is perceived as a tuple of floats with length 1. The momentum value is the sole entry in this
    # tuple.
    fine_tuned_model = fine_tune(pre_trained_model, train_loader, epochs, lr, momentum[0])

