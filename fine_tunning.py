import data_loader
import models
import os
import torch
import json
import torch.optim.lr_scheduler as lr_scheduler
from training_loop import set_seed, normalize
from data_loader import ImagenetteDataset, Rescale, RandomCrop, ToTensor
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import glob
import consts
import argparse
import wandb


parser = argparse.ArgumentParser(
    description='Process flags for unsupervised pre-training with MoCo.')
parser.add_argument('--pretrained_encoder_file_name',
                    type=str,
                    default=None,
                    required=False,
                    help="The filename of a saved encoder after MoCo pre-training.")
parser.add_argument('--fine_tuning_debug',
                    type=bool,
                    default=False,
                    required=False,
                    help="Whether or not to run fine-tuning in debug mode. In debug mode, the model learns over "
                         "a subset of the original dataset.")
parser.add_argument('--fine_tuning_epochs',
                    type=int,
                    default=2,
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
    Freeze all layers of the MoCo pre-trained encoder with the exception of the final fully connected layers.

    :param encoder: A pre-trained MoCo encoder Pytorch model. Typically, this is a variant of Resnet.
    :return: A frozen version of the inputted encoder model, and with the final fully connected layer reset.
    """
    # Freeze all layers but the last FC layers
    for name, param in encoder.named_parameters():
        if name not in ['fc1.0.weight', 'fc1.0.bias', 'fc1.2.weight', 'fc1.2.bias']:
            param.requires_grad = False
    return encoder


def add_classification_layers(model, hidden_size, num_of_labels, debug=False):
    """
    Add an additional fully connected layer with a softmax non-linearity on top of the pre-trained MoCo encoder.

    :param model: A pre-trained MoCo encoder Pytorch model. Typically, this is a variant of Resnet.
    :param hidden_size: The size of the hidden layer of our encoder. In Resnet for example, the default hidden dimension
        has size 2048.
    :param num_of_labels: The number of labels in the dataset we are fine-tuning our model on.
    :param debug: Whether or not we are running fine-tuning in a lightweight debug mode.
    :return: The original model, but with an additional linear layer followed by a softmax layer.
    """
    model.fc1 = nn.Linear(hidden_size, num_of_labels)
    model.non_linear_func = nn.Softmax(dim=1)
    # init the fc layer
    model.fc1.weight.data.normal_(mean=0.0, std=0.01)
    model.fc1.bias.data.zero_()
    model.fc1.requires_grad_(True)
    if debug:
        print(model)
    return model


def load_model(dir, filename=None):
    """
    Load a saved pre-trained MoCo encoder.

    :param dir: directory path where pt files and json configuration files exists
    :param filename: the name of the json and pt file with out the suffix
    :return: pre trained model that was loaded from a pt file and the relevant configuration dictionary that was loaded
     from a json file
    """
    assert os.path.exists(dir), "Directory that is supposed to contain pre-trained models does not exist. " \
                                "Please re-run `training_loop.py`"
    if filename is None:  # If filename is none get the latest file
        file_type = '/*pt'
        files = glob.glob(dir + file_type)
        model_path = max(files, key=os.path.getctime)
        json_type = '/*json'
        json_files = glob.glob(dir + json_type)
        json_path = max(json_files, key=os.path.getctime)
    else:
        json_path = os.path.join(dir, filename + consts.MODEL_CONFIGURATION_FILE_ENCODING)
        model_path = os.path.join(dir, filename + consts.MODEL_FILE_ENCODING)

    with open(json_path, "r") as fp:
        config = json.load(fp)

    model = models.Encoder(config[consts.ENCODER_OUTPUT_DIM]).double()
    print(f'model path": {model_path}')
    model.load_state_dict(torch.load(model_path))
    return model, config


def fine_tune(model, train_loader, epochs, lr, momentum, config, gamma=0.9):
    """Fine tune a pre-trained MoCo encoder on the Imagenette image classification dataset.

    :param model: A pre-trained MoCo encoder Pytorch model. Typically, this is a variant of Resnet.
    :param train_loader: `torch.utils.data.DataLoader` instance that provides batches of training data for MoCo to
        fine-tune on.
    :param epochs: The number of training epochs used during fine-tuning.
    :param lr: The initial learning rate used during fine-tuning.
    :param momentum: The momentum value used by the optimizer during fine-tuning.
    :param config: A dictionary specifying the parameters used during pre-training.
    :param gamma: A float that determines the extent of learning rate decay from the perspective of the optimizer.
    :return: The fine-tuned model. Note that the outputted model has a new classifier head relative to the input model.
        The new classifier head of the model predicts imagenette labels.
    """
    wandb.init(project="semi_supervised_cv", entity="zbamberger", config=config)
    # wandb.init(project="semi_supervised_cv", entity="noambenmoshe", config=config)
    wandb.watch(model)
    loss_fn = nn.CrossEntropyLoss()

    model = freeze_encoder_init_last_fc(model)
    model = add_classification_layers(model,
                                      hidden_size=consts.HIDDEN_REPRESENTATION_DIM,
                                      num_of_labels=consts.NUM_OF_CLASSES)

    # Filter out model parameters that don't require gradient updates.
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum)
    scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=gamma)

    device = torch.device(consts.CUDA if torch.cuda.is_available() else consts.CPU)
    model = model.double().to(device)
    acc = []
    loss = []

    for epoch in range(epochs):
        print(f'start epoch {epoch}')
        for minibatch, lables in train_loader:
            minibatch = minibatch.double().to(device)
            optimizer.zero_grad()

            output = model.forward(minibatch, device)  # Output shape is [batch_size,number_of_classes]
            loss_minibatch = loss_fn(output, lables.to(torch.int64).to(device))
            loss.append(loss_minibatch)
            preds = torch.argmax(output, dim=1)
            optimizer.zero_grad()
            loss_minibatch.backward()
            optimizer.step()
            acc1 = torch.eq(preds, lables.to(device)).sum().float().item() / preds.shape[0]
            acc.append(acc1)
            wandb.log({consts.MINI_BATCH_LOSS: loss_minibatch,
                       consts.MINI_BATCH_ACCURACY: acc1})
            print(f'\t{consts.MINI_BATCH_INDEX} = {epoch}\t'
                  f'{consts.MINI_BATCH_LOSS} = {loss_minibatch}'
                  f' {consts.MINI_BATCH_ACCURACY} = {acc1}')

        scheduler.step()
        avg_acc = sum(acc) / len(acc)
        avg_loss = sum(loss) / len(loss)
        wandb.log({consts.EPOCH_LOSS: avg_loss,
                   consts.EPOCH_ACCURACY: avg_acc})
        print(f'{consts.EPOCH_INDEX} = {epoch}\t'
              f'{consts.EPOCH_LOSS} = {avg_loss}\t'
              f'{consts.EPOCH_ACCURACY} = {avg_acc}')
    return model


def evaluate(fine_tuned_model, validation_loader, device):
    """Evaluate the performance of a MoCo encoder that is pre-trained and then fine-tuned on Imagenette.
    :param fine_tuned_model: A fine-tuned MoCo encoder after it has been pre-trained. A torch model.
    :param validation_loader: A DataLoader instance containing a verification subset of the Imagenette dataset.
    :param device: A torch.device.Device instance representing the device on which downstream evaluation is run.
    :return: The average accuracy of the fine-tuned model on the verification set of Imagenette.
    """
    acc = []
    print('Running fine-tuning evaluation...')
    for minibatch, lables in validation_loader:
        minibatch = minibatch.double()
        with torch.no_grad():
            output = fine_tuned_model.forward(minibatch, device)  # Output shape is [batch_size,number_of_classes]

        preds = torch.argmax(output, dim=1)
        acc1 = torch.eq(preds, lables).sum().float().item() / preds.shape[0]
        acc.append(acc1)

        print(f'\t{consts.MINI_BATCH_ACCURACY} = {acc1}')

    avg_acc = sum(acc) / len(acc)
    print(f'validation avg_Acc = {avg_acc}')
    return avg_acc


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    debug = args.fine_tuning_debug
    epochs = 5 if debug else args.fine_tuning_epochs
    lr = 10 if debug else args.fine_tuning_learning_rate
    momentum = args.fine_tuning_momentum,
    if not os.path.exists(consts.validation_filename):
        data_loader.create_csv_file(dir=consts.image_dir_validation, filename=consts.validation_filename)
    pre_trained_model, config = load_model(dir=consts.SAVED_ENCODERS_DIR, filename=args.pretrained_encoder_file_name)
    set_seed(1 if debug else config[consts.SEED])

    imagenette_dataset = ImagenetteDataset(csv_file=consts.csv_filename,
                                           root_dir=consts.image_dir,
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor(),
                                               normalize
                                           ]),
                                           labels=True,
                                           debug=debug)
    train_loader = DataLoader(imagenette_dataset, batch_size=args.fine_tuning_batch_size, shuffle=True)
    imagenette_dataset_validation = ImagenetteDataset(csv_file=consts.validation_filename,
                                                      root_dir=consts.image_dir_validation,
                                                      transform=transforms.Compose([
                                                          Rescale(256),
                                                          RandomCrop(224),
                                                          ToTensor(),
                                                          normalize
                                                      ]),
                                                      labels=True,
                                                      debug=debug)
    validation_loader = DataLoader(imagenette_dataset_validation,
                                   batch_size=args.fine_tuning_batch_size,
                                   shuffle=True)
    # Note: momentum is perceived as a tuple of floats with length 1. The momentum value is the sole entry in this
    # tuple.
    fine_tuned_model = fine_tune(pre_trained_model, train_loader, epochs, lr, momentum[0], config)
    device = torch.device(consts.CUDA if torch.cuda.is_available() else consts.CPU)
    avg_acc_val = evaluate(fine_tuned_model, validation_loader, device)

    if avg_acc_val > 0.8:
        if not os.path.exists(consts.SAVED_FINE_TUNED_ENCODERS_DIR):
            os.mkdir(consts.SAVED_FINE_TUNED_ENCODERS_DIR)
        main_name = "_".join(["fine_tuned",
                              "debug",
                              str(debug),
                              consts.RESNET_50,
                              str(config[consts.FINE_TUNING_EPOCHS]),
                              consts.FINE_TUNING_EPOCHS,
                              str(config[consts.FINE_TUNING_LEARNING_RATE]).replace(".", "_"),
                              consts.FINE_TUNING_LEARNING_RATE,
                              str(config[consts.FINE_TUNING_BATCH_SIZE]),
                              consts.FINE_TUNING_BATCH_SIZE,
                              'avg_acc_val',
                              str(avg_acc_val)])

        file_name = main_name + consts.MODEL_FILE_ENCODING
        torch.save(fine_tuned_model.state_dict(), os.path.join(consts.SAVED_FINE_TUNED_ENCODERS_DIR, file_name))
        config_path = os.path.join(consts.SAVED_FINE_TUNED_ENCODERS_DIR,
                                   main_name + consts.MODEL_CONFIGURATION_FILE_ENCODING)
        print(f'Saved pre-trained model to {config_path}')
