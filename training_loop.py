import os

import torch
import torch.nn as nn
import random
import numpy as np
import json
import consts

import models
from data_loader import ImagenetteDataset, Rescale, RandomCrop, ToTensor
from torch.utils.data import DataLoader
from torchvision import transforms
from augment import augment
import argparse
import wandb
import matplotlib.pyplot as plt

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
parser = argparse.ArgumentParser(
    description='Process flags for unsupervised pre-training with MoCo.')
parser.add_argument('--pre_training_debug',
                    type=bool,
                    default=False,
                    required=False,
                    help="Whether or not to run pre-training in debug mode. In debug mode, the model learns over "
                         "a subset of the original dataset.")
parser.add_argument('--seed',
                    type=int,
                    default=1,
                    required=False,
                    help='The seed used for random sampling.')
parser.add_argument('--pretraining_epochs',
                    type=int,
                    default=3,
                    required=False,
                    help='The number of epochs used during pre-training.')
parser.add_argument('--pretraining_learning_rate',
                    type=float,
                    default=1e-2,
                    help='The initial learning rate used during pre-training.')
parser.add_argument('--momentum',
                    type=float,
                    default=0.9,
                    help='The momentum value used to transfer weights between encoders during pre-training.')
parser.add_argument('--pretraining_batch_size',
                    type=int,
                    default=64,
                    help='The mini-batch size used during pre-training with MoCo. Keys and queries are generated '
                         'from the entries of a mini-batch.')
parser.add_argument('--mul_for_num_of_keys',
                    type=int,
                    default=4,
                    help="The number of keys is a multiple of batch size times this value.")
parser.add_argument('--encoder_output_dim',
                    type=int,
                    default=128,
                    help='The encoder\'s output dim')
parser.add_argument('--temperature',
                    type=float,
                    default=0.07,
                    help='The temperature used in the Contrastive loss')
parser.add_argument('--m',
                    type=float,
                    default=0.999,
                    help='The momentum used to update the key\'s encoder parameters')

# Sample run from server command line:
# srun python3 training_loop.py --pre_training_debug False --seed 2 --pretraining_epochs 100 \
# --pretraining_learning_rate 0.001 --number_of_keys 64 --pretraining_batch_size 256

# TODO: Document all functions and classes in this repository.


# Train function
def pre_train(encoder,
              m_encoder,
              train_loader,
              epochs=3,
              lr=0.001,
              momentum=0.9,
              t=0.07,
              m=0.999,
              number_of_keys=3,
              debug=False):
    """
    :param encoder:
    :param m_encoder:
    :param epochs:
    :param lr:
    :param momentum:
    :param t:
    :param m:
    :param number_of_keys:
    :return:
    """
    wandb.watch(encoder)
    queue_dict = []  # Will add in FIFO order keys of mini batches
    for i in range(number_of_keys):
        # 2048 is the output dimension of Resnet50
        queue_dict.append(torch.rand(encoder.final_num_of_features))
    loss_fn = nn.BCEWithLogitsLoss()

    # The optimization is done only to the encoder weights and not to the momentum encoder
    optimizer = torch.optim.SGD(encoder.parameters(), lr=lr, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[120, 160], gamma=0.1)

    for epoch in range(epochs):
        print(f'start epoch {epoch}')

        batch_index = 0
        epoch_loss = []
        epoch_acc = []
        for minibatch_index, minibatch in enumerate(train_loader):

            minibatch = minibatch.double()
            optimizer.zero_grad()
            x_q = augment(images=minibatch,
                          jitter_prob=0.8,
                          horizontal_flip_prob=0.5,
                          grayscale_conversion_prob=0.2,
                          gaussian_blur_prob=0.5)
            x_k = augment(images=minibatch,
                          jitter_prob=0.8,
                          horizontal_flip_prob=0.5,
                          grayscale_conversion_prob=0.2,
                          gaussian_blur_prob=0.5)

            if debug:
                if batch_index % 5 == 0:
                    # plt.figure()
                    # f, ax = plt.subplots(2, 1)
                    # ax[0].imshow(x_q[0].T)
                    # ax[1].imshow(x_k[0].T)
                    # plt.show()
                    pass

            q = encoder.forward(x_q)  # Queries have shape [N, C] where C = 2048 * 1 * 128

            with torch.no_grad():
                k = m_encoder.forward(x_k)  # Keys have shape [N, C]
                k = k.detach()  # No gradients to keys

            # Positive logits have shape [N, 1]
            q = torch.flatten(q, start_dim=1)
            q = torch.unsqueeze(q, dim=1)
            k = torch.flatten(k, start_dim=1)
            k = torch.unsqueeze(k, dim=2)
            l_pos = (q @ k).squeeze(dim=2)

            queue_view = torch.concat([queue_dict[i].unsqueeze(dim=1) for i in range(len(queue_dict))], 1)
            queue_view.detach()
            q = torch.squeeze(q, dim=1)

            l_neg = q @ queue_view.double() # Negative logits are a tensor of shape [N, K]
            logits = torch.concat((l_pos, l_neg), dim=1)  # Lots have shape [N, K + 1]
            labels = torch.zeros(l_pos.shape[0])
            one_hot_labels = torch.nn.functional.one_hot(labels.to(torch.int64), num_classes=logits.shape[1])
            loss = loss_fn(logits / t, one_hot_labels.double())
            epoch_loss.append(loss)
            preds = torch.argmax(input=logits, dim=1)
            accuracy = (torch.sum(preds == labels) / logits.shape[0])
            wandb.log({'mini-batch loss': loss,
                       'mini-batch accuracy': accuracy})
            epoch_acc.append(accuracy)
            if batch_index % 5 == 0:
                print(f'\tBatch Index = {batch_index},\tBatch Loss = {loss},\tBatch Accuracy = {accuracy}')

            # SGD update query network
            loss.backward()
            optimizer.step()  # Update only encoder parmas and not m_encoder params
            scheduler.step()

            with torch.no_grad():  # no gradient to keys
                # Momentum update key network
                m_encoder_state_dict = m_encoder.state_dict()
                encoder_state_dict = encoder.state_dict()
                for m_name, m_param in m_encoder_state_dict.items():
                    # Transform the parameter as required.
                    transformed_param = m * m_param + (1 - m) * encoder_state_dict[m_name]
                    # Update the parameter.
                    m_encoder_state_dict[m_name].copy_(transformed_param)

                for i in range(k.shape[0]):
                    # Enqueue queue and queue dict
                    queue_dict.append(k[i].squeeze(dim=1))
                    # Dequeue the oldest mini batch
                    queue_dict.pop(0)

            batch_index += 1
        epoch_loss = sum(epoch_loss) / len(epoch_loss)
        epoch_acc = sum(epoch_acc) / len(epoch_acc)
        print(f'Epoch #:{epoch},\tEpoch Average Loss: {epoch_loss},\tEpoch Average Accuracy: {epoch_acc}')
        # wandb.log({"epoch loss": epoch_loss,
        #            "epoch accuracy": epoch_acc})
    print('Finished pre-training!')
    return encoder


def set_seed(seed=42):
    """
    :param seed: The integer seed used for random number generation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    args = parser.parse_args()
    debug = args.pre_training_debug
    seed = args.seed
    epochs = args.pretraining_epochs
    lr = args.pretraining_learning_rate
    momentum = args.momentum
    bs = args.pretraining_batch_size
    mul_for_num_of_keys = args.mul_for_num_of_keys
    # number_of_keys = args.number_of_keys
    encoder_output_dim = args.encoder_output_dim
    t = args.temperature
    m = args.m

    config_args = {'pretraining_epochs': epochs,
                   'pretraining_learning_rate': lr,
                   'momentum': momentum,
                   'pretraining_batch_size': bs,
                   consts.MUL_FOR_NUM_KEYS: mul_for_num_of_keys,
                   consts.ENCODER_OUTPUT_DIM: encoder_output_dim,
                   consts.TEMPERATURE: t,
                   'm': m,
                   consts.SEED: seed}

    #wandb.init(project="semi_supervised_cv", entity="zbamberger", config=config_args)
    wandb.init(project="semi_supervised_cv", entity="noambenmoshe", config=config_args)
    config = wandb.config

    number_of_keys = config.mul_for_num_of_keys * config.pretraining_batch_size
    assert number_of_keys % config.pretraining_batch_size == 0, f'{number_of_keys} is not divisible by '\
                                                                f'{config.pretraining_batch_size}.\n'
    print(config)
    set_seed(seed)
    imagenette_dataset = ImagenetteDataset(csv_file=consts.csv_filename,
                                           root_dir=consts.image_dir,
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor(),
                                               normalize
                                           ]),
                                           debug=debug)

    train_loader = DataLoader(imagenette_dataset, batch_size=config.pretraining_batch_size, shuffle=True)

    encoder = models.Encoder(encoder_output_dim).double()
    m_endcoder = models.Encoder(encoder_output_dim).double()

    # Train model
    device = torch.device(consts.CUDA if torch.cuda.is_available() else consts.CPU)
    encoder.to(device)
    m_endcoder.to(device)

    # initialize  parameters in both encoders to be the same
    for param, m_param in zip(encoder.parameters(), m_endcoder.parameters()):
        m_param.data.copy_(param.data)

    encoder = pre_train(encoder,
                        m_endcoder,
                        train_loader,
                        epochs=config.pretraining_epochs,
                        lr=config.pretraining_learning_rate,
                        momentum=config.momentum,
                        number_of_keys=number_of_keys,
                        t=config.temperature,
                        m=config.m,
                        debug=debug)
    # Freeze the encoder
    encoder.requires_grad_(False)

    # Save model state
    config_dict = {}
    for k in config.keys():
        config_dict[k] = config[k]

    if not os.path.exists(consts.SAVED_ENCODERS_DIR):
        os.mkdir(consts.SAVED_ENCODERS_DIR)
    main_name = "_".join([consts.RESNET_50,
                          str(config.pretraining_epochs),
                          consts.EPOCHS,
                          str(config.pretraining_learning_rate).replace(".", "_"),
                          consts.LEARNING_RATE])
    file_name = main_name + consts.MODEL_FILE_ENCODING
    torch.save(encoder.state_dict(), os.path.join(consts.SAVED_ENCODERS_DIR, file_name))
    config_path = os.path.join(consts.SAVED_ENCODERS_DIR, main_name + consts.MODEL_CONFIGURATION_FILE_ENCODING)
    with open(config_path, 'w') as fp:
        json.dump(config_dict, fp, indent=4)
