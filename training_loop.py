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


wandb.init(project="semi_supervised_cv", entity="zbamberger")
# wandb.init(project="semi_supervised_cv", entity="noambenmoshe")

parser = argparse.ArgumentParser(
    description='Process flags for unsupervised pre-training with MoCo.')
parser.add_argument('--pre_training_debug',
                    type=bool,
                    default=True,
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
parser.add_argument('--number_of_keys',
                    type=int,
                    default=8,
                    help='The number of keys used during MoCo\'s pre-training. As the number of keys increases, '
                         'the number of adversarial candidates during pre-training increases. Thus, as the number of'
                         'keys increases, so does the model\'s output space and problem difficulty.')
parser.add_argument('--encoder_output_dim',
                    type=int,
                    default=128,
                    help='The encoder\'s output dim')
parser.add_argument('--temperature',
                    type=int,
                    default=0.07,
                    help='The temperature used in the Contrastive loss')
parser.add_argument('--m',
                    type=int,
                    default=0.999,
                    help='The momentume used to update the key\'s encoder parameters')


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
              number_of_keys=3):
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
    wandb.run.define_metric(
        name="mini-batch accuracy",
        step_metric="mini-batch index",
        goal="maximize")
    wandb.run.define_metric(
        name="mini-batch loss",
        step_metric="mini-batch index",
        goal="minimize")
    wandb.run.define_metric(
        name="epoch accuracy",
        step_metric="epoch index",
        goal="maximize")
    wandb.run.define_metric(
        name="epoch loss",
        step_metric="epoch index",
        goal="minimize")
    queue_dict = []  # Will add in FIFO order keys of mini batches
    for i in range(number_of_keys):
        # 2048 is the output dimension of Resnet50
        queue_dict.append(torch.rand(encoder.final_num_of_features))
    loss_fn = nn.BCEWithLogitsLoss()

    # The optimization is done only to the encoder weights and not to the momentum encoder
    optimizer = torch.optim.SGD(encoder.parameters(), lr=lr, momentum=momentum)

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
            q = encoder.forward(x_q)  # Queries have shape [N, C] where C = 2048 * 1 * 128
            k = m_encoder.forward(x_k)  # Keys have shape [N, C]
            k = k.detach()  # No gradients to keys

            # Positive logits have shape [N, 1]
            q = torch.flatten(q, start_dim=1)
            q = torch.unsqueeze(q, dim=1)
            k = torch.flatten(k, start_dim=1)
            k = torch.unsqueeze(k, dim=2)
            l_pos = (q @ k).squeeze(dim=2)

            queue_view = torch.concat([queue_dict[i].unsqueeze(dim=1) for i in range(len(queue_dict))], 1)
            q = torch.squeeze(q, dim=1)

            l_neg = q @ queue_view.double() # Negative logits are a tensor of shape [N, K]
            logits = torch.concat((l_pos, l_neg), dim=1)  # Lots have shape [N, K + 1]
            labels = torch.zeros(l_pos.shape[0])
            one_hot_labels = torch.nn.functional.one_hot(labels.to(torch.int64), num_classes=logits.shape[1])
            loss = loss_fn(logits / t, one_hot_labels.double())
            epoch_loss.append(loss)
            preds = torch.argmax(input=logits, dim=1)
            accuracy = (torch.sum(preds == labels) / logits.shape[0])
            epoch_acc.append(accuracy)
            wandb.log({"epoch index": epoch,
                       "mini-batch index": minibatch_index,
                       "mini-batch loss": loss,
                       "mini-batch accuracy": accuracy})
            
            if batch_index % 5 == 0:
                print(f'\tBatch Index = {batch_index},\tBatch Loss = {loss},\tBatch Accuracy = {accuracy}')

            # SGD update query network
            loss.backward()
            optimizer.step()  # Update only encoder parmas and not m_encoder params

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
        wandb.log({"epoch index": epoch,
                   "epoch loss": epoch_loss,
                   "epoch accuracy": epoch_acc})
    print('Finished pre-training!')
    return encoder


def set_seed(seed=42):
    """
    :param seed:
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
    number_of_keys = args.number_of_keys
    encoder_output_dim = args.encoder_output_dim
    t = args.temperature
    m = args.m

    assert bs % number_of_keys == 0, f'{bs} is not divisible by {number_of_keys}.\n' \
                                     f'Choose a different batch size so it will be a multiple of the number of keys.'

    config = vars(args)
    print(config)

    set_seed(seed)

    imagenette_dataset = ImagenetteDataset(csv_file=consts.csv_filename,
                                           root_dir=consts.image_dir,
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()
                                           ]),
                                           debug=debug)

    train_loader = DataLoader(imagenette_dataset, batch_size=bs, shuffle=True)
    # TODO: set the m_encoder to be a copy of the encoder.
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
                        epochs=epochs,
                        lr=lr,
                        momentum=momentum,
                        number_of_keys=number_of_keys,
                        t= t,
                        m= m)
    # Freeze the encoder
    encoder.requires_grad_(False)

    # Save model state
    config_dict = {}
    for k in vars(args):
        config_dict[k] = config[k]

    if not os.path.exists(consts.SAVED_ENCODERS_DIR):
        os.mkdir(consts.SAVED_ENCODERS_DIR)
    main_name = "_".join([consts.RESNET_50, str(epochs), consts.EPOCHS, str(lr).replace(".", "_"), 'lr'])
    file_name = main_name + consts.MODEL_FILE_ENCODING
    torch.save(encoder.state_dict(), os.path.join(consts.SAVED_ENCODERS_DIR, file_name))
    config_path = os.path.join(consts.SAVED_ENCODERS_DIR, main_name + consts.MODEL_CONFIGURATION_FILE_ENCODING)
    with open(config_path, 'w') as fp:
        json.dump(config_dict, fp, indent=4)
