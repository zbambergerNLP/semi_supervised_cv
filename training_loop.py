import torch
import torch.nn as nn
import random
import numpy as np

import consts
import models
import matplotlib.pyplot as plt
from augment import load_imagenette
from data_loader import ImagenetteDataset, Rescale, RandomCrop,ToTensor
from torch.utils.data import DataLoader
from torchvision import transforms, utils

# train function
def train(encoder,m_encoder, train_loader, epochs=20, bs = 8, lr=0.001, momentum=0.9, t = 0.07, m = 0.999, number_of_keys=3):

    dict = {} #keys are the ouptut
    queue_dict = [ ] # will add in FIFO order keys of mini batches
    for i in range(number_of_keys):
        queue_dict.append(torch.rand(262144)) #TODO think of a better way than hard coded
    loss_fn = nn.BCEWithLogitsLoss()

    #the optimization is done only to the encoder weights and not to the momentum encoder
    optimizer = torch.optim.SGD(encoder.parameters(), lr=lr, momentum=momentum)


    for epoch in range(epochs):
        print(f'start epoch {epoch}')

        for x in train_loader:
            x = x.double()
            optimizer.zero_grad()
            #TODO call augmentaion function

            q = encoder.forward(x) #queries NxC
            k = m_encoder.forward(x) #keys NxC
            k = k.detach() # no gradients to keys

            #positive logits Nx1
            q = torch.flatten(q, start_dim=1)
            q = torch.unsqueeze(q, dim=1)
            k = torch.flatten(k, start_dim=1)
            k = torch.unsqueeze(k, dim=2)
            l_pos = (q @ k).squeeze(dim = 2)


            queue_view = torch.concat([queue_dict[i].unsqueeze(dim =1) for i in range(len(queue_dict))], 1)
            q = torch.squeeze(q, dim=1)
            #negative logits NxK
            l_neg = q @ queue_view.double()
            logits = torch.concat((l_pos, l_neg), dim=1)  # Nx(k+1)


            labels = torch.zeros(l_pos.shape[0])
            one_hot_labels = torch.nn.functional.one_hot(labels.to(torch.int64), num_classes= logits.shape[1])

            loss = loss_fn(logits / t, one_hot_labels.double())
            #SGD update query network
            loss.backward()
            optimizer.step() #update only encoder parmas and not m_encoder params #TODO make sure that this is what happens

            #momentum update key network
            m_encoder_state_dict = m_encoder.state_dict()
            encoder_state_dict = encoder.state_dict()


            for m_name, m_param in m_encoder_state_dict.items():
                # Transform the parameter as required.
                transformed_param = m * m_param + (1 - m) * encoder_state_dict[m_name]
                # Update the parameter.
                m_encoder_state_dict[m_name].copy_(transformed_param)

            for i in range(k.shape[0]):
                # enqueue queue and queue dict
                queue_dict.append(k[i].squeeze(dim=1))
                # dequeue the oldest mini batch
                queue_dict.pop(0)


# evalutate function top-1 accuracy with linear evaluation

# save model

# load model

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    seed = 1
    epochs = 2
    lr = 0.1
    momentum = 0.9
    bs = 8
    number_of_keys = 8
    assert number_of_keys % bs ==0 ,f'Choose a different number of keys so it will be a multiple of the batch size'

    set_seed(seed)

    imagenette_dataset = ImagenetteDataset(csv_file=consts.csv_filename,
                                        root_dir=consts.image_dir,
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()
                                           ]))

    train_loader = DataLoader(imagenette_dataset, batch_size=bs)

    # fig = plt.figure()
    #
    # for i in range(len(imagenette_dataset)):
    #     sample = imagenette_dataset[i]
    #
    #     print(i, sample.shape)
    #
    #     ax = plt.subplot(1, 4, i + 1)
    #     plt.tight_layout()
    #     ax.set_title('Sample #{}'.format(i))
    #     ax.axis('off')
    #
    #
    #     if i == 3:
    #         plt.show()
    #         break

    encoder = models.Encoder().double()
    m_endcoder  = models.Encoder().double()

    # train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    m_endcoder.to(device)

    train(encoder, m_endcoder, train_loader, epochs=epochs,bs = bs,lr = lr,  momentum= momentum, number_of_keys = number_of_keys)

