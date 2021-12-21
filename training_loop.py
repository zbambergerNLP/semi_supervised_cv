import torch
import torch.nn as nn

import models
from augment import load_imagenette



# train function
def train(encoder,m_encoder, train_loader, epochs=20, lr=0.001, momentum=0.9, t = 0.07, m = 0.999):

    dict = {}
    queue_dict = [] #will add in FIFO order keys of mini batches
    loss_fn = nn.CrossEntropyLoss

    #the optimization is done only to the encoder weights and not to the momentum encoder
    optimizer = torch.optim.SGD(encoder.parameters(), lr=lr, momentum=momentum)


    for epoch in range(epochs):
        print(f'start epoch {epoch}')

        for x, y  in train_loader:
            optimizer.zero_grad()
            #TODO call augmentaion function

            q = encoder.forward(x) #queries NxC
            k = m_encoder.forward(x) #keys NxC
            k = k.detach() # no gradients to keys

            #positive logits Nx1
            l_pos = q @ k

            queue_view = torch.concat([queue_dict[x] for x in queue_dict], 1)

            #negative logits NxK
            l_neg = q @ queue_view

            logits = torch.concat((l_pos,l_neg), dim= 0) #Nx(k+1)

            labels = torch.zeros(l_pos.shape[0])
            labels[0] = 1 #first logit is  from the same image so the label is 1

            loss = loss_fn(logits / t , labels)
            #SGD update query network
            loss.backward()
            optimizer.step() #update only encoder parmas and not m_encoder params #TODO make sure that this is what happens

            #momentum update key network
            m_encoder_state_dict = m_encoder.state_dict()
            encoder_state_dict = encoder.state_dict()

            for m_name, m_param, name, param in zip(m_encoder_state_dict.items(), encoder_state_dict.item()):
                # Transform the parameter as required.
                transformed_param = m * m_param  + (1-m) * param

                # Update the parameter.
                m_encoder_state_dict[m_name].copy_(transformed_param)

            #enqueue queue and queue dict
            queue_dict.append(k)

            minibatch_dict = {}












# evalutate function top-1 accuracy with linear evaluation

# data loaders function

# save model

# load model

if __name__ == '__main__':
    seed = 1
    epochs = 2
    learning_rate = 0.1
    momentum = 0.9
    set_seed(seed)

    train_loader = load_imagenette() #TODO add batch size

    encoder = models.Encoder()
    m_endcoder  = models.Encoder()

    # train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    m_endcoder.to(device)

    train(encoder, m_encoder, train_loader, epochs=epochs,lr = learning_rate,  momentum= momentum)

