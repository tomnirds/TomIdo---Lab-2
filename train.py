import torch
from torch import nn
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import torch.nn.functional as F
from vizzes import *
import time


def dsc_criterion(scores_for_real, scores_for_fake, real_labels, fake_labels, criterion):
    D_loss_real = criterion(scores_for_real.squeeze(), real_labels)
    D_loss_fake = criterion(scores_for_fake.squeeze(), fake_labels)
    D_loss = D_loss_real + D_loss_fake
    return D_loss


def train_cgan(G, D, data_loader, num_epochs, device, checkpoint_start=299):
    D_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    G_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    print_every = 10
    produce_every = 100
    img_dim = G.dim
    z_dim = G.z_dim
    y_dim = G.y_dim

    # Label preprocess
    fill = torch.zeros([y_dim, y_dim, img_dim, img_dim])
    for index in range(y_dim):
        fill[index, index, :, :] = 1

    # fixed noise & label for visualizations
    #TODO: change to 10,10 matrix
    togen_latent = torch.randn(100, z_dim).view(-1, z_dim, 1, 1)
    togen_labels = torch.cat([torch.full((1, 10), i) for i in range(10)]).view(-1)
    togen_labels = F.one_hot(togen_labels).view(-1, y_dim, 1, 1).type(torch.FloatTensor)

    criterion = nn.BCELoss()

    D_avg_losses = []
    G_avg_losses = []
    start_time = time.time()
    times = []
    for epoch in range(num_epochs):
        print(f"EPOCH {epoch + 1}")
        D_losses = []
        G_losses = []

        # LR adjustments
        #if epoch == 5 or epoch == 10:
            #G_optimizer.param_groups[0]['lr'] *= 2
            #D_optimizer.param_groups[0]['lr'] *= 2

        for index, (real_images, let_labels) in enumerate(data_loader):

            # image data
            batch_size = real_images.shape[0]
            real_images = real_images.to(device)

            # labels
            real_labels = torch.ones(batch_size).to(device)
            fake_labels = torch.zeros(batch_size).to(device)
            let_labels_preprocessed = fill[let_labels].to(device)

            # Discriminator Training
            D_real_scores = D(real_images, let_labels_preprocessed)
            fake_atts = (torch.rand(batch_size, 1) * y_dim).type(torch.LongTensor).squeeze()
            fake_atts_onehot = F.one_hot(fake_atts).view(batch_size, y_dim, 1, 1).type(torch.FloatTensor).to(device)
            gen_image = G.sample(batch_size, fake_atts_onehot)
            let_labels_preprocessed = fill[fake_atts].to(device)
            D_fake_scores = D(gen_image, let_labels_preprocessed)

            D_loss = dsc_criterion(D_real_scores, D_fake_scores, real_labels, fake_labels, criterion)
            D.zero_grad()
            D_loss.backward()
            D_optimizer.step()

            # Generator Training
            fake_atts = (torch.rand(batch_size, 1) * y_dim).type(torch.LongTensor).squeeze()
            fake_atts_onehot = F.one_hot(fake_atts).view(batch_size, y_dim, 1, 1).type(torch.FloatTensor).to(device)
            gen_image = G.sample(batch_size, fake_atts_onehot)

            let_labels_preprocessed = fill[fake_atts].to(device)
            D_fake_scores = D(gen_image, let_labels_preprocessed).squeeze()
            G_loss = criterion(D_fake_scores, real_labels)

            G.zero_grad()
            G_loss.backward()
            G_optimizer.step()

            # loss values
            D_losses.append(D_loss.item())
            G_losses.append(G_loss.item())
            if (index + 1) % print_every == 0:
                print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f'
                      % (epoch + 1, num_epochs, index + 1, len(data_loader), D_loss.item(), G_loss.item()))
            if (index + 1) % produce_every == 0:
                plot_result(G, togen_latent, togen_labels, epoch, save=True)

        D_avg_loss = torch.mean(torch.FloatTensor(D_losses))
        G_avg_loss = torch.mean(torch.FloatTensor(G_losses))
        D_avg_losses.append(D_avg_loss.item())
        G_avg_losses.append(G_avg_loss.item())
        if epoch%2 == 0:
            plot_result(G, togen_latent, togen_labels, epoch, save=True)
        times.append(time.time() - start_time)
        # plot_loss(D_avg_losses, G_avg_losses, times, save=True, save_dir='figs/cp_')
        if epoch > checkpoint_start:
            print("Saving Checkpoint!")
            torch.save(D.state_dict(), 'dsc_cropped_cp.pkl')
            torch.save(G.state_dict(), 'gen_cropped_cp.pkl')
            pickle.dump(D_avg_losses, open('dsc_loss_cropped_cp.pkl', 'wb'))
            pickle.dump(G_avg_losses, open('gen_loss_cropped_cp.pkl', 'wb'))
            plot_loss(D_avg_losses, G_avg_losses, times, save=True, save_dir='figs/cp_')

    print("Saving!")
    torch.save(D.state_dict(), 'dsc_cropped.pkl')
    torch.save(G.state_dict(), 'gen_cropped.pkl')
    pickle.dump(D_avg_losses, open('dsc_loss_cropped.pkl', 'wb'))
    pickle.dump(G_avg_losses, open('gen_loss_cropped.pkl', 'wb'))

    plot_loss(D_avg_losses, G_avg_losses, times, save=True, save_dir='figs/')
    plot_result(G, togen_latent, togen_labels, 20, save=True)
    print("Done!")