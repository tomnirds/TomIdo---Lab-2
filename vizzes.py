import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
import numpy as np
import imageio
import pandas as pd
import glob
from sklearn.manifold import TSNE
from cdcgan import Generator
import torch.nn.functional as F
import torch
import seaborn as sns;

sns.set()


#Possible Signature - generate 3 new 'I' pictures and save them inside 'newly_generated' folder
#vizzes.create_n_pictures(3, 0, 'newly_generated')
def create_n_pictures(amount, letter_num, tostore_folder, model_folder=None):
    z_dim = 100
    y_dim = 10
    img_dim = 32
    gen_kw = {'z_dim': z_dim, 'y_dim': y_dim, 'img_dim': img_dim}
    G = Generator(**gen_kw)
    if model_folder:
        path = model_folder + '/gen_cropped.pkl'
    else:
        path = 'gen_cropped.pkl'
    G.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    togen_latent = torch.randn(amount, z_dim).view(-1, z_dim, 1, 1)
    togen_labels = torch.cat([torch.range(0, 9).view(-1, 1), torch.full((amount, 1), letter_num)]).view(-1).type(torch.LongTensor)
    togen_labels = F.one_hot(togen_labels).view(-1, y_dim, 1, 1).type(torch.FloatTensor)
    togen_labels = togen_labels[10:]
    for i, (z_cont, z_const) in enumerate(zip(togen_latent, togen_labels)):
        gen_image(tostore_folder, G, z_cont, z_const, i)

def show_some_imgs(dataloader):
    """
    Show one batch of images using the dataloader
    """
    x, y = next(iter(dataloader))
    for i in range(x.shape[0]):
        plt.imshow(x[i].permute(1, 2, 0))
        plt.title(f"Sample")
        plt.axis('off')
        #plt.savefig(f'foo{i}.png', bbox_inches='tight')
        plt.show()



# Image denormalizing
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def plot_loss(d_losses, g_losses, times, save_dir, save=False):
    fig, ax = plt.subplots()
    ax.set_xlim(0, max(times) / 60)
    ax.set_ylim(0, max(np.max(g_losses), np.max(d_losses)) * 1.1)
    res_df = pd.DataFrame(
        {'Discriminator': d_losses, 'Generator': g_losses, 'Training Time': [time / 60 for time in times]})
    sns.lineplot(data=res_df, x='Training Time', y='Discriminator', ax=ax, label='Discriminator')
    sns.lineplot(data=res_df, x='Training Time', y='Generator', ax=ax, label='Generator').set(
        title=f'Discriminator and Generator Loss', ylabel='Loss')
    plt.legend(loc="lower right")

    # save figure
    if save:
        save_fn = save_dir + 'closses.png'
        plt.savefig(save_fn)
    else:
        plt.show()
    plt.close()


def plot_result(generator, noise, label, num_epoch, save=False, save_dir='figs/', name=None, fig_size=(5, 5)):
    generator.eval()
    device = next(generator.parameters()).device
    noise = noise.to(device)
    label = label.to(device)
    gen_image = generator(noise, label)
    gen_image = denorm(gen_image)

    generator.train()

    n_rows = np.sqrt(noise.size()[0]).astype(np.int32)
    n_cols = np.sqrt(noise.size()[0]).astype(np.int32)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
    for ax, img in zip(axes.flatten(), gen_image):
        ax.axis('off')
        ax.set_adjustable('box')
        # Scale to 0-255
        img = (((img - img.min()) * 255) / (img.max() - img.min())).cpu().data.numpy().transpose(1, 2, 0).astype(
            np.uint8)
        ax.imshow(img, cmap=None, aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    title = 'Epoch {0}'.format(num_epoch + 1)
    fig.text(0.5, 0.04, title, ha='center')
    print("produced!")

    # save figure
    if save:
        if name:
            save_fn = save_dir + name + '.png'
        if not name:
            save_fn = save_dir + 'cbin_epoch_{:d}'.format(num_epoch + 1) + '.png'
        plt.savefig(save_fn)
    else:
        plt.show()
    plt.close()


def create_gif(fig_folder):
    all_figs = glob.glob(fig_folder+'/*')
    images = {}
    for filename in all_figs:
        index = int(filename.split('_')[-1].split('.')[0])
        images[(index-1)*2] = imageio.imread(filename)
        images[(index-1)*2+1] = imageio.imread(filename)
    images_sorted = list(dict(sorted(images.items())).values())
    imageio.mimsave('trainingif.gif', images_sorted)


def gen_image (fig_folder, G , z_cont,z_const, i):
    """
    Generate a single image and save it for later use.
    """
    plt.figure(figsize=(1.2, 1.2))
    img = denorm(G(z_cont.unsqueeze(0), z_const.unsqueeze(0))).squeeze()
    img = (((img - img.min()) * 255) / (img.max() - img.min())).cpu().data.numpy().transpose(1, 2, 0).astype(
        np.uint8)
    plt.imshow(img, cmap=None, aspect='equal')
    plt.axis('off')
    plt.grid(b=None)
    save_fn = fig_folder + f'/pic_{i}.png'
    plt.savefig(save_fn, bbox_inches='tight', pad_inches=0)
    plt.close()


def visualize_latent(fig_folder, model_folder, gen_kwargs, sex='male'):
    z_dim = gen_kwargs['z_dim']
    y_dim = gen_kwargs['y_dim']
    G = Generator(**gen_kwargs)
    G.load_state_dict(torch.load(model_folder+'/gen_cropped.pkl', map_location=torch.device('cpu')))

    # Sample Z's and generate images:
    torch.manual_seed(66) if sex == 'male' else torch.manual_seed(42)
    togen_latent = torch.randn(25, z_dim).view(-1, z_dim, 1, 1)
    togen_labels = torch.ones(26).type(torch.LongTensor) if sex == 'male' else torch.zeros(26).type(torch.LongTensor)
    togen_labels[0] = togen_labels[0] -1 if sex == 'male' else togen_labels[0] + 1
    togen_labels = F.one_hot(togen_labels.view(-1)).view(-1, y_dim, 1, 1).type(torch.FloatTensor)
    togen_labels = togen_labels[1:]
    for i, (z_cont, z_const) in enumerate(zip(togen_latent, togen_labels)):
        gen_image(fig_folder, G, z_cont, z_const, i)

    # Fit transform the TSNE model
    tsne_model = TSNE()
    images = [fig_folder + f'/pic_{i}.png' for i in range(len(togen_latent))]
    tsne_coords = tsne_model.fit_transform(togen_latent.view(25, z_dim))

    # Plot
    ax = plt.gca()
    ax.set_xlim(-170, 300)
    ax.set_ylim(-300, 300)
    plt.figure(figsize=(50, 50))
    for i in range(len(images)):
        imageFile = images[i]
        img = mpimg.imread(imageFile)
        imgplot = ax.imshow(img)
        tx, ty = tsne_coords[i]
        transform = mpl.transforms.Affine2D().translate(tx, ty)
        imgplot.set_transform(transform + ax.transData)

    plt.show()

create_n_pictures(750, 0, "new_i", model_folder=None):
create_n_pictures(750, 1, "new_ii", model_folder=None):
create_n_pictures(750, 2, "new_iii", model_folder=None):
create_n_pictures(750, 3, "new_iv", model_folder=None):
create_n_pictures(750, 4, "new_v", model_folder=None):
create_n_pictures(750, 5, "new_vi", model_folder=None):
create_n_pictures(750, 6, "new_vii", model_folder=None):
create_n_pictures(750, 7, "new_viii", model_folder=None):
create_n_pictures(750, 8, "new_ix", model_folder=None):
create_n_pictures(750, 9, "new_x", model_folder=None):
