from torchvision.transforms import Resize, Compose, ToTensor, Normalize, CenterCrop
import matplotlib.pyplot as plt
import torch
import torchvision.datasets as dsets
import torchvision.utils
from torch.utils.data import Dataset, DataLoader
import cdcgan
import vizzes
from train import train_cgan
import torch.nn.functional as F
import argparse
import pandas as pd
import glob
import tqdm
from PIL import Image

label_to_roman = {0: 'i', 1: 'ii', 2: 'iii', 3: 'iv', 4: 'v', 5: 'vi', 6: 'vii', 7: 'viii', 8: 'ix', 9: 'x'}
def get_rmnist(motherland_folder):
    """
    Convert the data to a dataloader-friendly format
    """
    roman_to_label = {'i': 0, 'ii': 1, 'iii': 2, 'iv': 3, 'v': 4, 'vi': 5, 'vii': 6, 'viii': 7, 'ix': 8, 'x': 9}
    all_train_folders = glob.glob(motherland_folder + '/train' + '/*')
    pics = []
    labels = []
    for label_folder in tqdm.tqdm(all_train_folders, desc='Creating Dataframe'):
        label = roman_to_label[label_folder.split('\\')[-1]]
        all_label_pics = glob.glob(label_folder + '/*')
        all_label_pics = [picpath.split('\\')[-1] for picpath in all_label_pics]
        pics.extend(all_label_pics)
        labels.extend([label] * len(all_label_pics))
    rmnist_df = pd.DataFrame({'image': pics, 'label': labels})
    rmnist_df.to_csv('rmnist.csv', index=False)


class RMnistDataset(Dataset):
    """
    Creates a dataset to be used alongside the dataloader.
    """

    def __init__(self, ds_path, transform):
        self.dataset = pd.read_csv(ds_path)
        self.transform = transform
        self.img_paths, self.labels = self.dataset['image'], self.dataset['label']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.img_paths.iloc[idx]
        lbl = self.labels[idx]
        path = 'data/train/'+label_to_roman[lbl]+'/'+img_path
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, lbl


def create_dataloader(ds_path, transform, bs, shuffle):
    cds = RMnistDataset(ds_path=ds_path, transform=transform)
    return DataLoader(cds, batch_size=bs, shuffle=shuffle, num_workers=1, pin_memory=True, drop_last=True)


if __name__ == '__main__':
    #get_rmnist('data')
    img_dim = 32
    batch_size = 128
    label_dim = 10
    z_dim = 100
    trans = Compose([Resize((img_dim, img_dim)),
                     ToTensor(),
                     Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    data_loader = create_dataloader('rmnist.csv', trans, 128, True)
    #vizzes.show_some_imgs(data_loader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    D = cdcgan.Discriminator(img_dim=img_dim, y_dim=label_dim).to(device)
    G = cdcgan.Generator(img_dim=img_dim, z_dim=z_dim, y_dim=label_dim).to(device)
    G.apply(cdcgan.weights_init)
    D.apply(cdcgan.weights_init)
    print(D)
    print(G)
    train_cgan(G, D, data_loader, 300, device)

    # Visualizations:
    # gen_kw = {'z_dim': z_dim, 'y_dim': label_dim, 'img_dim': img_dim}
    # vizzes.visualize_latent('model_fin/latent_pics', 'model_fin', gen_kw, sex='male')
    # vizzes.create_gif('model_fin/training_pics')
