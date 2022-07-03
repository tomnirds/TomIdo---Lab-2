from torch import nn
import torch

ACTIVATIONS = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid(), 'logsoftmax': nn.LogSoftmax(dim=1),
               "leakyrelu": nn.LeakyReLU(0.2)}
CONVS = {"conv2d": nn.Conv2d, "convtranspose2d": nn.ConvTranspose2d}

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class CNN(nn.Module):
    """
    A CNN class, with control over activations, conv types and regularization.
    """

    def __init__(self, in_channel, channels,
                 kernel_size, stride, padding,
                 activation_type, final_activation_type, conv_type,
                 final_overrides=None,
                 resize_dim=(64, 64)):

        super(CNN, self).__init__()
        self.in_channel = in_channel
        self.im_dims = resize_dim
        activation = ACTIVATIONS[activation_type]
        final_activation = ACTIVATIONS[final_activation_type]
        conv = CONVS[conv_type]
        conv_layers = []
        N = len(channels)
        conv_layers.extend(
            [conv(in_channel, channels[0], kernel_size=kernel_size, stride=stride, padding=padding),
             nn.BatchNorm2d(channels[0]), activation])
        K = N - 2
        for i in range(K):
            conv_layers.extend(
                [conv(channels[i], channels[i + 1], kernel_size=kernel_size, stride=stride, padding=padding),
                 nn.BatchNorm2d(channels[i + 1]),
                 activation])
        if final_overrides:
            kernel_size, stride, padding = final_overrides
        conv_layers.extend(
            [conv(channels[-2], channels[-1], kernel_size=kernel_size, stride=stride, padding=padding),
             final_activation])
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        out = self.conv(x)
        return out


class Generator(nn.Module):
    def __init__(self, z_dim, y_dim, img_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.dim = img_dim
        self.y_dim = y_dim
        self.labelconv = nn.Sequential(
            nn.ConvTranspose2d(y_dim, img_dim * 4, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(img_dim * 4),
            nn.ReLU())

        self.latentconv = nn.Sequential(
            nn.ConvTranspose2d(z_dim, img_dim * 4, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(img_dim * 4),
            nn.ReLU())

        cnn_kwargs = {'in_channel': img_dim * 8,
                      'channels': [img_dim * 4, img_dim * 2, 3],
                      'kernel_size': 4,
                      'stride': 2,
                      'padding': 1,
                      'activation_type': 'relu',
                      'final_activation_type': 'tanh',
                      'conv_type': 'convtranspose2d'}

        self.model = CNN(**cnn_kwargs)

    def sample(self, n, properties):
        device = next(self.parameters()).device
        pre_construct = torch.randn(size=(n, self.z_dim)).view(-1, self.z_dim, 1, 1).to(device)
        samples = self(pre_construct, properties).to(device)  # Creating fake images

        return samples

    def forward(self, z, c):
        latent_emb = self.latentconv(z)
        label_emb = self.labelconv(c)
        x = torch.cat([latent_emb, label_emb], 1)
        image = self.model(x)
        return image


class Discriminator(nn.Module):
    def __init__(self, y_dim, img_dim):
        super(Discriminator, self).__init__()

        self.dim = img_dim
        self.imgconv = nn.Sequential(
            nn.Conv2d(3, img_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2))

        self.labelconv = nn.Sequential(
            nn.Conv2d(y_dim, img_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2))

        cnn_kwargs = {'in_channel': img_dim * 2,
                      'channels': [img_dim * 4, img_dim * 8, 1],
                      'kernel_size': 4,
                      'stride': 2,
                      'padding': 1,
                      'activation_type': 'leakyrelu',
                      'final_activation_type': 'sigmoid',
                      'conv_type': 'conv2d',
                      'final_overrides': [4, 1, 0]}

        self.model = CNN(**cnn_kwargs)

    def forward(self, z, c):
        imgout = self.imgconv(z)
        labelout = self.labelconv(c)
        x = torch.cat([imgout, labelout], 1)
        out = self.model(x)
        return out
