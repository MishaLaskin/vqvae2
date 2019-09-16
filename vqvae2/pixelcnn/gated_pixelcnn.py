

import argparse
import vqvae2.utils
from vqvae2.pixelcnn.models import GatedPixelCNN
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
from torchvision.utils import save_image
import time
import os
import sys
from vqvae2.utils import ImageDataset
"""
add vqvae and pixelcnn dirs to path
make sure you run from vqvae directory
"""
current_dir = sys.path.append(os.getcwd())
pixelcnn_dir = sys.path.append(os.getcwd() + '/pixelcnn')


"""
Hyperparameters
"""
parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--log_interval", type=int, default=100)
parser.add_argument("-save", action="store_true")
parser.add_argument("-gen_samples", action="store_true")

parser.add_argument("--dataset",  type=str, default='LATENT_BLOCK',
                    help='accepts CIFAR10 | MNIST | FashionMNIST | LATENT_BLOCK')

parser.add_argument("--data_file_path", type=str,
                    default='/home/misha/research/data/paths/latent_sawyer_push_medium_48.npy')
parser.add_argument("--save_path", type=str,
                    default='/home/misha/research/data/saved_models/xxxxx_pixelcnn_sawyer_push_medium_48.npy')

parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--img_dim", type=int, default=3)
parser.add_argument("--input_dim", type=int, default=1,
                    help='1 for grayscale 3 for rgb')
parser.add_argument("--n_embeddings", type=int, default=64,
                    help='number of embeddings from VQ VAE')
parser.add_argument("--n_layers", type=int, default=15)
parser.add_argument("--learning_rate", type=float, default=3e-4)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
data loaders
"""
transform_chain = transforms.Compose([
    transforms.ToTensor(),

])
data = ImageDataset(args.data_file_path,
                    transform=None)

dataloader = torch.utils.data.DataLoader(
    data, batch_size=args.batch_size, shuffle=True)


model = GatedPixelCNN(args.n_embeddings, args.img_dim **
                      2, args.n_layers).to(device)
criterion = nn.CrossEntropyLoss().cuda()
opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


"""
train, test, and log
"""


def train():
    train_loss = []
    for batch_idx, (x, label) in enumerate(dataloader):

        print(x.shape)

        start_time = time.time()
        if args.dataset == 'LATENT_BLOCK':
            x = (x[:, 0]).long().to(device)
        else:
            x = (x[:, 0] * (K-1)).long().to(device)
        label = label.to(device)

        # Train PixelCNN with images
        logits = model(x, label)

        print(x.shape, label.shape)
        print(logits.shape)
        assert False
        logits = logits.permute(0, 2, 3, 1).contiguous()

        loss = criterion(
            logits.view(-1, args.n_embeddings),
            x.view(-1)
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss.append(loss.item())

        if (batch_idx + 1) % args.log_interval == 0:
            print('\tIter: [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
                batch_idx * len(x), len(dataloader.dataset),
                args.log_interval * batch_idx / len(dataloader),
                np.asarray(train_loss)[-args.log_interval:].mean(0),
                time.time() - start_time
            ))


def test():
    start_time = time.time()
    val_loss = []
    with torch.no_grad():
        for batch_idx, (x, label) in enumerate(dataloader):
            if args.dataset == 'LATENT_BLOCK':
                x = (x[:, 0]).cuda()
            else:
                x = (x[:, 0] * (args.n_embeddings-1)).long().cuda()
            label = label.cuda()

            logits = model(x, label)

            logits = logits.permute(0, 2, 3, 1).contiguous()
            loss = criterion(
                logits.view(-1, args.n_embeddings),
                x.view(-1)
            )

            val_loss.append(loss.item())

    print('Validation Completed!\tLoss: {} Time: {}'.format(
        np.asarray(val_loss).mean(0),
        time.time() - start_time
    ))
    return np.asarray(val_loss).mean(0)


def generate_samples(epoch):
    label = torch.arange(10).expand(10, 10).contiguous().view(-1)
    label = label.long().cuda()

    x_tilde = model.generate(label, shape=(
        args.img_dim, args.img_dim), batch_size=100)

    print(x_tilde[0])


BEST_LOSS = 999
LAST_SAVED = -1
for epoch in range(1, args.epochs):
    print("\nEpoch {}:".format(epoch))
    train()
    cur_loss = test()

    if args.save or cur_loss <= BEST_LOSS:
        BEST_LOSS = cur_loss
        LAST_SAVED = epoch

        print("Saving model!")
        torch.save(model.state_dict(), args.save_path)
    else:
        print("Not saving model! Last saved: {}".format(LAST_SAVED))
    if args.gen_samples:
        generate_samples(epoch)
