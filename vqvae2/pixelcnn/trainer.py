

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
#current_dir = sys.path.append(os.getcwd())
#pixelcnn_dir = sys.path.append(os.getcwd() + '/pixelcnn')


"""
Hyperparameters
"""

"""
parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--log_interval", type=int, default=100)
parser.add_argument("-save", action="store_true")
parser.add_argument("-gen_samples", action="store_true")

parser.add_argument("--dataset",  type=str, default='LATENT_BLOCK',
                    help='accepts CIFAR10 | MNIST | FashionMNIST | LATENT_BLOCK')

parser.add_argument("--data_file_path", type=str,
                    default='/home/misha/research/data/paths/sawyer_push_rot3_48.npy')
parser.add_argument("--save_path", type=str,
                    default='/home/misha/research/data/saved_models/x_pixelcnn_sawyer_push_0918_48.npy')

parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--img_dim", type=int, default=3)
parser.add_argument("--input_dim", type=int, default=1,
                    help='1 for grayscale 3 for rgb')
parser.add_argument("--n_embeddings", type=int, default=64,
                    help='number of embeddings from VQ VAE')
parser.add_argument("--n_layers", type=int, default=15)
parser.add_argument("--learning_rate", type=float, default=1e-3)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


train, test, and log
"""


class PixelCNNTrainer:
    def __init__(self,
                 data_file_path='/home/misha/research/data/paths/latent_sawyer_push_medium_48.npy',
                 batch_size=64,
                 n_embeddings=64,
                 img_dim=48,
                 n_layers=15,
                 learning_rate=1e-3,
                 log_interval=100,
                 epochs=10,
                 save_path='/home/misha/research/data/saved_models/x_pixelcnn_sawyer_push_0918_48.npy',
                 **kwargs):
        super().__init__()
        self.data_file_path = data_file_path
        self.batch_size = batch_size
        self.n_embeddings = n_embeddings
        self.img_dim = img_dim
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.save_path = save_path

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        """
        data loaders
        """
        transform_chain = transforms.Compose([
            transforms.ToTensor(),

        ])
        self.data = ImageDataset(self.data_file_path,
                                 transform=transform_chain)

        self.dataloader = torch.utils.data.DataLoader(
            self.data, batch_size=self.batch_size, shuffle=True)

        self.model = GatedPixelCNN(self.n_embeddings, self.img_dim **
                                   2, self.n_layers).to(self.device)
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate)

    def train(self):
        train_loss = []
        for batch_idx, (x, label) in enumerate(self.dataloader):

            # print(x.shape)

            start_time = time.time()
            x = (x[:, 0]).long().to(self.device)
            label = label.long().cuda()
            # Train PixelCNN with images
            logits = self.model(x, label)

            #print(x.shape, label.shape)
            # print(logits.shape)
            #assert False
            logits = logits.permute(0, 2, 3, 1).contiguous()

            loss = self.criterion(
                logits.view(-1, self.n_embeddings),
                x.view(-1)
            )

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            train_loss.append(loss.item())

            if (batch_idx + 1) % self.log_interval == 0:
                print('\tIter: [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
                    batch_idx * len(x), len(dataloader.dataset),
                    self.log_interval * batch_idx / len(dataloader),
                    np.asarray(train_loss)[-self.log_interval:].mean(0),
                    time.time() - start_time
                ))

    def test(self):
        start_time = time.time()
        val_loss = []
        with torch.no_grad():
            for batch_idx, (x, label) in enumerate(dataloader):
                #print('data in', x.shape, label.shape)

                x = (x[:, 0]).cuda()

                label = label.long().cuda()

                logits = self.model(x, label)

                #print('og logits', logits.shape)

                logits = logits.permute(0, 2, 3, 1).contiguous()
                #print('data and logits', x.shape, label.shape, logits.shape)
                #assert False
                loss = self.criterion(
                    logits.view(-1, args.n_embeddings),
                    x.view(-1)
                )

                val_loss.append(loss.item())

        print('Validation Completed!\tLoss: {} Time: {}'.format(
            np.asarray(val_loss).mean(0),
            time.time() - start_time
        ))
        return np.asarray(val_loss).mean(0)

    def run(self):

        BEST_LOSS = 999
        LAST_SAVED = -1
        for epoch in range(1, self.epochs):
            print("\nEpoch {}:".format(epoch))
            self.train()
            cur_loss = self.test()

            if True or (args.save or cur_loss <= BEST_LOSS):
                BEST_LOSS = cur_loss
                LAST_SAVED = epoch

                print("Saving model!")
                torch.save(self.model.state_dict(), args.save_path)
            else:
                print("Not saving model! Last saved: {}".format(LAST_SAVED))


if __name__ == '__main__':
    trainer = PixelCNNTrainer()
    trainer.run()
