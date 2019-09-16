import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from vqvae2.models.vqvae import VQVAE
from vqvae2.pixelcnn.models import GatedPixelCNN

from vqvae2.utils import ImageDataset, readable_timestamp, save_model_and_results
from vqvae2.config import imsize48_default_architecture
parser = argparse.ArgumentParser()

"""
Hyperparameters
"""
timestamp = readable_timestamp()


parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--n_updates", type=int, default=int(1e5))
# parser.add_argument("--n_hiddens", type=int, default=128)
# parser.add_argument("--n_residual_hiddens", type=int, default=32)
# parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=2)
parser.add_argument("--n_embeddings", type=int, default=64)
parser.add_argument("--beta", type=float, default=.25)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--log_interval", type=int, default=100)
parser.add_argument("--dataset",  type=str, default='PUSHER')
parser.add_argument("--imsize", type=int, default=48)
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--pix_layers", type=int, default=15)

# whether or not to save model
parser.add_argument("-save", action="store_true")
# parser.add_argument("-temporal", action="store_true")
parser.add_argument("--saved_name",  type=str,
                    default='/home/misha/research/data/saved_models/vqvae_pix_sawyer_push_medium_48.pth')
parser.add_argument("--data_file_path", type=str,
                    default='/home/misha/research/data/paths/sawyer_push_medium_48.npy')
args = parser.parse_args()


device = torch.device("cuda:"+str(args.gpu_id)
                      if torch.cuda.is_available() else "cpu")


if args.save:

    print('Results will be saved in ./results/'+args.saved_name)

"""
Load data and define batch data loaders
"""

transform_chain = transforms.Compose([
    # transforms.Grayscale(),
    # transforms.ColorJitteGatedPixelCNNr(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
    transforms.ToTensor(),
    # transforms.Normalize([0., 0., 0, ], [1., 1., 1.]),

])
data = ImageDataset(args.data_file_path,
                    transform=transform_chain)
dataloader = torch.utils.data.DataLoader(
    data, batch_size=args.batch_size, shuffle=True)

"""
Set up VQ-VAE model with components defined in ./models/ folder
"""
architecture = imsize48_default_architecture

model = VQVAE(architecture, args.imsize, args.n_embeddings,
              args.embedding_dim, args.beta, args.gpu_id)
device = model.device
model = model.to(device)

# pix_model = GatedPixelCNN(args.n_embeddings, args.imsize **
#                          2, args.pix_layers).to(device)
pix_model = model.pixelcnn
"""
Set up optimizer and training loop
"""
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
pix_optimizer = optim.Adam(
    model.parameters(), lr=args.learning_rate, amsgrad=True)
pix_criterion = nn.CrossEntropyLoss().to(device)

model.train()


results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
    'perplexities': [],
    'pix_loss_vals': []
}


def reconstruct(x, model):

    vq_encoder_output = model.pre_quantization_conv(model.encoder(x))
    _, z_q, _, _, e_indices = model.vector_quantization(vq_encoder_output)
    z_q = model.pre_dequantization_conv(z_q)
    x_recon = model.decoder(z_q)

    return x_recon


def generate_samples(model, z_dim, n):
    pix_model = model.pixelcnn
    labels = torch.ones(n).long().to(device)
    samples = pix_model.generate(labels, (z_dim, z_dim), n)
    # print(samples[0])

    samples = torch.tensor(samples).reshape(-1, 1).long().to(device)
    # print(samples.shape)
    e_weights = model.vector_quantization.embedding.weight
    weight_dim = np.sqrt(e_weights.shape[0]).astype(np.int32)
    min_encodings = torch.zeros(
        samples.shape[0], weight_dim**2).long().to(device)
    # print(min_encodings.shape)
    min_encodings.scatter_(1, samples, 1)
    #print(min_encodings[:9, :])
    # assert False

    # print(min_encodings.shape, e_weights.shape,
    #      np.sqrt(e_weights.shape[0]).astype(np.int32))
    # assert False

    z_q = torch.matmul(min_encodings.float(),
                       e_weights.float()).view((n, z_dim, z_dim, args.embedding_dim))
    # print(z_q.sum(3)[0])
    #assert False
    z_q = z_q.permute(0, 3, 1, 2).contiguous()
    z_q = model.pre_dequantization_conv(z_q.float())
    x_recon = model.decoder(z_q)

    return x_recon


"""
def reconstruct_from_pixelcnn(model, samples):

    min_encoding_indices = torch.tensor(
        samples).reshape(-1, 1).long().to(device)
    x_recon, z_q, e_indices = generate_samples(min_encoding_indices, model)
    x_val_recon = x_val_recon.permute(0, 1, 3, 2)

    return x_recon
"""


def train_pixelcnn(x, label, model, criterion, opt):

    x = (x[:, 0]).long().to(device)

    # Train PixelCNN with images
    logits = model(x, label)

    logits = logits.permute(0, 2, 3, 1).contiguous()

    loss = criterion(
        logits.view(-1, args.n_embeddings),
        x.view(-1)
    )

    opt.zero_grad()
    loss.backward()
    opt.step()

    return loss.item()


def one_vqvae_optimization_step(x, model, opt):

    x = x.float().to(device)
    opt.zero_grad()

    # train vq vae
    embedding_loss, z, x_hat, perplexity, binaries = model(
        x, include_binaries=True)

    recon_loss = 100.0 * torch.mean((x_hat - x)**2)  # / x_train_var
    loss = recon_loss + embedding_loss

    loss.backward()
    opt.step()

    # train pixelcnn
    z_dim = model.z_dim
    binaries = binaries.view(
        args.batch_size, 1, z_dim, z_dim).to(device)
    labels = torch.tensor(np.ones(args.batch_size)).long().to(device)
    # print(binaries.shape, args.batch_size)
    pix_loss = train_pixelcnn(binaries, labels, pix_model,
                              pix_criterion, pix_optimizer)
    return loss.item(), pix_loss, perplexity.item()


def train_vqvae():

    for i in tqdm(range(args.n_updates)):
        (x, _) = next(iter(dataloader))
        x = x.permute(0, 2, 1, 3).float().to(device)
        z_dim = model.z_dim
        loss, pix_loss, perplexity = one_vqvae_optimization_step(
            x, model, optimizer)

        #print(loss, pix_loss, perplexity)

        # gather results for logging
        results["perplexities"].append(perplexity)
        results["loss_vals"].append(loss)
        results["pix_loss_vals"].append(pix_loss)
        results["n_updates"] = i

        if i % args.log_interval == 0:
            """
            save model and print values
            """
            if args.save:
                hyperparameters = args.__dict__
                save_dict = {
                    'model_state_dict': model.state_dict(),
                    'pixelcnn_state_dict': pix_model.state_dict(),
                    'hyperparameters': hyperparameters,
                    'architecture': imsize48_default_architecture,
                    'results': results
                }

                torch.save(save_dict, args.saved_name)

            if i % 10*args.log_interval == 0 and i > 0:
                # reconstruct batch and sample from prior
                x_recon = reconstruct(x, model)
                save_image(
                    x_recon[:8], '/home/misha/research/data/images/vqvae/r'+str(i)+'.png', 4)
                x_sample = generate_samples(model, z_dim, 8)
                save_image(
                    x_sample[:8], '/home/misha/research/data/images/vqvae/s'+str(i)+'.png', 4)

            print('Update #', i,
                  'Loss', np.mean(results["loss_vals"][-args.log_interval:]),
                  'Pix Loss', np.mean(
                      results["pix_loss_vals"][-args.log_interval:]),
                  'Perplexity:', np.mean(results["perplexities"][-args.log_interval:]))
            # assert False


if __name__ == "__main__":
    train_vqvae()
