from collections import OrderedDict
from os import path as osp
import numpy as np
import torch
from torch import optim
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from multiworld.core.image_env import normalize_image
from rlkit.core import logger
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.data import (
    ImageDataset,
    InfiniteWeightedRandomSampler,
    InfiniteRandomSampler,
)
from rlkit.util.ml_util import ConstantSchedule
import torch.nn as nn


def relative_probs_from_log_probs(log_probs):
    """
    Returns relative probability from the log probabilities. They're not exactly
    equal to the probability, but relative scalings between them are all maintained.

    For correctness, all log_probs must be passed in at the same time.
    """
    probs = np.exp(log_probs - log_probs.mean())
    assert not np.any(probs <= 0), 'choose a smaller power'
    return probs


def compute_log_p_log_q_log_d(
    model,
    data,
    decoder_distribution='bernoulli',
    num_latents_to_sample=1,
    sampling_method='importance_sampling'
):
    assert data.dtype == np.float64, 'images should be normalized'
    imgs = ptu.from_numpy(data)
    latent_distribution_params = model.encode(imgs)
    batch_size = data.shape[0]
    representation_size = model.representation_size
    log_p, log_q, log_d = ptu.zeros((batch_size, num_latents_to_sample)), ptu.zeros(
        (batch_size, num_latents_to_sample)), ptu.zeros((batch_size, num_latents_to_sample))
    true_prior = Normal(ptu.zeros((batch_size, representation_size)),
                        ptu.ones((batch_size, representation_size)))
    mus, logvars = latent_distribution_params
    for i in range(num_latents_to_sample):
        if sampling_method == 'importance_sampling':
            latents = model.rsample(latent_distribution_params)
        elif sampling_method == 'biased_sampling':
            latents = model.rsample(latent_distribution_params)
        elif sampling_method == 'true_prior_sampling':
            latents = true_prior.rsample()
        else:
            raise EnvironmentError('Invalid Sampling Method Provided')

        stds = logvars.exp().pow(.5)
        vae_dist = Normal(mus, stds)
        log_p_z = true_prior.log_prob(latents).sum(dim=1)
        log_q_z_given_x = vae_dist.log_prob(latents).sum(dim=1)
        if decoder_distribution == 'bernoulli':
            decoded = model.decode(latents)[0]
            log_d_x_given_z = torch.log(
                imgs * decoded + (1 - imgs) * (1 - decoded) + 1e-8).sum(dim=1)
        elif decoder_distribution == 'gaussian_identity_variance':
            _, obs_distribution_params = model.decode(latents)
            dec_mu, dec_logvar = obs_distribution_params
            dec_var = dec_logvar.exp()
            decoder_dist = Normal(dec_mu, dec_var.pow(.5))
            log_d_x_given_z = decoder_dist.log_prob(imgs).sum(dim=1)
        else:
            raise EnvironmentError('Invalid Decoder Distribution Provided')

        log_p[:, i] = log_p_z
        log_q[:, i] = log_q_z_given_x
        log_d[:, i] = log_d_x_given_z
    return log_p, log_q, log_d


def compute_p_x_np_to_np(
    model,
    data,
    power,
    decoder_distribution='bernoulli',
    num_latents_to_sample=1,
    sampling_method='importance_sampling'
):
    assert data.dtype == np.float64, 'images should be normalized'
    assert power >= - \
        1 and power <= 0, 'power for skew-fit should belong to [-1, 0]'

    log_p, log_q, log_d = compute_log_p_log_q_log_d(
        model,
        data,
        decoder_distribution,
        num_latents_to_sample,
        sampling_method
    )

    if sampling_method == 'importance_sampling':
        log_p_x = (log_p - log_q + log_d).mean(dim=1)
    elif sampling_method == 'biased_sampling' or sampling_method == 'true_prior_sampling':
        log_p_x = log_d.mean(dim=1)
    else:
        raise EnvironmentError('Invalid Sampling Method Provided')
    log_p_x_skewed = power * log_p_x

    return ptu.get_numpy(log_p_x_skewed)


class ConvVQVAETrainer(object):
    def __init__(
            self,
            train_dataset,
            test_dataset,
            model,
            batch_size=128,
            log_interval=0,
            # beta=0.5,
            # beta_schedule=None,
            lr=None,
            do_scatterplot=False,
            normalize=False,
            mse_weight=0.1,
            is_auto_encoder=False,
            background_subtract=False,
            use_parallel_dataloading=True,
            train_data_workers=2,
            # skew_dataset=False,
            # skew_config=None,
            priority_function_kwargs=None,
            start_skew_epoch=0,
            weight_decay=0,
            **kwargs
    ):

        self.log_interval = log_interval
        self.batch_size = batch_size

        assert lr is not None

        self.imsize = model.imsize
        self.do_scatterplot = do_scatterplot
        self.representation_size = model.representation_size
        model.to(ptu.device)

        self.model = model
        self.pixelcnn = model.pixelcnn

        self.input_channels = model.input_channels
        self.imlength = model.imlength

        self.lr = lr
        params = list(self.model.parameters())
        self.optimizer = optim.Adam(params,
                                    lr=self.lr,
                                    weight_decay=weight_decay,
                                    )
        self.pixelcnn_opt = optim.Adam(
            self.pixelcnn.parameters(), lr=self.lr, weight_decay=weight_decay, amsgrad=True)
        self.pixelcnn_criterion = nn.CrossEntropyLoss().to(ptu.device)

        self.train_dataset, self.test_dataset = train_dataset, test_dataset
        assert self.train_dataset.dtype == np.uint8
        assert self.test_dataset.dtype == np.uint8
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.batch_size = batch_size
        self.use_parallel_dataloading = use_parallel_dataloading
        self.train_data_workers = train_data_workers
        if priority_function_kwargs is None:
            self.priority_function_kwargs = dict()
        else:
            self.priority_function_kwargs = priority_function_kwargs

        self._train_weights = None

        if use_parallel_dataloading:
            self.train_dataset_pt = ImageDataset(
                train_dataset,
                should_normalize=True
            )
            self.test_dataset_pt = ImageDataset(
                test_dataset,
                should_normalize=True
            )

            #base_sampler = InfiniteRandomSampler(self.train_dataset)
            self.train_dataloader = DataLoader(
                self.train_dataset_pt,
                sampler=InfiniteRandomSampler(self.train_dataset),
                batch_size=batch_size,
                drop_last=False,
                num_workers=train_data_workers,
                pin_memory=True,
            )
            self.test_dataloader = DataLoader(
                self.test_dataset_pt,
                sampler=InfiniteRandomSampler(self.test_dataset),
                batch_size=batch_size,
                drop_last=False,
                num_workers=0,
                pin_memory=True,
            )
            self.train_dataloader = iter(self.train_dataloader)
            self.test_dataloader = iter(self.test_dataloader)

        self.normalize = normalize
        self.mse_weight = mse_weight
        self.background_subtract = background_subtract

        if self.normalize or self.background_subtract:
            self.train_data_mean = np.mean(self.train_dataset, axis=0)
            self.train_data_mean = normalize_image(
                np.uint8(self.train_data_mean)
            )
        self.eval_statistics = OrderedDict()
        self._extra_stats_to_log = None

    def get_dataset_stats(self, data):
        # edit for VQVAE
        torch_input = ptu.from_numpy(normalize_image(data))
        mus, log_vars = self.model.encode(torch_input)
        mus = ptu.get_numpy(mus)
        mean = np.mean(mus, axis=0)
        std = np.std(mus, axis=0)
        return mus, mean, std

    def update_train_weights(self):
        print('update train weights (pass)')
        pass

    def _compute_train_weights(self):
        print('compute train weights (pass)')
        pass

    def set_vae(self, vae):
        self.model = vae
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def one_pixelcnn_optimization_step(self, z_batch):
        criterion = self.pixelcnn_criterion

        labels = torch.ones(z_batch.shape[0]).long().to(ptu.device)
        z_batch = (z_batch[:, 0]).long().to(ptu.device)

        # Train PixelCNN with images
        logits = self.pixelcnn(z_batch, labels)

        logits = logits.permute(0, 2, 3, 1).contiguous()
        # CHANGE THIS LATER should be in architeture
        n_embeddings = 64

        loss = criterion(
            logits.view(-1, n_embeddings),
            z_batch.view(-1)
        )

        self.pixelcnn_opt.zero_grad()
        loss.backward()
        self.pixelcnn_opt.step()

        return loss

    def one_vqvae_optimization_step(self, x):

        x = x.float().to(ptu.device)
        self.optimizer.zero_grad()

        # train vq vae
        embedding_loss, z, x_hat, perplexity, binaries = self.model(
            x, include_binaries=True)
        #print(x.shape, x_hat.shape, z.shape)
        if len(x_hat.shape) != 2:
            x_hat = x_hat.view(x.shape[0], -1)
        recon_loss = 100.0 * torch.mean((x_hat - x)**2)  # / x_train_var
        loss = recon_loss + embedding_loss

        loss.backward()
        self.optimizer.step()

        # train pixelcnn
        z_dim = self.model.z_dim
        binaries = binaries.view(
            self.batch_size, 1, z_dim, z_dim).to(ptu.device)

        pix_loss = self.one_pixelcnn_optimization_step(binaries)
        return recon_loss.item(), loss.item(), pix_loss.item(), perplexity.item()

    def get_batch(self, train=True, epoch=None):
        if self.use_parallel_dataloading:
            if not train:
                dataloader = self.test_dataloader
            else:
                dataloader = self.train_dataloader
            samples = next(dataloader).to(ptu.device)
            return samples

        dataset = self.train_dataset if train else self.test_dataset

        ind = np.random.randint(0, len(dataset), self.batch_size)
        samples = normalize_image(dataset[ind, :])
        if self.normalize:
            samples = ((samples - self.train_data_mean) + 1) / 2
        if self.background_subtract:
            samples = samples - self.train_data_mean
        return ptu.from_numpy(samples)

    def get_debug_batch(self, train=True):
        dataset = self.train_dataset if train else self.test_dataset
        X, Y = dataset
        ind = np.random.randint(0, Y.shape[0], self.batch_size)
        X = X[ind, :]
        Y = Y[ind, :]
        return ptu.from_numpy(X), ptu.from_numpy(Y)

    def train_epoch(self, epoch, sample_batch=None, batches=100, from_rl=False):
        self.model.train()
        self.pixelcnn.train()
        losses = []
        pixelcnn_losses = []
        perplexities = []
        log_probs = []
        zs = []
        for batch_idx in range(batches):
            if sample_batch is not None:
                data = sample_batch(self.batch_size, epoch)
                # obs = data['obs']
                next_obs = data['next_obs']
                # actions = data['actions']
            else:
                next_obs = self.get_batch(epoch=epoch)
                obs = None
                actions = None
            self.optimizer.zero_grad()
            """
            start: change this for VQ VAE
            """
            log_prob, loss, pixelcnn_loss, perplexity = self.one_vqvae_optimization_step(
                next_obs)

            losses.append(loss)
            pixelcnn_losses.append(pixelcnn_loss)
            perplexities.append(perplexity)
            log_probs.append(log_prob)

            _, latents, _, _ = self.model(next_obs)

            z_data = ptu.get_numpy(latents.cpu())
            for i in range(len(z_data)):
                zs.append(z_data[i, :])

        if not from_rl:
            #assert False, 'only support training from RL'
            zs = np.array(zs)
            #self.model.dist_mu = zs.mean(axis=0)
            #self.model.dist_std = zs.std(axis=0)

        self.eval_statistics['train/log prob'] = np.mean(log_probs)
        self.eval_statistics['train/pixelcnn_loss'] = np.mean(pixelcnn_losses)
        self.eval_statistics['train/loss'] = np.mean(losses)
        self.eval_statistics['train/perplexities'] = np.mean(perplexities)
        """
        end: change this for VQ VAE
        """

    def get_diagnostics(self):
        return self.eval_statistics

    def test_epoch(
            self,
            epoch,
            save_reconstruction=True,
            save_vae=True,
            from_rl=False,
    ):
        self.model.eval()
        losses = []
        log_probs = []
        pixelcnn_losses = []
        perplexities = []
        zs = []
        for batch_idx in range(10):
            next_obs = self.get_batch(train=False)
            """
            start: change this for VQ VAE
            """

            log_prob, loss, pixelcnn_loss, perplexity = self.one_vqvae_optimization_step(
                next_obs)

            losses.append(loss)
            pixelcnn_losses.append(pixelcnn_loss)
            perplexities.append(perplexity)
            log_probs.append(log_prob)

            _, latents, reconstructions, _ = self.model(next_obs)

            z_data = ptu.get_numpy(latents.cpu())
            for i in range(len(z_data)):
                zs.append(z_data[i, :])

            if batch_idx == 0 and save_reconstruction:
                n = min(next_obs.size(0), 8)
                comparison = torch.cat([
                    next_obs[:n].narrow(start=0, length=self.imlength, dim=1)
                    .contiguous().view(
                        -1, self.input_channels, self.imsize, self.imsize
                    ).transpose(2, 3),
                    reconstructions.view(
                        self.batch_size,
                        self.input_channels,
                        self.imsize,
                        self.imsize,
                    )[:n].transpose(2, 3)
                ])
                save_dir = osp.join(logger.get_snapshot_dir(),
                                    'r%d.png' % epoch)
                save_image(comparison.data.cpu(), save_dir, nrow=n)

        zs = np.array(zs)

        self.eval_statistics['epoch'] = epoch
        self.eval_statistics['train/log prob'] = np.mean(log_probs)
        self.eval_statistics['train/pixelcnn_loss'] = np.mean(pixelcnn_losses)
        self.eval_statistics['train/loss'] = np.mean(losses)
        self.eval_statistics['train/perplexities'] = np.mean(perplexities)
        if not from_rl:
            for k, v in self.eval_statistics.items():
                logger.record_tabular(k, v)
            logger.dump_tabular()
            if save_vae:
                logger.save_itr_params(epoch, self.model)
        """
        end: change this for VQ VAE
        """

    def debug_statistics(self):
        """
        Given an image $$x$$, samples a bunch of latents from the prior
        $$z_i$$ and decode them $$\hat x_i$$.
        Compare this to $$\hat x$$, the reconstruction of $$x$$.
        Ideally
         - All the $$\hat x_i$$s do worse than $$\hat x$$ (makes sure VAE
           isn’t ignoring the latent)
         - Some $$\hat x_i$$ do better than other $$\hat x_i$$ (tests for
           coverage)
        """
        debug_batch_size = 64
        data = self.get_batch(train=False)
        reconstructions, _, _ = self.model(data)
        img = data[0]
        recon_mse = ((reconstructions[0] - img) ** 2).mean().view(-1)
        img_repeated = img.expand((debug_batch_size, img.shape[0]))

        samples = ptu.randn(debug_batch_size, self.representation_size)
        random_imgs, _ = self.model.decode(samples)
        random_mses = (random_imgs - img_repeated) ** 2
        mse_improvement = ptu.get_numpy(random_mses.mean(dim=1) - recon_mse)
        stats = create_stats_ordered_dict(
            'debug/MSE improvement over random',
            mse_improvement,
        )
        stats.update(create_stats_ordered_dict(
            'debug/MSE of random decoding',
            ptu.get_numpy(random_mses),
        ))
        stats['debug/MSE of reconstruction'] = ptu.get_numpy(
            recon_mse
        )[0]
        if self.skew_dataset:
            stats.update(create_stats_ordered_dict(
                'train weight',
                self._train_weights
            ))
        return stats

    def dump_samples(self, epoch):
        self.model.eval()
        sample = ptu.randn(64, self.representation_size)
        sample = self.model.decode(sample)[0].cpu()
        save_dir = osp.join(logger.get_snapshot_dir(), 's%d.png' % epoch)
        save_image(
            sample.data.view(64, self.input_channels,
                             self.imsize, self.imsize).transpose(2, 3),
            save_dir
        )

    def _dump_imgs_and_reconstructions(self, idxs, filename):
        imgs = []
        recons = []
        for i in idxs:
            img_np = self.train_dataset[i]
            img_torch = ptu.from_numpy(normalize_image(img_np))
            recon, *_ = self.model(img_torch.view(1, -1))

            img = img_torch.view(self.input_channels,
                                 self.imsize, self.imsize).transpose(1, 2)
            rimg = recon.view(self.input_channels, self.imsize,
                              self.imsize).transpose(1, 2)
            imgs.append(img)
            recons.append(rimg)
        all_imgs = torch.stack(imgs + recons)
        save_file = osp.join(logger.get_snapshot_dir(), filename)
        save_image(
            all_imgs.data,
            save_file,
            nrow=len(idxs),
        )

    def log_loss_under_uniform(self, model, data, priority_function_kwargs):
        import torch.nn.functional as F
        log_probs_prior = []
        log_probs_biased = []
        log_probs_importance = []
        kles = []
        mses = []
        for i in range(0, data.shape[0], self.batch_size):
            img = normalize_image(
                data[i:min(data.shape[0], i + self.batch_size), :])
            torch_img = ptu.from_numpy(img)
            reconstructions, obs_distribution_params, latent_distribution_params = self.model(
                torch_img)

            priority_function_kwargs['sampling_method'] = 'true_prior_sampling'
            log_p, log_q, log_d = compute_log_p_log_q_log_d(
                model, img, **priority_function_kwargs)
            log_prob_prior = log_d.mean()

            priority_function_kwargs['sampling_method'] = 'biased_sampling'
            log_p, log_q, log_d = compute_log_p_log_q_log_d(
                model, img, **priority_function_kwargs)
            log_prob_biased = log_d.mean()

            priority_function_kwargs['sampling_method'] = 'importance_sampling'
            log_p, log_q, log_d = compute_log_p_log_q_log_d(
                model, img, **priority_function_kwargs)
            log_prob_importance = (log_p - log_q + log_d).mean()

            kle = model.kl_divergence(latent_distribution_params)
            mse = F.mse_loss(torch_img, reconstructions,
                             reduction='elementwise_mean')
            mses.append(mse.item())
            kles.append(kle.item())
            log_probs_prior.append(log_prob_prior.item())
            log_probs_biased.append(log_prob_biased.item())
            log_probs_importance.append(log_prob_importance.item())

        logger.record_tabular(
            "Uniform Data Log Prob (True Prior)", np.mean(log_probs_prior))
        logger.record_tabular(
            "Uniform Data Log Prob (Biased)", np.mean(log_probs_biased))
        logger.record_tabular(
            "Uniform Data Log Prob (Importance)", np.mean(log_probs_importance))
        logger.record_tabular("Uniform Data KL", np.mean(kles))
        logger.record_tabular("Uniform Data MSE", np.mean(mses))

    def dump_uniform_imgs_and_reconstructions(self, dataset, epoch):
        idxs = np.random.choice(range(dataset.shape[0]), 4)
        filename = 'uniform{}.png'.format(epoch)
        imgs = []
        recons = []
        for i in idxs:
            img_np = dataset[i]
            img_torch = ptu.from_numpy(normalize_image(img_np))
            recon, *_ = self.model(img_torch.view(1, -1))

            img = img_torch.view(self.input_channels,
                                 self.imsize, self.imsize).transpose(1, 2)
            rimg = recon.view(self.input_channels, self.imsize,
                              self.imsize).transpose(1, 2)
            imgs.append(img)
            recons.append(rimg)
        all_imgs = torch.stack(imgs + recons)
        save_file = osp.join(logger.get_snapshot_dir(), filename)
        save_image(
            all_imgs.data,
            save_file,
            nrow=4,
        )
