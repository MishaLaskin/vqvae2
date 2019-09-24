import copy
import random
import warnings

import torch

import cv2
import numpy as np
from gym.spaces import Box, Dict
import rlkit.torch.pytorch_util as ptu
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict
from rlkit.envs.wrappers import ProxyEnv


class VQVAEWrappedEnv(ProxyEnv, MultitaskEnv):
    """This class wraps an image-based environment with a VAE.
    Assumes you get flattened (channels,84,84) observations from wrapped_env.
    This class adheres to the "Silent Multitask Env" semantics: on reset,
    it resamples a goal.
    """

    def __init__(
        self,
        wrapped_env,
        vae,
        vae_input_key_prefix='image',
        sample_from_true_prior=False,
        decode_goals=False,
        render_goals=False,
        render_rollouts=False,
        reward_params=None,
        goal_sampling_mode="vae_prior",
        imsize=84,
        obs_size=None,
        norm_order=2,
        epsilon=2,
        presampled_goals=None,

    ):
        if reward_params is None:
            reward_params = dict()
        super().__init__(wrapped_env)
        self.vae = vae
        # print(vae)
        if hasattr(vae, 'representation_size'):
            self.representation_size = self.vae.representation_size
        else:
            self.representation_size = None
        self.input_channels = self.vae.input_channels
        self.sample_from_true_prior = sample_from_true_prior
        self._decode_goals = decode_goals
        self.render_goals = render_goals
        self.render_rollouts = render_rollouts
        self.default_kwargs = dict(
            decode_goals=decode_goals,
            render_goals=render_goals,
            render_rollouts=render_rollouts,
        )
        self.imsize = imsize
        self.reward_params = reward_params
        self.reward_type = self.reward_params.get("type", 'latent_distance')
        self.norm_order = self.reward_params.get("norm_order", norm_order)
        self.epsilon = self.reward_params.get("epsilon", epsilon)
        self.reward_min_variance = self.reward_params.get("min_variance", 0)
        latent_space = Box(
            -10 * np.ones(obs_size or self.representation_size),
            10 * np.ones(obs_size or self.representation_size),
            dtype=np.float32,
        )
        spaces = self.wrapped_env.observation_space.spaces
        spaces['observation'] = latent_space
        spaces['desired_goal'] = latent_space
        spaces['achieved_goal'] = latent_space
        spaces['latent_observation'] = latent_space
        spaces['latent_desired_goal'] = latent_space
        spaces['latent_achieved_goal'] = latent_space
        self.observation_space = Dict(spaces)
        self._presampled_goals = presampled_goals
        if self._presampled_goals is None:
            self.num_goals_presampled = 0
        else:
            self.num_goals_presampled = presampled_goals[random.choice(
                list(presampled_goals))].shape[0]

        self.vae_input_key_prefix = vae_input_key_prefix
        assert vae_input_key_prefix in {'image', 'image_proprio'}
        self.vae_input_observation_key = vae_input_key_prefix + '_observation'
        self.vae_input_achieved_goal_key = vae_input_key_prefix + '_achieved_goal'
        self.vae_input_desired_goal_key = vae_input_key_prefix + '_desired_goal'
        self._mode_map = {}
        self.desired_goal = {'latent_desired_goal': latent_space.sample()}
        self._initial_obs = None
        self._custom_goal_sampler = None
        self._goal_sampling_mode = goal_sampling_mode
        self.running_obs_mean = np.zeros(self.representation_size)+1e-3
        self.running_obs_std = np.zeros(self.representation_size)+1e-3
        self.running_k = 1
        #assert False, 'Using VQVAE'

    def reset(self):
        obs = self.wrapped_env.reset()
        goal = self.sample_goal()

        self.set_goal(goal)

        self._initial_obs = obs

        return self._update_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        new_obs = self._update_obs(obs)
        self._update_info(info, new_obs)
        reward = self.compute_reward(
            action,
            {'latent_achieved_goal': new_obs['latent_achieved_goal'],
             'latent_desired_goal': new_obs['latent_desired_goal']}
        )
        self.try_render(new_obs)
        return new_obs, reward, done, info

    def _update_obs(self, obs):
        latent_obs = self._encode_one(obs[self.vae_input_observation_key])

        obs['latent_observation'] = latent_obs
        obs['latent_achieved_goal'] = latent_obs
        obs['observation'] = latent_obs
        obs['achieved_goal'] = latent_obs
        obs = {**obs, **self.desired_goal}
        return obs

    def _update_info(self, info, obs):
        # CHANGE TO VQ VAE
        latent_distribution_params = self.vae.encode(
            ptu.from_numpy(obs[self.vae_input_observation_key].reshape(1, -1))
        )
        latent_obs, logvar = ptu.get_numpy(latent_distribution_params[0])[
            0], ptu.get_numpy(latent_distribution_params[1])[0]
        # assert (latent_obs == obs['latent_observation']).all()
        latent_goal = self.desired_goal['latent_desired_goal']
        dist = latent_goal - latent_obs
        #var = np.exp(logvar.flatten())
        #var = np.maximum(var, self.reward_min_variance)

        # m_k = m_{k-1} + (x_k-m_{k-1})/k
        m_k_prev = self.running_obs_mean.copy()
        x_k = latent_obs
        k = self.running_k
        m_k = m_k_prev + (x_k - m_k_prev)/k

        # s_k = s_{k-1} + (x-m_{k-1})*(x-m_{k})

        s_k_prev = self.running_obs_std.copy()
        s_k = s_k_prev + (x_k - m_k_prev)*(x_k - m_k)

        # var = s_k / (k-1) if k >= 2

        var = s_k / np.max([1, k-1])
        self.running_obs_mean = m_k
        self.running_obs_std = s_k
        self.running_k = k + 1

        #print('m_k', self.running_obs_mean)
        #print('s_k', self.running_obs_std)
        #print('k', self.running_k)

        err = dist * dist / 2 / var
        mdist = np.sum(err)  # mahalanobis distance
        #print('err', err)
        #print('mdist', mdist)

        # get the right bits
        b1 = self.discretize_z(latent_obs)
        b2 = self.discretize_z(latent_goal)
        bits_matched = np.sum(b1 == b2)/(self.vae.z_dim**2)

        info["vae_mdist"] = mdist
        info["vae_dist"] = np.linalg.norm(dist, ord=self.norm_order)
        info["vae_dist_l1"] = np.linalg.norm(dist, ord=1)
        info["vae_dist_l2"] = np.linalg.norm(dist, ord=2)
        info["vae_success"] = 1 if mdist < self.epsilon else 0
        info['bits_matched'] = bits_matched
        if False:

            print('latent goal', latent_goal)
            print('latent obs', latent_obs)
            print('dist', dist)
            for k, v in info.items():
                print(k, v)
            print('margin', self.epsilon)
            assert False

    """
    Multitask functions
    """

    def sample_goals(self, batch_size):
        # CHANGE TO VQ VAE
        # TODO: make mode a parameter you pass in
        if self._goal_sampling_mode == 'custom_goal_sampler':
            return self.custom_goal_sampler(batch_size)
        elif self._goal_sampling_mode == 'presampled':
            idx = np.random.randint(0, self.num_goals_presampled, batch_size)
            sampled_goals = {
                k: v[idx] for k, v in self._presampled_goals.items()
            }
            # ensures goals are encoded using latest vae
            if 'image_desired_goal' in sampled_goals:
                sampled_goals['latent_desired_goal'] = self._encode(
                    sampled_goals['image_desired_goal'])
            return sampled_goals
        elif self._goal_sampling_mode == 'env':
            goals = self.wrapped_env.sample_goals(batch_size)
            latent_goals = self._encode(goals[self.vae_input_desired_goal_key])
        elif self._goal_sampling_mode == 'reset_of_env':
            assert batch_size == 1
            goal = self.wrapped_env.get_goal()
            goals = {k: v[None] for k, v in goal.items()}
            latent_goals = self._encode(
                goals[self.vae_input_desired_goal_key]
            )
        elif self._goal_sampling_mode == 'vae_prior':
            goals = {}
            latent_goals = self._sample_vae_prior(batch_size)
        else:
            raise RuntimeError("Invalid: {}".format(self._goal_sampling_mode))

        if self._decode_goals:
            decoded_goals = self._decode(latent_goals)
        else:
            decoded_goals = None
        image_goals, proprio_goals = self._image_and_proprio_from_decoded(
            decoded_goals
        )

        goals['desired_goal'] = latent_goals
        goals['latent_desired_goal'] = latent_goals
        if proprio_goals is not None:
            goals['proprio_desired_goal'] = proprio_goals
        if image_goals is not None:
            goals['image_desired_goal'] = image_goals
        if decoded_goals is not None:
            goals[self.vae_input_desired_goal_key] = decoded_goals
        return goals

    def get_goal(self):
        return self.desired_goal

    def compute_reward(self, action, obs):
        actions = action[None]
        next_obs = {
            k: v[None] for k, v in obs.items()
        }
        return self.compute_rewards(actions, next_obs)[0]

    def compute_rewards(self, actions, obs):
        # TODO: implement log_prob/mdist
        if self.reward_type == 'latent_distance':
            achieved_goals = obs['latent_achieved_goal']
            desired_goals = obs['latent_desired_goal']
            dist = np.linalg.norm(
                desired_goals - achieved_goals, ord=self.norm_order, axis=1)
            return -dist
        elif self.reward_type == 'vectorized_latent_distance':
            achieved_goals = obs['latent_achieved_goal']
            desired_goals = obs['latent_desired_goal']
            return -np.abs(desired_goals - achieved_goals)
        elif self.reward_type == 'latent_sparse':
            achieved_goals = obs['latent_achieved_goal']
            desired_goals = obs['latent_desired_goal']

            # np.linalg.norm(desired_goals - achieved_goals, ord=self.norm_order, axis=1)
            dist = desired_goals - achieved_goals

            var = self.running_obs_std / np.max([1, self.running_k-1])

            #print('dist', dist.shape)
            #print('var', var.shape)
            errs = dist**2 / 2 / var[None, :]
            mdists = np.sum(errs, 1)

            n = achieved_goals.shape[0]
            b1 = self.discretize_z(achieved_goals, n)
            b2 = self.discretize_z(desired_goals, n)
            bit_mask = b1 == b2
            bits_matched = np.sum(bit_mask, axis=1)

            #print('mdists', mdists.shape, mdists)
            # l2
            # reward = np.array(
            #    [0 if d < self.epsilon else -1 for d in np.linalg.norm(dist, axis=1)])
            # mdist
            # reward = np.array(
            #    [0 if d < self.epsilon else -1 for d in np.linalg.norm(dist, axis=1)])
            reward = (1.*bits_matched/(self.vae.z_dim**2) - 1.)
            return reward
        elif self.reward_type == 'state_distance':
            achieved_goals = obs['state_achieved_goal']
            desired_goals = obs['state_desired_goal']
            return - np.linalg.norm(desired_goals - achieved_goals, ord=self.norm_order, axis=1)
        elif self.reward_type == 'wrapped_env':
            return self.wrapped_env.compute_rewards(actions, obs)
        else:
            raise NotImplementedError

    @property
    def goal_dim(self):
        return self.representation_size

    def set_goal(self, goal):
        """
        Assume goal contains both image_desired_goal and any goals required for wrapped envs

        :param goal:
        :return:
        """
        self.desired_goal = goal
        # TODO: fix this hack / document this
        if self._goal_sampling_mode in {'presampled', 'env'}:
            self.wrapped_env.set_goal(goal)

    def get_diagnostics(self, paths, **kwargs):
        # CHANGE TO VQ VAE
        statistics = self.wrapped_env.get_diagnostics(paths, **kwargs)
        for stat_name_in_paths in ["vae_mdist", "vae_success", "vae_dist"]:
            stats = get_stat_in_paths(paths, 'env_infos', stat_name_in_paths)
            statistics.update(create_stats_ordered_dict(
                stat_name_in_paths,
                stats,
                always_show_all_stats=True,
            ))
            final_stats = [s[-1] for s in stats]
            statistics.update(create_stats_ordered_dict(
                "Final " + stat_name_in_paths,
                final_stats,
                always_show_all_stats=True,
            ))
        return statistics

    """
    Other functions
    """
    @property
    def goal_sampling_mode(self):
        return self._goal_sampling_mode

    @goal_sampling_mode.setter
    def goal_sampling_mode(self, mode):
        assert mode in [
            'custom_goal_sampler',
            'presampled',
            'vae_prior',
            'env',
            'reset_of_env'
        ], "Invalid env mode"
        self._goal_sampling_mode = mode
        if mode == 'custom_goal_sampler':
            test_goals = self.custom_goal_sampler(1)
            if test_goals is None:
                self._goal_sampling_mode = 'vae_prior'
                warnings.warn(
                    "self.goal_sampler returned None. " +
                    "Defaulting to vae_prior goal sampling mode"
                )

    @property
    def custom_goal_sampler(self):
        return self._custom_goal_sampler

    @custom_goal_sampler.setter
    def custom_goal_sampler(self, new_custom_goal_sampler):
        assert self.custom_goal_sampler is None, (
            "Cannot override custom goal setter"
        )
        self._custom_goal_sampler = new_custom_goal_sampler

    @property
    def decode_goals(self):
        return self._decode_goals

    @decode_goals.setter
    def decode_goals(self, _decode_goals):
        self._decode_goals = _decode_goals

    def get_env_update(self):
        """
        For online-parallel. Gets updates to the environment since the last time
        the env was serialized.

        subprocess_env.update_env(**env.get_env_update())
        """
        return dict(
            mode_map=self._mode_map,
            gpu_info=dict(
                use_gpu=ptu._use_gpu,
                gpu_id=ptu._gpu_id,
            ),
            vae_state=self.vae.__getstate__(),
        )

    def update_env(self, mode_map, vae_state, gpu_info):
        self._mode_map = mode_map
        self.vae.__setstate__(vae_state)
        gpu_id = gpu_info['gpu_id']
        use_gpu = gpu_info['use_gpu']
        ptu.device = torch.device("cuda:" + str(gpu_id) if use_gpu else "cpu")
        self.vae.to(ptu.device)

    def enable_render(self):
        self._decode_goals = True
        self.render_goals = True
        self.render_rollouts = True

    def disable_render(self):
        self._decode_goals = False
        self.render_goals = False
        self.render_rollouts = False

    def try_render(self, obs):
        if self.render_rollouts:
            img = obs['image_observation'].reshape(
                self.input_channels,
                self.imsize,
                self.imsize,
            ).transpose()
            cv2.imshow('env', img)
            cv2.waitKey(1)
            reconstruction = self._reconstruct_img(
                obs['image_observation']).transpose()
            cv2.imshow('env_reconstruction', reconstruction)
            cv2.waitKey(1)
            init_img = self._initial_obs['image_observation'].reshape(
                self.input_channels,
                self.imsize,
                self.imsize,
            ).transpose()
            cv2.imshow('initial_state', init_img)
            cv2.waitKey(1)
            init_reconstruction = self._reconstruct_img(
                self._initial_obs['image_observation']
            ).transpose()
            cv2.imshow('init_reconstruction', init_reconstruction)
            cv2.waitKey(1)

        if self.render_goals:
            goal = obs['image_desired_goal'].reshape(
                self.input_channels,
                self.imsize,
                self.imsize,
            ).transpose()
            cv2.imshow('goal', goal)
            cv2.waitKey(1)

    def discretize_z(self, z, batch_size=1):
        z_dim = self.vae.z_dim

        #vq_encoder_output = model.pre_quantization_conv(model.encoder(x))
        z = torch.tensor(z).float().to(ptu.device)
        z = z.view(batch_size, 2, z_dim, z_dim)
        _, _, _, _, e_indices = self.vae.vector_quantization(z)

        b = e_indices.detach().cpu().view(batch_size, -1).numpy()
        return b

    def _sample_vae_prior(self, batch_size):
        """
        if self.sample_from_true_prior:
            mu, sigma = 0, 1  # sample from prior
        else:
            mu, sigma = self.vae.dist_mu, self.vae.dist_std
        n = np.random.randn(batch_size, self.representation_size)
        """

        z_dim = self.vae.z_dim
        labels = torch.zeros(batch_size).long().to(ptu.device)

        samples = self.vae.pixelcnn.generate(
            labels, shape=(z_dim, z_dim), batch_size=batch_size)
        samples = torch.tensor(samples).reshape(-1, 1).long().to(ptu.device)

        #print('samples', samples, samples.shape)

        e_weights = self.vae.vector_quantization.embedding.weight
        weight_dim = np.sqrt(e_weights.shape[0]).astype(np.int32)
        min_encodings = torch.zeros(
            samples.shape[0], weight_dim**2).long().to(ptu.device)
        # print(min_encodings.shape)
        min_encodings.scatter_(1, samples, 1)

        #print('min_encodings', min_encodings, min_encodings.shape)
        #print(min_encodings[:9, :])
        # assert False

        # print(min_encodings.shape, e_weights.shape,
        #      np.sqrt(e_weights.shape[0]).astype(np.int32))
        # assert False
        embedding_dim = 2
        z_q = torch.matmul(min_encodings.float(),
                           e_weights.float()).view((batch_size, z_dim, z_dim, embedding_dim))
        # print(z_q.sum(3)[0])
        #assert False
        #z_q = z_q.permute(0, 3, 1, 2).contiguous()
        #z_q = self.vae.pre_dequantization_conv(z_q.float())
        #x_recon = self.vae.decoder(z_q)
        # print('z_q', z_q, z_q.shape)
        z_q = z_q.view(batch_size, -1).detach().cpu().numpy()

        # if batch_size == 1:
        #    return z_q.reshape(-1)
        return z_q

    def _decode(self, latents):
        #print('decode A', latents.shape)
        embedding_dim = 2
        z_dim = self.vae.z_dim
        batch_size = latents.shape[0]
        latents = latents.reshape(batch_size, z_dim, z_dim, embedding_dim)
        latents = torch.tensor(latents).float().to(ptu.device)
        latents = latents.permute(0, 3, 1, 2).contiguous()
        latents = self.vae.pre_dequantization_conv(latents.float())

        reconstructions = self.vae.decoder(latents)
        decoded = ptu.get_numpy(reconstructions)
        decoded = decoded.reshape(batch_size, -1)
        #print('decode B', decoded.shape)
        return decoded

    def _encode_one(self, img):
        return self._encode(img[None])[0]

    def _encode(self, imgs):
        latent_distribution_params = self.vae.encode(ptu.from_numpy(imgs))
        return ptu.get_numpy(latent_distribution_params[0])

    def _reconstruct_img(self, flat_img):
        latent_distribution_params = self.vae.encode(
            ptu.from_numpy(flat_img.reshape(1, -1)))

        reconstructions = self._decode(latent_distribution_params[0])
        #reconstructions, _ = self.vae.decode(latent_distribution_params[0])
        imgs = ptu.get_numpy(reconstructions)
        imgs = imgs.reshape(
            1, self.input_channels, self.imsize, self.imsize
        )
        return imgs[0]

    def _image_and_proprio_from_decoded(self, decoded):
        if decoded is None:
            return None, None
        if self.vae_input_key_prefix == 'image_proprio':
            images = decoded[:, :self.image_length]
            proprio = decoded[:, self.image_length:]
            return images, proprio
        elif self.vae_input_key_prefix == 'image':
            return decoded, None
        else:
            raise AssertionError("Bad prefix for the vae input key.")

    def __getstate__(self):
        state = super().__getstate__()
        state = copy.copy(state)
        state['_custom_goal_sampler'] = None
        warnings.warn('VAEWrapperEnv.custom_goal_sampler is not saved.')
        return state

    def __setstate__(self, state):
        warnings.warn('VAEWrapperEnv.custom_goal_sampler was not loaded.')
        super().__setstate__(state)


def temporary_mode(env, mode, func, args=None, kwargs=None):
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    cur_mode = env.cur_mode
    env.mode(env._mode_map[mode])
    return_val = func(*args, **kwargs)
    env.mode(cur_mode)
    return return_val
