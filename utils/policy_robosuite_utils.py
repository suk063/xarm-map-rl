import robosuite as suite
import robosuite.macros as macros
from robosuite.wrappers import GymWrapper
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config

import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as p
import torch.nn.functional as F
from torch.distributions.normal import Normal

import numpy as np
from typing import Union, Optional

macros.IMAGE_CONVENTION = "opencv"

np.set_printoptions(precision=4, suppress=True)

Activation = Union[str, nn.Module]
_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(0.2, inplace=True),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


class LegacyGymWrapper(GymWrapper):
    def __init__(self, env, keys):
        super(LegacyGymWrapper, self).__init__(env, keys)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, (terminated | truncated), info

    def reset(self):
        return super().reset()[0]        

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
        else:
            print("Seed must be an integer value!")

def make_robosuite_env(
    env_name, 
    robots="Panda", 
    controller_type='OSC_POSE', 
    render=False,
    use_camera_obs=False,
    **kwargs
):
    arm_controller_config = suite.load_part_controller_config(default_controller=controller_type)
    controller_configs = refactor_composite_controller_config(arm_controller_config, robots, ["right"])

    env = suite.make(
        env_name=env_name,
        robots=robots,
        reward_shaping=True,
        has_renderer=render,
        has_offscreen_renderer=use_camera_obs,
        use_camera_obs=use_camera_obs,
        use_object_obs=True,
        controller_configs=controller_configs,
        **kwargs,
    )
    env._max_episode_steps = env.horizon
    return env

def make(
    env_name, 
    robots="Panda", 
    controller_type='OSC_POSE', 
    obs_keys=None, 
    render=False,
    seed=1,
    **kwargs
):
    env = make_robosuite_env(
        env_name, 
        robots=robots, 
        controller_type=controller_type, 
        render=render,
        **kwargs
    )
    if obs_keys is None:
        obs_keys = [
            'robot0_eef_pos',
            'robot0_eef_quat',
            'robot0_gripper_qpos',
            'object-state',
        ]
    env = LegacyGymWrapper(env, keys=obs_keys)
    env.seed(seed)
    return env

def build_mlp(
    input_size: int,
    output_size: int,
    n_layers: int,
    size: int,
    activation: Optional[Activation] = 'relu',
    output_activation: Optional[Activation] = 'identity',
    spectral_norm: Optional[bool] = False
):
    """
        Builds a feedforward neural network
        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            scope: variable scope of the network
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer
            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer
        returns:
            output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layer = nn.Linear(in_size, size)
        if spectral_norm:
            layer = p.spectral_norm(layer)
        layers.append(layer)
        layers.append(activation)
        in_size = size
    layer = nn.Linear(in_size, output_size)
    if spectral_norm:
        layer = p.spectral_norm(layer)
    layers.append(layer)
    layers.append(output_activation)
    return nn.Sequential(*layers)


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


class ActorSAC(nn.Module):
    """MLP actor network."""
    def __init__(self, obs_dim, action_dim, n_layers, hidden_dim, device):
        super().__init__()

        self.log_std_min = -10
        self.log_std_max = 2
        self.device = device

        self.trunk = build_mlp(obs_dim, action_dim*2, n_layers, hidden_dim)

    def forward(self, obs, compute_log_pi=False): 

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        std = log_std.exp()
        noise = torch.randn_like(mu)
        pi = mu + noise * std

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        outputs = {'mu': mu, 'pi': pi, 'log_pi': log_pi, 'log_std': log_std}
        return outputs

    def sample_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs = torch.from_numpy(obs).float().to(self.device)
            obs = obs.unsqueeze(0)
            outputs = self(obs)
        
        act = outputs['mu'] if deterministic else outputs['pi']
        return act.cpu().data.numpy().flatten()

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, n_layers, hidden_dim, device):
        super().__init__()

        self.device = device

        self.trunk = build_mlp(state_dim, action_dim, 
            n_layers, hidden_dim,
            activation='relu', output_activation='tanh')

    def forward(self, state):
        h = self.trunk(state)
        return h

    def sample_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs = torch.from_numpy(obs).float().to(self.device)
            obs = obs.unsqueeze(0)        
            act = self(obs)
        
        act = act.cpu().data.numpy().flatten()
        if not deterministic:
            act += np.random.normal(0, 0.1, size=act.shape[0])
        return act


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

class ActorCustomInit(nn.Module):
    def __init__(self, state_dim, action_dim, n_layers, hidden_dim, device=None):
        super().__init__()

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.trunk = build_mlp(state_dim, action_dim, n_layers, 
            hidden_dim, activation='relu', output_activation='tanh')
        self.apply(weight_init)

    def forward(self, state):
        h = self.trunk(state)
        return h
    
    def sample_action(self, obs, deterministic=True):
        with torch.no_grad():
            obs = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
            act = self(obs)
        
        act = act.cpu().data.numpy().flatten()
        if not deterministic:
            act += np.random.normal(0, 0.1, size=act.shape[0])
        return act


class ActorBCManiSkill(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, device: Optional[torch.device] = None):
        super(ActorBCManiSkill, self).__init__()

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
    
    def sample_action(self, obs, deterministic=True):
        with torch.no_grad():
            obs = torch.Tensor(obs).unsqueeze(0).float().to(self.device)
            act = self(obs)
        
        act = act.cpu().data.numpy().flatten()
        if not deterministic:
            act += np.random.normal(0, 0.1, size=act.shape[0])
        return act


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorPPOManiSkill(nn.Module):
    def __init__(self, n_obs, n_act, device=None):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_obs, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1, device=device)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(n_obs, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, n_act, device=device), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, n_act, device=device))    
        self.device = device
        
    def get_value(self, x):
        return self.critic(x)    
    
    def get_action_and_value(self, obs, action=None):
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = action_mean + action_std * torch.randn_like(action_mean)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)
    
    def sample_action(self, obs, deterministic=True):
        with torch.no_grad():
            obs = torch.Tensor(obs).unsqueeze(0).float().to(self.device)
            act = self.actor_mean(obs)
        
        act = act.cpu().data.numpy().flatten()
        if not deterministic:
            act += np.random.normal(0, 0.1, size=act.shape[0])
        return act