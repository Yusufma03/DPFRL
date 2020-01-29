import sys
import os
import time
import logging
import collections
import multiprocessing
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from gym.envs.registration import register

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from torch.optim.lr_scheduler import LambdaLR

from envs import make_env
from storage import RolloutStorage
import utils
from tensorboardX import SummaryWriter
from PIL import Image
from pfrnn_model import PFRNN_Policy

# Create Sacred Experiment
ex = Experiment("POMRL")
ex.captured_out_filter = apply_backspaces_and_linefeeds
# np.seterr(all='raise')

# Get name of environment yaml file.
# Should be specified in command line using
# 'python main.py with environment.config_file=name_of_env_config_file.yaml'
environment_yaml = utils.get_environment_yaml(ex)

# Add defautl.yaml and the <environment_name>.yaml file to the sacred configuration
DIR = os.path.dirname(sys.argv[0])
DIR = '.' if DIR == '' else DIR
ex.add_config(DIR + '/conf/default.yaml')
ex.add_config(DIR + '/conf/' + environment_yaml)

from sacred.observers import FileStorageObserver
ex.observers.append(FileStorageObserver.create('saved_runs'))

saved_runs = os.listdir('./saved_runs/')
tmp = []
for i in saved_runs:
    if '_' not in i:
        tmp.append(int(i))

saved_runs = tmp
try:
    cnt = max(saved_runs) + 1
except:
    cnt = 0
log_path = os.path.join('./tfboard_runs/', str(cnt))
writer = SummaryWriter(log_path)

# The background mask value is stored in a json file
background_config = DIR + '/environments/background_values.json'
import json
with open(background_config, 'r') as fin:
    bk_config = json.load(fin)

# This function is called by sacred before the experiment is started
# All args are provided by sacred and filled with configuration values
@ex.config
def general_config(cuda, algorithm, environment, rl_setting, loss_function, log):
    """
    - Sets device=cuda or, if cuda is 'auto' sets it depending on availability of cuda
    - Entries in algorithm.model are overriden with new values from environment.model_adaptation
    - Entries in rl_setting are overriden with new values from environment.rl_setting_adaptation
    - algorithm.model.batch_size is set to rl_setting.num_processes
    """

    if cuda == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cuda

    # This updates values in environment.model based on values in environment.model_adaption
    # This allows environment specific model configuration to be specified in the environment.yaml
    for key1 in environment['model_adaptation']:
        algorithm['model'][key1] = environment['model_adaptation'][key1]

    # Same for values in rl_setting
    for key2 in environment['rl_setting_adaptation']:
        rl_setting[key2] = environment['rl_setting_adaptation'][key2]

    # Delete keys so we don't have them in the sacred configuration
    del key1
    del key2

    algorithm['model']['batch_size'] = rl_setting['num_processes']

    from sys import platform
    if platform == "darwin":
        # rl_setting['num_processes'] = 2
        if environment['config_file'] == 'openaiEnv.yaml':
            # Workaround for bug in openCV on MacOS
            # Problem araises in WarpFrame wrapper in cv2
            # See here: https://github.com/opencv/opencv/issues/5150
            multiprocessing.set_start_method('spawn')

    imgnet_train = './train_imgnet.pkl'
    imgnet_test = './test_imgnet.pkl'
    game_name = environment['name'].split('No')[0]
    try:
        bk_value = bk_config[game_name]
    except:
        bk_value = None
    del game_name


@ex.command(unobserved=True)
def setup(rl_setting, device, _run, _log, log, seed, cuda):
    """
    Do everything required to set up the experiment:
    - Create working dir
    - Set's cuda seed (numpy is set by sacred)
    - Set and configure logger
    - Create n_e environments
    - Create model
    - Create 'RolloutStorage': A helper class to save rewards and compute the advantage loss
    - Creates and initialises current_memory, a dictionary of (for each of the n_e environment):
      - past observation
      - past latent state
      - past action
      - past reward
      This is used as input to the model to compute the next action.

    Args:
        All args are automatically provided by sacred by passing the equally named configuration
        variables that are either defined in the yaml files or the command line.

    Returns:
        id_temp_dir (str): The newly created working directory
        envs: Vector of environments
        actor_critic: The model
        rollouts: A helper class (RolloutStorage) to store rewards and compute TD errors
        current_memory: Dictionary to keep track of current obs, actions, latent states and rewards
    """

    # Create working dir
    id_tmp_dir = "{}/{}/".format(log['tmp_dir'], _run._id)
    utils.safe_make_dirs(id_tmp_dir)

    np.set_printoptions(precision=2)

    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    logger = logging.getLogger()
    if _run.debug or _run.pdb:
        logger.setLevel(logging.DEBUG)

    envs = register_and_create_Envs(id_tmp_dir, train=True)
    test_envs = register_and_create_Envs(id_tmp_dir, train=False)

    actor_critic = create_model(envs)

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0], *obs_shape[1:])

    rollouts = RolloutStorage(rl_setting['num_steps'], rl_setting['num_processes'], obs_shape,
                              envs.action_space)
    current_obs = torch.zeros(rl_setting['num_processes'], *obs_shape)

    obs = envs.reset()
    test_obs = test_envs.reset()
    if not actor_critic.observation_type == 'fc':
        obs = obs / 255.
        test_obs = test_obs / 255.
    current_obs = torch.from_numpy(obs).float()
    current_obs_test = torch.from_numpy(test_obs).float()
    # init_states = Variable(torch.zeros(rl_setting['num_processes'], actor_critic.state_size))
    init_states = actor_critic.new_latent_state()
    init_states_test = actor_critic.new_latent_state()

    init_rewards = torch.zeros([rl_setting['num_processes'], 1])
    init_rewards_test = torch.zeros([rl_setting['num_processes'], 1])

    if envs.action_space.__class__.__name__ == "Discrete":
        action_shape = 1
    else:
        action_shape = envs.action_space.shape[0]
    init_actions = torch.zeros(rl_setting['num_processes'], action_shape)
    init_actions_test = torch.zeros(rl_setting['num_processes'], action_shape)

    try:
        init_states = init_states.to(device)
        init_states_test = init_states_test.to(device)
    except:
        init_states = utils.cudify_state(init_states, device)
        init_states_test = utils.cudify_state(init_states_test, device)

    init_actions = init_actions.to(device)
    current_obs = current_obs.to(device)
    init_rewards = init_rewards.to(device)

    init_actions_test = init_actions_test.to(device)
    current_obs_test = current_obs_test.to(device)
    init_rewards_test = init_rewards_test.to(device)

    actor_critic.to(device)
    rollouts.to(device)

    current_memory = {
        'current_obs': current_obs,
        'states': init_states,
        'oneHotActions': utils.toOneHot(
            envs.action_space,
            init_actions),
        'rewards': init_rewards
    }

    current_memory_test = {
        'current_obs': current_obs_test,
        'states': init_states_test,
        'oneHotActions': utils.toOneHot(
            envs.action_space,
            init_actions_test),
        'rewards': init_rewards_test
    }

    return id_tmp_dir, envs, test_envs, actor_critic, rollouts, current_memory,\
        current_memory_test


@ex.command
def create_model(envs, algorithm, rl_setting, environment):
    """
    Creates the actor-critic model.

    Note that those values can be overwritten by the environment (see config() function).

    Args:
        envs: Vector of environments. Usually created by register_and_create_Envs()
        All other args are automatically provided by sacred by passing the equally named
        configuration variables that are either defined in the yaml files or the command line.

    Returns:
        model: The actor_critic model
    """
    action_space = envs.action_space
    nr_inputs = envs.observation_space.shape[0]

    # Pass in configuration only from algorithm.model
    model = PFRNN_Policy(
        action_space,
        nr_inputs,
        **algorithm['model'])

    return model


@ex.capture
def register_and_create_Envs(id_tmp_dir, seed, environment, rl_setting, train):
    """
    Register environment, create vector of n_e environments and return it.

    Args:
        id_temp_dir (str): Working directory.
        All other args are automatically provided by sacred by passing the equally named
        configuration variables that are either defined in the yaml files or the command line.

    """
    if environment['entry_point']:
        try:
            register(
                id=environment['name'],
                entry_point=environment['entry_point'],
                kwargs=environment['config'],
                max_episode_steps=environment['max_episode_steps']
            )
        except Exception:
            pass

    envs = [make_env(environment['name'], seed, i, id_tmp_dir,
                     frameskips_cases=environment['frameskips_cases'],
                     train=train)
            for i in range(rl_setting['num_processes'])]

    # Vectorise envs
    if rl_setting['num_processes'] > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    # Normalise rewards. Unnecessary for Atari, unwanted for Mountain Hike.
    # Probably useful for MuJoCo?
    # if len(envs.observation_space.shape) == 1:
    if environment['vec_norm']:
        envs = VecNormalize(envs)

    return envs


@ex.capture
def noisify_obs(obs, video, pointer, envs, rl_setting, environment, bk_value):
    vid_len = video.shape[1]

    def get_bk_mask(obs, bk_value):
        mask = np.abs(obs - bk_value) < 1e-3
        return mask

    def get_quarter_mask(indices):
        masks = [
                [[0, 1], [1, 1]],
                [[1, 0], [1, 1]],
                [[1, 1], [0, 1]],
                [[1, 1], [1, 0]]
                ]
        masks = np.array(masks)

        selected_masks = masks[indices]

        selected_masks = np.repeat(selected_masks, 42, axis=1)
        selected_masks = np.repeat(selected_masks, 42, axis=2)

        return selected_masks

    n_type = environment['noise_type']

    if 'back' in n_type:
        fake_imgs = batch_sample_imgs(video, pointer)
        fake_imgs = fake_imgs.astype(float) / 255

    if 'blank' in n_type:
        blank_mask = np.random.choice(
            [0, 1],
            size=rl_setting['num_processes'],
            p=[environment['p_blank'], 1-environment['p_blank']])
        obs_dims = [1] * len(envs.observation_space.shape)
        blank_mask = np.reshape(
            blank_mask,
            (rl_setting['num_processes'], *obs_dims))
        if n_type == 'blank':
            return obs * blank_mask, pointer, blank_mask

    if 'back' in n_type:
        mask = get_bk_mask(obs, bk_value)
        pointer = (pointer[0], (pointer[1]+1) % vid_len)
        noise_obs = obs * (1 - mask) + fake_imgs * mask

        if n_type == 'back':
            return noise_obs, pointer, mask
        elif n_type == 'blank_back':
            return noise_obs * blank_mask + fake_imgs * (1 - blank_mask), pointer, blank_mask
        else:
            raise NotImplementedError

    if n_type == 'normal':
        # ugly hack to create the mask
        mask = get_bk_mask(obs, bk_value)
        return obs, pointer, mask

@ex.capture
def run_model(actor_critic, current_memory, envs,
              environment, rl_setting, device, algorithm, 
              video,
              pointer):
    """
    Runs the model.

    Args:
        actor_critic: Agent model
        current_memory: Dict with past observations, actions, latent_states and rewards
        envs: Vector of n_e environments
        All other args are automatically provided by sacred by passing the equally named
        configuration variables that are either defined in the yaml files or the command line.

    Returns:
        policy_return: Named tuple with, besides other values, new latent state, V, a, log p(a),
                       H[p(a)], encding loss L^{ELBO} 
    """

    # Run model
    policy_return = actor_critic(
        current_memory=current_memory
        )

    # Execute on environment
    cpu_actions = policy_return.action.detach().squeeze(1).cpu().numpy()
    obs, reward, done, info = envs.step(cpu_actions)
    if not actor_critic.observation_type == 'fc':
        obs = obs / 255.

    if environment['name'] != 'DeathValley-v0':
        obs, new_pointer, blank_mask = noisify_obs(obs, video, pointer, envs)
        new_vid_id = np.random.randint(0, len(video), size=rl_setting['num_processes'])
        new_vid_id = new_pointer[0] * (1 - done) + new_vid_id * done
        new_img_id = np.random.randint(0, video.shape[1],
                size=rl_setting['num_processes'])
        new_img_id = new_pointer[1] * (1 - done) + new_img_id * done

        new_pointer = (new_vid_id, new_img_id)
    else:
        blank_mask = np.random.choice(
            [0, 1],
            size=rl_setting['num_processes'],
            p=[environment['p_blank'], 1-environment['p_blank']])
        obs_dims = [1] * len(envs.observation_space.shape)
        blank_mask = np.reshape(
            blank_mask,
            (rl_setting['num_processes'], *obs_dims))
        obs = obs * blank_mask
        new_pointer = (None, None)

    obs = torch.from_numpy(obs).float()
    
    # Make reward into tensor so we can use it as input to model
    reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()

    # If trajectory ended, create mask to clean reset actions and latent states
    masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
    masks = masks.to(device)

    # Update current_memory
    # current_memory['current_obs'] = torch.from_numpy(obs).float()
    current_memory['current_obs'] = obs

    # Create new latent states for new episodes
    current_memory['states'] = actor_critic.vec_conditional_new_latent_state(
        policy_return.latent_state,
        masks)

    # Set first action to 0 for new episodes
    # Also, if action is discrete, convert it to one-hot vector
    current_memory['oneHotActions'] = utils.toOneHot(
        envs.action_space,
        policy_return.action * masks.type(policy_return.action.type()))

    current_memory['rewards'][:] = reward

    return policy_return, current_memory, blank_mask, masks, reward, new_pointer


@ex.capture
def track_values(tracked_values, policy_return, algorithm):
    """
    Save various values from policy_return into 'tracked_values', a dictionary of lists (or deques).

    Args:
        tracked_values: Dictionary of list-like objects.
        policy_return: Named tuple returned by actor_critic model
        old_observation
    """

    tracked_values['values'].append(policy_return.value_estimate)
    tracked_values['action_log_probs'].append(policy_return.action_log_probs)
    tracked_values['dist_entropy'].append(policy_return.dist_entropy)

    return tracked_values

@ex.capture
def track_rewards(tracked_rewards, reward, masks, blank_mask, rl_setting,
        environment):
    masks = masks.cpu()
    # Initialise first time

    # Track episode and final rewards as well as how many episodes have ended so var
    tracked_rewards['episode_rewards'] += reward
    tracked_rewards['num_ended_episodes'] += rl_setting['num_processes'] - sum(masks)[0]
    tracked_rewards['final_rewards'] *= masks
    tracked_rewards['final_rewards'] += (1 - masks) * tracked_rewards['episode_rewards']
    tracked_rewards['episode_rewards'] *= masks
    if environment['name'] != "DeathValley-v0":
        n_type = environment['noise_type']
        if 'quarter' not in n_type and 'back' not in n_type and 'normal' not in\
            n_type:
            tracked_rewards['nr_observed_screens'].append(float(sum(blank_mask)))
        else:
            tracked_rewards['nr_observed_screens'].append(float(len(blank_mask)))
    else:
        tracked_rewards['nr_observed_screens'].append(float(len(blank_mask)))

    avg_nr_observed = (sum(list(tracked_rewards['nr_observed_screens'])[0:rl_setting['num_steps']])
                       / rl_setting['num_steps'] / rl_setting['num_processes'])

    return tracked_rewards['final_rewards'], avg_nr_observed, tracked_rewards['num_ended_episodes']

@ex.capture
def load_imgnet_images(imgnet_train, imgnet_test):
    import pickle
    print("start to load imagenet videos")

    with open(imgnet_train, 'rb') as fin:
        train_trajs = pickle.load(fin)

    with open(imgnet_test, 'rb') as fin:
        test_trajs = pickle.load(fin)
    
    return train_trajs, test_trajs

@ex.capture
def batch_sample_imgs(video, pointers):
    imgs = video[pointers[0], pointers[1], :, :, :]
    return imgs

@ex.automain
def main(_run,
         seed,
         opt,
         environment,
         rl_setting,
         log,
         algorithm,
         loss_function):
    """
    Entry point. Contains main training loop.
    """

    # Setup directory, vector of environments, actor_critic model, a 'rollouts' helper class
    # to compute target values and 'current_memory' which maintains the last action/observation/latent_state values
    id_tmp_dir, envs, test_envs, actor_critic, rollouts, current_memory,\
        current_memory_test = setup()

    num_processes = rl_setting['num_processes']

    if environment['name'] != 'DeathValley-v0':
        img_trajs, img_trajs_test = load_imgnet_images()

        traj_pointer = np.random.randint(0, img_trajs.shape[0],
                rl_setting['num_processes'])
        img_pointer = np.random.randint(0, img_trajs.shape[1],
                rl_setting['num_processes'])

        traj_pointer_test = np.random.randint(0, img_trajs_test.shape[0],
                rl_setting['num_processes'])
        img_pointer_test = np.random.randint(0, img_trajs_test.shape[1],
                rl_setting['num_processes'])

    else:
        img_trajs = None
        traj_pointer = None
        img_pointer = None

        img_trajs_test = None
        traj_pointer_test = None
        img_pointer_test = None

    tracked_rewards = {
        # Used to tracked how many screens weren't blanked out. Usually not needed
        'nr_observed_screens': collections.deque([0], maxlen=rl_setting['num_steps'] + 1),
        'episode_rewards': torch.zeros([rl_setting['num_processes'], 1]),
        'final_rewards': torch.zeros([rl_setting['num_processes'], 1]),
        'num_ended_episodes': 0
    }

    num_updates = int(float(loss_function['num_frames'])
                      // rl_setting['num_steps']
                      // rl_setting['num_processes'])

    # Count parameters
    num_parameters = 0
    for p in actor_critic.parameters():
        num_parameters += p.nelement()

    # Initialise optimiser
    if opt['optimizer'] == 'RMSProp':
        optimizer = optim.RMSprop(actor_critic.parameters(), opt['lr'],
                                  eps=opt['eps'], alpha=opt['alpha'])
    elif opt['optimizer'] == 'Adam':
        optimizer = optim.Adam(actor_critic.parameters(), opt['lr'],
                               eps=opt['eps'], betas=opt['betas'])
    else:
        raise NotImplementedError

    lr_scheduler = LambdaLR(optimizer=optimizer, 
                    lr_lambda=lambda x: utils.linear_decay(x, num_updates))

    print(actor_critic)
    logging.info('Number of parameters =\t{}'.format(num_parameters))
    logging.info("Total number of updates: {}".format(num_updates))
    logging.info("Learning rate: {}".format(opt['lr']))

    start = time.time()

    # Main training loop
    for j in range(num_updates):

        # Main Loop over n_s steps for one gradient update
        tracked_values = collections.defaultdict(lambda: [])

        for step in range(rl_setting['num_steps']):

            old_observation = current_memory['current_obs']

            policy_return, current_memory, blank_mask, masks, reward,\
                (traj_pointer, img_pointer) = run_model(
                actor_critic=actor_critic,
                current_memory=current_memory,
                envs=envs,
                video=img_trajs,
                pointer=(traj_pointer, img_pointer))


            if step % 50 == 0:
                with torch.no_grad():
                    _, current_obs_test, _, _, _, (traj_pointer_test,
                            img_pointer_test) = run_model(
                                    actor_critic=actor_critic,
                                    current_memory=current_memory_test,
                                    envs=test_envs,
                                    video=img_trajs_test,
                                    pointer=(traj_pointer_test,
                                        img_pointer_test)
                                    )

            # Save in rollouts (for loss computation)
            rollouts.insert(step, reward, masks)

            # Track all bunch of stuff and also save intermediate images and stuff
            tracked_values = track_values(tracked_values, policy_return)

            if j % log['log_interval'] == 0:
                # save_images(policy_return, old_observation, id_tmp_dir, j, step)
                if environment['name'] != "DeathValley-v0":
                    writer.add_image('imgs/gt', old_observation[0], j)

            # Keep track of rewards
            final_rewards, avg_nr_observed, num_ended_episodes = track_rewards(
                tracked_rewards, reward, masks, blank_mask)

        # Compute bootstrapped value
        with torch.no_grad():
            policy_return = actor_critic(
                current_memory=current_memory
                )

        next_value = policy_return.value_estimate

        # Compute targets (consisting of discounted rewards + bootstrapped value)
        rollouts.compute_returns(next_value, rl_setting['gamma'])

        # Compute losses:
        values = torch.stack(tuple(tracked_values['values']), dim=0)
        action_log_probs = torch.stack(tuple(tracked_values['action_log_probs']), dim=0)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()
        action_loss = -(Variable(advantages.detach()) * action_log_probs).mean()

        # Average over batch and time
        dist_entropy = torch.stack(tuple(tracked_values['dist_entropy'])).mean()

        total_loss = (value_loss * loss_function['value_loss_coef']
                + action_loss * loss_function['action_loss_coef']
                - dist_entropy * loss_function['entropy_coef'])

        optimizer.zero_grad()

        # Only reset the computation graph every 'multiplier_backprop_length' iterations
        retain_graph = j % algorithm['multiplier_backprop_length'] != 0
        total_loss.backward(retain_graph=retain_graph)

        if opt['max_grad_norm'] > 0:
            nn.utils.clip_grad_norm_(actor_critic.parameters(), opt['max_grad_norm'])

        optimizer.step()

        if opt['use_scheduler']:
            lr_scheduler.step()

        if not retain_graph:
            current_memory['states'] = utils.detach_state(current_memory['states'])

        rollouts.after_update()

        if log['save_model_interval'] > 0 and j % log['save_model_interval'] == 0:
            utils.save_model(id_tmp_dir,
                             'model_epoch_{}'.format(j),
                             actor_critic,
                             _run)

        # Logging = saving to database
        if j % log['log_interval'] == 0:
            end = time.time()
            utils.log_and_print(j, num_updates, end - start, id_tmp_dir, tracked_values, value_loss,
                                action_loss, dist_entropy, rl_setting, _run, writer)

    # Save final model
    utils.save_model(id_tmp_dir, 'model_final', actor_critic, _run)
