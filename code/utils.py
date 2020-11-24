import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import errno
import os
import json
import sys
from glob import glob
import os.path as osp
import logging
from docopt import docopt
from sacred.arg_parser import get_config_updates


def safe_make_dirs(path):
    """
    Given a path, makes a directory. Doesn't make directory if it already exists. Treats possible
    race conditions safely.
    http://stackoverflow.com/questions/273192/how-to-check-if-a-directory-exists-and-create-it-if-necessary
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def toOneHot(action_space, actions):
    """
    If action_space is "Discrete", return a one hot vector, otherwise just return the same `actions` vector.

    actions: [batch_size, 1] or [batch_size, n, 1]

    If action space is continuous, just return the same action vector.
    """
    # One hot encoding buffer that you create out of the loop and just keep reusing
    if action_space.__class__.__name__ == "Discrete":
        nr_actions = action_space.n
        actions_onehot_dim = list(actions.size())
        actions_onehot_dim[-1] = nr_actions

        actions = actions.view(-1, 1).long()
        action_onehot = torch.FloatTensor(actions.size(0), nr_actions)

        return_variable = False
        if isinstance(actions, Variable):
            actions = actions.data
            return_variable = True

        # In your for loop
        action_onehot.zero_()
        if actions.is_cuda:
            action_onehot = action_onehot.cuda()
        action_onehot.scatter_(1, actions, 1)

        if return_variable:
            action_onehot = Variable(action_onehot)

        action_onehot.view(*actions_onehot_dim)

        return action_onehot
    else:
        return actions.detach()


def save_model(dir, name, model, _run):
    """
    Save the model to the observer using the `name`.
    _run is the _run object from sacred.
    """

    name_model = dir + '/' + name
    torch.save(model.state_dict(), name_model)

    s_current = os.path.getsize(name_model) / (1024 * 1024)

    _run.add_artifact(name_model)
    os.remove(name_model)

    logging.info('Saving model {}: Size: {} MB'.format(name, s_current))


def save_numpy(dir, name, array, _run):
    """
    Save a numpy array to the observer, using the `name`.
    _run is the _run object from sacred.
    """

    name = dir + '/' + name
    np.save(name, array.astype(np.float32))
    s_current = os.path.getsize(name) / (1024 * 1024)
    _run.add_artifact(name)
    os.remove(name)
    logging.info('Saving observations {}: Size: {} MB'.format(name, 2 * s_current))


def load_results(dir):
    """
    Since we are using clipped rewards (e.g. in Atari games), we need to access the monitor
    log files to get the true returns.

    Args:
        dir: Directory of the monitor files

    Returns:
        df: A pandas dataframe. Forgot the dimensions but it works with the function `log_and_print`
    """
    import pandas
    monitor_files = (glob(osp.join(dir, "*monitor.csv")))
    if not monitor_files:
        raise Exception("no monitor files of the found")
    dfs = []
    headers = []
    for run_nr, fname in enumerate(monitor_files):
        with open(fname, 'rt') as fh:
            firstline = fh.readline()
            assert firstline[0] == '#'
            header = json.loads(firstline[1:])
            df = pandas.read_csv(fh, index_col=None)
            headers.append(header)
            df['t'] += header['t_start']
            df['run_nr'] = run_nr
        dfs.append(df)
    df = pandas.concat(dfs)
    df.sort_values('t', inplace=True)
    df.reset_index(inplace=True)
    df['t'] -= min(header['t_start'] for header in headers)
    df.headers = headers  # HACK to preserve backwards compatibility
    return df

def load_results_numpy(dir, mode):
    monitor_files = (glob(osp.join(dir, mode + "*monitor.csv")))
    if not monitor_files:
        raise Exception("no monitor files of the found")

    last_results = []

    for fname in monitor_files:
        data = np.genfromtxt(fname, dtype=float, delimiter=',', skip_header=2)
        last = data[-1, 0]
        last_results.append(last)

    return np.mean(last_results)

def log_and_print(j, num_updates, T, id_tmp_dir, tracking,
                  value_loss, action_loss, dist_entropy,
                  rl_setting, _run, writer):

    total_num_steps = (j + 1) * rl_setting['num_processes'] * rl_setting['num_steps']
    fps = int(total_num_steps / T)
    try:
        last_true_result = load_results_numpy(id_tmp_dir, 'train')
    except:
        last_true_result = -500

    try:
        last_true_result_test = load_results_numpy(id_tmp_dir, 'test')
    except:
        last_true_result_test = -500

    num_frames = j * rl_setting['num_steps'] * rl_setting['num_processes']

    writer.add_scalar('reward/true_reward', last_true_result, num_frames)
    writer.add_scalar('reward/true_reward_test', last_true_result_test,
            num_frames)
    writer.add_scalar('loss/value_loss', value_loss.item(), num_frames)
    writer.add_scalar('loss/action_loss', action_loss.item(), num_frames)
    writer.add_scalar('loss/entropy', dist_entropy.item(), num_frames)
    # Log scalars
    _run.log_scalar("result.true", last_true_result, num_frames)
    _run.log_scalar("result.true_test", last_true_result_test, num_frames)

    _run.log_scalar("particles.killed",
                    np.mean(tracking['num_killed_particles']),
                    num_frames)
    _run.log_scalar("obs.fps", fps, num_frames)
    _run.log_scalar("loss.value", value_loss, num_frames)
    _run.log_scalar("loss.action", action_loss, num_frames)
    _run.log_scalar("loss.entropy", dist_entropy, num_frames)

    logging.info('Updt: {:5} |FPS {:5}||R_TRAIN {:5}|R_VALID {:5}|ENTRO {:5}|VAL {:5}|ACT {:5}'.format(
        str(j / num_updates)[:5],
        str(fps),
        str(last_true_result)[:5],
        str(last_true_result_test)[:5],
        str(dist_entropy.item())[:5],
        str(value_loss.item())[:5],
        str(action_loss.item())[:5],))

def get_environment_yaml(ex):
    """
    Get the name of the environment_yaml file that should be specified in the command line as:
    'python main.py -p with environment.config_file=<env_config_file>.yaml [...]'
    """
    _, _, usage = ex.get_usage()
    args = docopt(usage, [str(a) for a in sys.argv[1:]], help=False)
    config_updates, _ = get_config_updates(args['UPDATE'])
    # updates = arg_parser.get_config_updates(args['UPDATE'])[0]
    environment_yaml = config_updates.get('environment', {}).get('config_file', None)
    return environment_yaml

def linear_decay(epoch, total_num_updates):
    r"""Returns a multiplicative factor for linear value decay

    Args:
        epoch: current epoch number
        total_num_updates: total number of epochs

    Returns:
        multiplicative factor that decreases param value linearly
    """
    return 1 - (epoch / float(total_num_updates))


def detach_state(state):
    if type(state) == tuple:
        return tuple([detach_state(s) for s in state])
    else:
        return state.detach()


def cudify_state(state, device):
    if type(state) == tuple:
        return tuple(cudify_state(s, device) for s in state)
    else:
        return state.to(device)


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


class FnModule(nn.Module):
    def __init__(self, fn):
        super(FnModule, self).__init__()
        self.fn = fn

    def forward(self, *input):
        return self.fn(*input)


def conv(batchNorm, in_channels, out_channels, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels)
        )
    else:
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.ReLU(inplace=True)
        )


def conv1x1(batchNorm, in_channels, out_channels):
    if batchNorm:
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels)
        )
    else:
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )


def count_parameters(model):
    total = 0
    for var in model.parameters():
        n = 1
        for x in var.size():
            n *= x
        total += n
    return total


def ortho_init(tensor, scale=1.0):
    # if isinstance(tensor, Variable):
    #     ortho_init(tensor.data, scale=scale)
    #     return tensor
    shape = tensor.size()
    if len(shape) == 2:
        flat_shape = shape
    elif len(shape) == 4:
        flat_shape = (shape[0] * shape[2] * shape[3], shape[1])  # NCHW
    else:
        raise NotImplementedError
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v  # pick the one with the correct shape
    q = q.reshape(shape)
    w = (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    tensor = torch.tensor(w)
    return tensor


def nn_init(module, w_init=ortho_init, w_scale=1.0, b_init=nn.init.constant, b_scale=0.0):
    w_init(module.weight, w_scale)
    b_init(module.bias, b_scale)
    return module


USE_CUDA = torch.cuda.is_available()


def cudify(x):
    if USE_CUDA:
        return x.cuda()
    else:
        return x
