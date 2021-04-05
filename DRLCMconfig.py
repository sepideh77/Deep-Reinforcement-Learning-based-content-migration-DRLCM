# -*- coding: utf-8 -*-
import argparse
import pickle
import numpy as np
from random import randint

parser = argparse.ArgumentParser(description='Configuration file')
arg_lists = []


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def str2bool(v):
    return v.lower() in ('true', '1')


# Environment
env_arg = add_argument_group('Environment')
env_arg.add_argument('--num_rsu', type=int, default=2, help='number of RSU')
env_arg.add_argument('--num_Nomadic_Caches', type=int, default=3, help='number of Nomadic Caches')
env_arg.add_argument('--num_Users_per_Nomadic_caches', type=int, default=5, help='number users per Nomadic Caches')
env_arg.add_argument('--num_Plcd_NS_contents', type=int, default=10,
                     help='number of already placed non-safety contents')
env_arg.add_argument('--num_s_contents', type=int, default=4, help='number of safety contents')
env_arg.add_argument('--num_descriptors', type=int, default=env_arg._actions[4].default + env_arg._actions[5].default,
                     help='number of unique contents')

# contents(S/NS) size
env_arg.add_argument('--size_of_contents', type=int, default=1, help='Size of contents')

# RSUs Diameters
env_arg.add_argument('--Diameter_rsus', type=int, default=2, help='Diameter of RSUs')

# Nomadic caches Capacity
env_arg.add_argument('--Capacity_nomadic_caches', type=int, default=7, help='Nomadic caches Capacity')

# RSU's Capacity
env_arg.add_argument('--Capacity_RSUs', type=int, default=10, help='RSUs Capacity')

# Velocity of Speed of nomadic caches
env_arg.add_argument('--Nomadic_Caches_Velocity', type=int, default=60, help='Velocity of nomadic caches')

# Number of targeted Nomadic Caches for safety contents
env_arg.add_argument('--num_Targeted_Nomadic_caches', type=int, default=4,
                     help='number of targeted nomadic caches interested in safety contents')

# Constraints parameter (needs modification)
env_arg.add_argument('--Maximum_numreq_handled_CacheNode', type=int, default=7,
                     help='Maximum number of requests handled by each cache nodes')
env_arg.add_argument('--transfering_delay', type=int, default=3,
                     help='Delay for transmitting one byte from RSU r to cache node i')
env_arg.add_argument('--Cloud_transfering_delay', type=int, default=50,
                     help='Delay for transmitting one byte from Cloud to cache node i')
env_arg.add_argument('--small_transfering_delay', type=int, default=1,
                     help='Delay for transmitting one byte from nomadic cache i to its zone')

# weights Parameters

env_arg.add_argument('--w1', type=int, default=7, help='Weight of partial cost 1 in ∁_migration')
env_arg.add_argument('--w2', type=int, default=7, help='Weight of partial cost 2 in ∁_migration')
env_arg.add_argument('--w3', type=int, default=7, help='Weight of partial cost 3 in ∁_migration')
env_arg.add_argument('--w4', type=int, default=7, help='Weight of ∁_(access_delay )in ∁_Total')
env_arg.add_argument('--w5', type=int, default=7, help='Weight of ∁_migrationin ∁_Total')
env_arg.add_argument('--w6', type=int, default=7, help='Weight of ∁_(s_dl )in ∁_Total')

# Power consumption cost units
env_arg.add_argument('--G', type=int, default=7,
                     help='Power consumption cost on nomadic cache i for uploading one byte')
env_arg.add_argument('--P', type=int, default=7,
                     help='Power consumption cost on nomadic cache i for Downloading one byte')
env_arg.add_argument('--O', type=int, default=7, help='Cost of transferring one byte one hop')

# Initial Placement
num_rsu = env_arg._actions[1].default
num_Nomadic_Caches = env_arg._actions[2].default
num_total_cache_nodes = num_rsu + num_Nomadic_Caches
num_NS_contents = env_arg._actions[4].default  # already exist nonsafety contents
num_S_contents = env_arg._actions[5].default
Capacity_nomadic_caches = env_arg._actions[9].default
length_of_plc_chain = num_NS_contents + num_S_contents + num_NS_contents * num_Nomadic_Caches * num_total_cache_nodes
env_arg.add_argument('--Length_plc_chain', type=int, default=length_of_plc_chain, help='length of service chain')

try:
    with open('objs.pkl', 'rb') as f:
        Initial_plc = pickle.load(f)
except FileNotFoundError:
    # create a valid Initial Placement
    Initial_plc = np.empty(length_of_plc_chain)
    Initial_plc[:] = np.nan
    sub_chain_part1 = []  # Specifying placement of non-safety contents on nomadic caches
    sub_chain_part2 = []  # Specifying placement of safety contents on rsu s.
    sub_chain_part3 = []  # specifying number of req for non_safety content from each region of nomadic caches to be replyed by each cache node

    for i in range(length_of_plc_chain):
        if i < num_NS_contents:
            Initial_plc[i] = randint(0, num_total_cache_nodes - 1)
            sub_chain_part1.append(Initial_plc[i])
        if num_NS_contents <= i < num_NS_contents + num_S_contents:
            Initial_plc[i] = randint(num_total_cache_nodes - 1, num_rsu + num_Nomadic_Caches - 1)
            sub_chain_part2.append(Initial_plc[i])
        if num_NS_contents + num_S_contents <= i:
            Initial_plc[i] = randint(0, env_arg._actions[13].default)
            sub_chain_part3.append(Initial_plc[i])
    with open('InitialPlacement.pkl', 'wb') as f:
        pickle.dump(Initial_plc, f)
if (len(Initial_plc) != env_arg._actions[6].default) or (max(Initial_plc) > (num_rsu + num_Nomadic_Caches)):
    print("Warning!!!A new initial Placement should be created !!! ")

env_arg.add_argument('--Initial_Placement', type=int, default=Initial_plc,
                     help='Initial Placement of non safety contents')

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--hidden_dim', type=int, default=64, help='agent LSTM num_neurons')
net_arg.add_argument('--num_stacks', type=int, default=3, help='agent LSTM num_stacks')

# Data


data_arg = add_argument_group('Data')
data_arg.add_argument('--batch_size', type=int, default=11, help='batch size')
data_arg.add_argument('--min_length', type=int, default=env_arg._actions[6].default,
                      help='minimum service chain length')
data_arg.add_argument('--max_length', type=int, default=env_arg._actions[6].default,
                      help='maximum service chain length')

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--num_epoch', type=int, default=300, help='number of epochs')
train_arg.add_argument('--learning_rate', type=float, default=0.001, help='agent learning rate')


# Performance
perf_arg = add_argument_group('Performance')
perf_arg.add_argument('--enable_performance', type=str2bool, default=True,
                      help='compare performance agains Gecode solver')

# Misc
misc_arg = add_argument_group('User options')

misc_arg.add_argument('--train_mode', type=str2bool, default=True, help='switch between training and testing')
misc_arg.add_argument('--save_model', type=str2bool, default=True, help='whether or not model is loaded')
misc_arg.add_argument('--load_model', type=str2bool, default=False, help='whether or not model is retrieved')

misc_arg.add_argument('--save_to', type=str, default='save/model', help='saver sub directory')
misc_arg.add_argument('--load_from', type=str, default='save/model', help='loader sub directory')
misc_arg.add_argument('--log_dir', type=str, default='summary/repo', help='summary writer log directory')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


if __name__ == "__main__":
    config, _ = get_config()
