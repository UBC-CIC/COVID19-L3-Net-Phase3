import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def byte2mb(x):
    return x * 1e-6


def model_size(net):
    s = 0
    for p in net.parameters():
        s += p.numel() * 4
    return byte2mb(s)


def num_params(net):
    n = 0
    for p in net.parameters():
        n += p.numel()
    return n


def tensor_size(x):
    return byte2mb(x.numel() * 4)
