import prelude

import argparse

import torch

from checkpoint import export_deployment
from model import Brain, DQN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='training checkpoint')
    parser.add_argument('output', help='MahjongCopilot-compatible .pth file')
    args = parser.parse_args()

    state = torch.load(args.input, weights_only=True, map_location='cpu')
    config = state['config']
    version = config['control']['version']
    mortal = Brain(version=version, **config['resnet']).eval()
    dqn = DQN(version=version).eval()
    mortal.load_state_dict(state['mortal'])
    dqn.load_state_dict(state['current_dqn'])
    export_deployment(args.output, mortal, dqn, config)


if __name__ == '__main__':
    main()
