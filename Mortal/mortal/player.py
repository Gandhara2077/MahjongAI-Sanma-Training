import torch
import numpy as np
import os
import shutil
import secrets
import logging
import random
import hashlib
import json
from os import path
from dataclasses import dataclass
from model import Brain, DQN
from engine import MortalEngine
from online_training import choose_weighted, weighted_quota_index
from libriichi.consts import NUM_PLAYERS
from libriichi.stat import Stat
from config import config

if NUM_PLAYERS == 4:
    from libriichi.arena import OneVsThree as ARENA_CLASS
else:
    from libriichi.arena import OneVsTwo as ARENA_CLASS


@dataclass
class OpponentEngine:
    name: str
    weight: float
    engine: object
    source: str = 'checkpoint'
    parameter_version: str | None = None


def parameter_digest(brain, dqn):
    digest = hashlib.sha256()
    for prefix, state in (('brain', brain), ('dqn', dqn)):
        for name, value in sorted(state.items()):
            value = value.detach().cpu().contiguous()
            digest.update(f'{prefix}:{name}:{value.dtype}:{tuple(value.shape)}'.encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def compat_engine(engine, *, training=False):
    """Wrap an engine in the libriichi3p compatibility bridge.

    This is a no-op unless a [compat] section is present in the active
    config, so a yonma environment never touches the historical libriichi3p
    reference binary.
    """
    if training:
        backend = config.get('online_training', {}).get('selfplay_engine', 'compat')
        if backend not in ('native', 'compat'):
            raise ValueError(f'unknown selfplay engine: {backend}')
        if backend == 'native':
            return engine
    compat = config.get('compat')
    if not compat:
        return engine
    from libriichi3p_compat import CompatMjaiEngine  # lazy: sanma only
    return CompatMjaiEngine(
        engine,
        reference_path=compat.get('libriichi3p') or None,
        max_workers=compat.get('max_workers', 64),
    )


def _load_stable_models(opponent_cfg):
    state = torch.load(
        opponent_cfg['state_file'],
        weights_only=True,
        map_location=torch.device('cpu'),
    )
    cfg = state['config']
    version = cfg['control'].get('version', 1)
    conv_channels = cfg['resnet']['conv_channels']
    num_blocks = cfg['resnet']['num_blocks']
    stable_mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
    stable_dqn = DQN(version=version).eval()
    stable_mortal.load_state_dict(state['mortal'])
    stable_dqn.load_state_dict(state['current_dqn'])
    if opponent_cfg.get('enable_compile', False):
        stable_mortal.compile()
        stable_dqn.compile()
    return version, stable_mortal, stable_dqn


class TestPlayer:
    def __init__(self):
        baseline_cfg = config['baseline']['test']
        device = torch.device(baseline_cfg['device'])

        version, stable_mortal, stable_dqn = _load_stable_models({
            'state_file': baseline_cfg['state_file'],
            'enable_compile': baseline_cfg['enable_compile'],
        })

        self.baseline_engine = compat_engine(MortalEngine(
            stable_mortal,
            stable_dqn,
            is_oracle = False,
            version = version,
            device = device,
            enable_amp = True,
            enable_rule_based_agari_guard = True,
            name = 'baseline',
        ))
        self.chal_version = config['control']['version']
        self.log_dir = path.abspath(config['test_play']['log_dir'])

    def test_play(self, seed_count, mortal, dqn, device):
        torch.backends.cudnn.benchmark = False
        engine_chal = compat_engine(MortalEngine(
            mortal,
            dqn,
            is_oracle = False,
            version = self.chal_version,
            device = device,
            enable_amp = True,
            name = 'mortal',
        ))

        if path.isdir(self.log_dir):
            shutil.rmtree(self.log_dir)

        env = ARENA_CLASS(
            disable_progress_bar = False,
            log_dir = self.log_dir,
        )
        env.py_vs_py(
            challenger = engine_chal,
            champion = self.baseline_engine,
            seed_start = (10000, 0x2000),
            seed_count = seed_count,
        )

        stat = Stat.from_dir(self.log_dir, 'mortal')
        torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
        return stat

class TrainPlayer:
    def __init__(self):
        opponents_cfg = config.get('online_training', {}).get('opponents')
        if not opponents_cfg:
            opponents_cfg = [{
                'name': 'baseline',
                'weight': 1.0,
                **config['baseline']['train'],
            }]

        self.opponent_engines = []
        for opponent_cfg in opponents_cfg:
            source = opponent_cfg.get('source', 'checkpoint')
            if source not in ('checkpoint', 'current'):
                raise ValueError(f'unknown opponent source: {source}')
            device = torch.device(opponent_cfg['device'])
            version, stable_mortal, stable_dqn = _load_stable_models(opponent_cfg)

            name = opponent_cfg['name']
            engine = compat_engine(MortalEngine(
                stable_mortal,
                stable_dqn,
                is_oracle = False,
                version = version,
                device = device,
                enable_amp = opponent_cfg.get('enable_amp', True),
                enable_rule_based_agari_guard = opponent_cfg.get(
                    'enable_rule_based_agari_guard',
                    True,
                ),
                name = name,
            ), training=True)
            self.opponent_engines.append(OpponentEngine(
                name=name,
                weight=float(opponent_cfg['weight']),
                engine=engine,
                source=source,
                parameter_version=(parameter_digest(stable_mortal.state_dict(), stable_dqn.state_dict())
                                   if source == 'checkpoint' else None),
            ))
        self.parameter_version = None
        self.opponent_counts = [0] * len(self.opponent_engines)
        self.opponent_rng = random.Random(secrets.randbits(64))
        # 'quota' cycles opponents to match their configured weights exactly;
        # 'weighted' samples an opponent per session with weight probability.
        self.opponent_selection = config.get('online_training', {}).get(
            'opponent_selection',
            'quota',
        )

        profile = os.environ.get('TRAIN_PLAY_PROFILE', 'default')
        logging.info(f'using profile {profile}')
        cfg = config['train_play'][profile]
        self.chal_version = config['control']['version']
        self.log_dir = path.abspath(cfg['log_dir'])
        self.train_key = secrets.randbits(64)
        self.train_seed = 10000

        if cfg['games'] % NUM_PLAYERS != 0:
            raise ValueError(f'train_play games must be divisible by {NUM_PLAYERS}')
        self.seed_count = cfg['games'] // NUM_PLAYERS
        self.boltzmann_epsilon = cfg['boltzmann_epsilon']
        self.boltzmann_temp = cfg['boltzmann_temp']
        self.top_p = cfg['top_p']

        self.repeats = cfg['repeats']
        self.repeat_counter = 0

    def sync_opponents(self, mortal_state, dqn_state, parameter_version):
        if not isinstance(parameter_version, str) or not parameter_version:
            raise ValueError('parameter version must identify the client session and update')
        self.parameter_version = None
        dynamic = [opponent for opponent in self.opponent_engines if opponent.source == 'current']
        for opponent in dynamic:
            opponent.parameter_version = None
        for opponent in dynamic:
            delegate = getattr(opponent.engine, 'delegate', opponent.engine)
            delegate.brain.load_state_dict(mortal_state)
            delegate.dqn.load_state_dict(dqn_state)
            delegate.brain.eval()
            delegate.dqn.eval()
            if config.get('online_training', {}).get('verify_self_sync', False):
                expected = parameter_digest(mortal_state, dqn_state)
                actual = parameter_digest(delegate.brain.state_dict(), delegate.dqn.state_dict())
                if actual != expected:
                    raise RuntimeError('self engine parameters differ from incoming update')
                logging.info(f'self sync verified: {parameter_version} sha256={actual}')
        for opponent in dynamic:
            opponent.parameter_version = parameter_version
        self.parameter_version = parameter_version

    def _select_opponent(self):
        weights = [opponent.weight for opponent in self.opponent_engines]
        if self.opponent_selection == 'quota':
            opponent_index = weighted_quota_index(weights, self.opponent_counts)
            self.opponent_counts[opponent_index] += 1
            return self.opponent_engines[opponent_index]
        if self.opponent_selection == 'weighted':
            return choose_weighted(
                self.opponent_engines,
                weights,
                self.opponent_rng.random(),
            )
        raise ValueError(f'unknown opponent selection strategy: {self.opponent_selection}')

    def train_play(self, mortal, dqn, device):
        if any(opponent.source == 'current' and opponent.parameter_version is None
               for opponent in self.opponent_engines):
            raise RuntimeError('current opponents must be synchronized before a batch')
        torch.backends.cudnn.benchmark = False
        engine_chal = compat_engine(MortalEngine(
            mortal,
            dqn,
            is_oracle = False,
            version = self.chal_version,
            boltzmann_epsilon = self.boltzmann_epsilon,
            boltzmann_temp = self.boltzmann_temp,
            top_p = self.top_p,
            device = device,
            enable_amp = True,
            name = 'trainee',
        ), training=True)

        if path.isdir(self.log_dir):
            shutil.rmtree(self.log_dir)

        env = ARENA_CLASS(
            disable_progress_bar = False,
            log_dir = self.log_dir,
        )
        opponent = self._select_opponent()
        logging.info(f'using opponent: {opponent.name}')
        logging.info('opponent batch: ' + json.dumps({
            'name': opponent.name, 'source': opponent.source,
            'opponent_version': opponent.parameter_version,
            'trainee_version': self.parameter_version,
            'games': self.seed_count * NUM_PLAYERS,
            'seed_key': self.train_key, 'seed_start': self.train_seed,
            'seed_count': self.seed_count,
        }))
        rankings = env.py_vs_py(
            challenger = engine_chal,
            champion = opponent.engine,
            seed_start = (self.train_seed, self.train_key),
            seed_count = self.seed_count,
        )
        self.repeat_counter += 1
        if self.repeat_counter == self.repeats:
            self.train_seed += self.seed_count
            self.repeat_counter = 0

        rankings = np.array(rankings)
        file_list = list(map(lambda p: path.join(self.log_dir, p), os.listdir(self.log_dir)))

        torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
        return rankings, file_list
