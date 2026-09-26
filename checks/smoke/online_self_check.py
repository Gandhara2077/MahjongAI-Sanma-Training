from pathlib import Path
from types import SimpleNamespace
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('MORTAL_CFG', str(ROOT / 'Mortal/config/sanma-online-128x8.toml'))
sys.path.insert(0, str(ROOT / 'Mortal/mortal'))


def main():
    import torch
    from player import OpponentEngine, TrainPlayer
    from player import compat_engine, config
    from libriichi3p_compat import CompatMjaiEngine
    config['compat'] = {'libriichi3p': str(ROOT / '.cache/libriichi3p/libriichi3p-3.12-x86_64-pc-windows-msvc.pyd'), 'max_workers': 1}
    config.setdefault('online_training', {})['selfplay_engine'] = 'native'
    delegate = SimpleNamespace(version=4, is_oracle=False, name='check')
    assert compat_engine(delegate, training=True) is delegate
    assert isinstance(compat_engine(delegate), CompatMjaiEngine)
    config['online_training']['selfplay_engine'] = 'invalid'
    try:
        compat_engine(delegate, training=True)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid backend accepted')
    del config['online_training']['selfplay_engine']
    assert isinstance(compat_engine(delegate, training=True), CompatMjaiEngine)

    def engine():
        return SimpleNamespace(brain=torch.nn.Linear(2, 2), dqn=torch.nn.Linear(2, 2))

    player = TrainPlayer.__new__(TrainPlayer)
    fixed = engine()
    dynamic = engine()
    player.opponent_engines = [
        OpponentEngine('fixed', 0.5, fixed),
        OpponentEngine('self', 0.5, SimpleNamespace(delegate=dynamic)),
    ]
    player.opponent_engines[1].source = 'current'
    before = {name: value.clone() for name, value in fixed.brain.state_dict().items()}
    assert hasattr(player, 'sync_opponents'), 'TrainPlayer must synchronize explicit current opponents'
    for version in ('session:1', 'session:2'):
        incoming = engine()
        player.sync_opponents(incoming.brain.state_dict(), incoming.dqn.state_dict(), version)
        for network in ('brain', 'dqn'):
            actual = getattr(dynamic, network)
            expected = getattr(incoming, network)
            assert all(torch.equal(value, expected.state_dict()[name]) for name, value in actual.state_dict().items())
            assert not actual.training
        assert player.opponent_engines[1].parameter_version == version
        assert all(torch.equal(value, before[name]) for name, value in fixed.brain.state_dict().items())
    try:
        player.sync_opponents(incoming.brain.state_dict(), {}, 'session:3')
    except RuntimeError:
        assert player.parameter_version is None
        assert player.opponent_engines[1].parameter_version is None
    else:
        raise AssertionError('incomplete pair must fail closed')
    assert player.opponent_engines[0].source == 'checkpoint'
    print('ONLINE_SELF_CHECK_OK')


if __name__ == '__main__':
    main()
