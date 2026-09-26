"""Expert replay must continue across drains, not restart its workers/data."""
from pathlib import Path
import sys
import os
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Mortal/mortal'))


def main():
    from online_training import mixed_batches, repeating_batches
    starts = []

    def factory():
        starts.append(1)
        return iter([10, 20, 30])

    stream = repeating_batches(factory)
    observed = []
    for index in range(4):
        batch = list(mixed_batches([index], lambda: stream, 0.5))
        assert batch[0] == ('online', index)
        observed.append(batch[1][1])
    assert observed == [10, 20, 30, 10]
    assert len(starts) == 2, 'factory restarts only when the expert dataset is exhausted'
    try:
        next(repeating_batches(lambda: iter([])))
    except RuntimeError:
        pass
    else:
        raise AssertionError('empty expert dataset must fail, not spin forever')
    assert list(mixed_batches([1, 2], factory, 0)) == [('online', 1), ('online', 2)]
    os.environ['MORTAL_CFG'] = str(ROOT / 'Mortal/config/sanma-online-128x8-dynamic-self.toml')
    import common
    for interval in (5, 0.2):
        online = {'remote': {'host': '127.0.0.1', 'port': 1}}
        if interval != 5:
            online['drain_poll_seconds'] = interval
        with patch.object(common, 'config', {'online': online}), \
                patch.object(common.socket, 'socket', return_value=MagicMock()), \
                patch.object(common, 'send_msg'), \
                patch.object(common, 'recv_msg', side_effect=[{'count': 0}, {'count': 1, 'drain_dir': 'batch'}]), \
                patch.object(common.time, 'sleep') as sleep:
            assert common.drain() == 'batch'
            sleep.assert_called_once_with(interval)
    print('ONLINE_THROUGHPUT_CHECK_OK')


if __name__ == '__main__':
    main()
