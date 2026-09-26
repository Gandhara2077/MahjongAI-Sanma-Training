'''A kan-selection tile must not consume a primary-discard replay label.'''

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'Mortal/mortal'))
from libriichi3p_compat import OBS_SHAPE, _ScriptedCaptureEngine


def main():
    primary = np.zeros(OBS_SHAPE, dtype=np.float32)
    selection = primary.copy()
    selection[635, :] = 1
    primary_mask = np.zeros(44, dtype=np.bool_)
    primary_mask[[9, 29, 39]] = True
    selection_mask = np.zeros(44, dtype=np.bool_)
    selection_mask[29] = True
    for script in ([(0, 29, False)], [(0, 29, True), (1, 29, False)]):
        engine = _ScriptedCaptureEngine()
        engine.begin_event(458, script)
        engine.react_batch([selection, primary], [selection_mask, primary_mask], None)
        engine.finish_event()
        for index, _, kan_select in script:
            feature, mask = engine.captures[index]
            np.testing.assert_array_equal(feature, selection if kan_select else primary)
            np.testing.assert_array_equal(mask, selection_mask if kan_select else primary_mask)
    print('COMPAT_CAPTURE_SELECTION_OK')


if __name__ == '__main__':
    main()
