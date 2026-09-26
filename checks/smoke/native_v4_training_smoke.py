"""Smoke-test the direct native sanma v4 training path."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MORTAL = ROOT / "Mortal"
sys.path.insert(0, str(MORTAL / "mortal"))


def main() -> None:
    from dataloader import FileDatasetsIter
    from libriichi.dataset import GameplayLoader
    from model import Brain, DQN

    path = sorted((ROOT / "koromo" / "mjson" / "2026" / "01" / "04").glob("*.mjson"))[0]
    loader = GameplayLoader(
        version=4,
        oracle=False,
        always_include_kan_select=True,
        augmented=False,
        encode_observations=True,
    )
    games = loader.load_gz_log_files([str(path)])[0]
    entries = []
    for game in games:
        obs = game.take_obs()
        masks = game.take_masks()
        actions = game.take_actions()
        assert len(obs) == len(masks) == len(actions)
        for feature, mask, action in zip(obs, masks, actions):
            feature = np.asarray(feature, dtype=np.float32)
            mask = np.asarray(mask, dtype=np.bool_)
            assert feature.shape == (775, 34)
            assert mask.shape == (44,)
            assert mask[int(action)]
            assert np.isfinite(feature).all()
            entries.append((feature, mask, int(action)))

    batch = min(8, len(entries))
    observations = torch.from_numpy(np.stack([x[0] for x in entries[:batch]]))
    masks = torch.from_numpy(np.stack([x[1] for x in entries[:batch]])).bool()
    actions = torch.tensor([x[2] for x in entries[:batch]], dtype=torch.long)
    brain = Brain(version=4, conv_channels=32, num_blocks=2)
    dqn = DQN(version=4)
    optimizer = torch.optim.AdamW(
        list(brain.parameters()) + list(dqn.parameters()), lr=1e-5
    )
    q_values = dqn(brain(observations), masks)
    loss = -q_values.gather(1, actions[:, None]).mean()
    assert torch.isfinite(q_values[masks]).all()
    assert torch.isneginf(q_values[~masks]).all()
    assert torch.isfinite(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    output = MORTAL / "training" / "native_v4_training_smoke_tmp.pth"
    torch.save({"model": brain.state_dict(), "dqn": dqn.state_dict()}, output)
    assert output.is_file()
    output.unlink()

    dataset = FileDatasetsIter(
        version=4,
        file_list=[str(path)],
        pts=[6.0, 3.0, 0.0],
        file_batch_size=1,
        num_epochs=1,
        observation_encoder="native_v4",
        reference_path="Z:/missing/libriichi3p.pyd",
    )
    assert not dataset.use_compat_replay
    assert dataset.native_version is None
    assert "libriichi3p_compat" not in sys.modules
    print(f"NATIVE_V4_TRAINING_SMOKE_OK samples={len(entries)} batch={batch}")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, ImportError, OSError, RuntimeError, KeyError) as error:
        print(f"NATIVE_V4_TRAINING_SMOKE_FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
