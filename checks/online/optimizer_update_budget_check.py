from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Mortal/mortal'))


def main():
    import torch
    import online_training
    assert hasattr(online_training, 'step_optimizer'), 'successful optimizer update accounting is missing'
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    scaler = torch.amp.GradScaler('cpu', init_scale=2.0)
    successful = 0
    for overflow in (False, True, False):
        before = parameter.detach().clone()
        loss = parameter.square().sum()
        if overflow:
            loss = loss * float('inf')
        scaler.scale(loss).backward()
        updated = online_training.step_optimizer(optimizer, scaler, count_successful=True)
        optimizer.zero_grad(set_to_none=True)
        assert updated is not overflow
        if overflow:
            assert torch.equal(parameter, before)
        successful += updated
    assert successful == 2
    assert int(optimizer.state[parameter]['step']) == 2
    print('OPTIMIZER_UPDATE_BUDGET_OK')


if __name__ == '__main__':
    main()
