import prelude

import json
import random
import torch
import logging
from os import path
from glob import glob
from datetime import datetime
from pathlib import Path
from torch import optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter
from model import GRP
from libriichi.dataset import Grp
from common import tqdm
from config import config


def should_stop_training(step, max_steps):
    return max_steps > 0 and step >= max_steps


def update_early_stopping(
    best_loss,
    current_loss,
    stale_evaluations,
    min_delta,
):
    if current_loss < best_loss - min_delta:
        return current_loss, 0, True
    return best_loss, stale_evaluations + 1, False


def should_early_stop(stale_evaluations, patience):
    return patience > 0 and stale_evaluations >= patience


class GrpFileDatasetsIter(IterableDataset):
    def __init__(
        self,
        file_list,
        file_batch_size = 50,
        cycle = False,
        shuffle = True,
    ):
        super().__init__()
        self.file_list = list(file_list)
        self.file_batch_size = file_batch_size
        self.cycle = cycle
        self.shuffle = shuffle
        self.buffer = []
        self.iterator = None

    def build_iter(self):
        while True:
            if self.shuffle:
                random.shuffle(self.file_list)
            for start_idx in range(0, len(self.file_list), self.file_batch_size):
                self.populate_buffer(start_idx)
                buffer_size = len(self.buffer)
                indices = (
                    random.sample(range(buffer_size), buffer_size)
                    if self.shuffle
                    else range(buffer_size)
                )
                for i in indices:
                    yield self.buffer[i]
                self.buffer.clear()
            if not self.cycle:
                break

    def populate_buffer(self, start_idx):
        file_list = self.file_list[start_idx:start_idx + self.file_batch_size]
        data = Grp.load_gz_log_files(file_list)

        for game in data:
            feature = game.take_feature()
            rank_by_player = game.take_rank_by_player()

            for i in range(feature.shape[0]):
                inputs_seq = torch.as_tensor(feature[:i + 1], dtype=torch.float64)
                self.buffer.append((
                    inputs_seq,
                    rank_by_player,
                ))

    def __iter__(self):
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator

def collate(batch):
    inputs = []
    lengths = []
    rank_by_players = []
    for inputs_seq, rank_by_player in batch:
        inputs.append(inputs_seq)
        lengths.append(len(inputs_seq))
        rank_by_players.append(rank_by_player)

    lengths = torch.tensor(lengths)
    rank_by_players = torch.tensor(rank_by_players, dtype=torch.int64, pin_memory=True)

    padded = pad_sequence(inputs, batch_first=True)
    packed_inputs = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
    packed_inputs.pin_memory()

    return packed_inputs, rank_by_players


def save_checkpoint(state, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_file = destination.with_suffix(destination.suffix + '.tmp')
    torch.save(state, temp_file)
    temp_file.replace(destination)


def write_metrics(payload, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_file = destination.with_suffix(destination.suffix + '.tmp')
    with temp_file.open('w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2)
        file.write('\n')
    temp_file.replace(destination)


def evaluate_validation(
    grp,
    file_list,
    *,
    file_batch_size,
    batch_size,
    device,
    pts,
):
    dataset = GrpFileDatasetsIter(
        file_list=file_list,
        file_batch_size=file_batch_size,
        cycle=False,
        shuffle=False,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=False,
        num_workers=0,
        collate_fn=collate,
    )
    totals = {
        'samples': 0,
        'nll': 0.0,
        'correct': 0.0,
        'brier': 0.0,
        'expected_pt_mse': 0.0,
    }
    grp.eval()
    with torch.inference_mode():
        for inputs, rank_by_players in loader:
            inputs = inputs.to(dtype=torch.float64, device=device)
            rank_by_players = rank_by_players.to(dtype=torch.int64, device=device)
            logits = grp.forward_packed(inputs)
            labels = grp.get_label(rank_by_players)
            probabilities = logits.softmax(-1)
            one_hot = F.one_hot(
                labels,
                num_classes=probabilities.shape[-1],
            ).to(torch.float64)
            predicted_pts = grp.calc_matrix(logits) @ pts
            actual_pts = F.one_hot(
                rank_by_players,
                num_classes=len(pts),
            ).to(torch.float64) @ pts
            count = labels.numel()
            totals['samples'] += count
            totals['nll'] += F.cross_entropy(
                logits,
                labels,
                reduction='sum',
            ).item()
            totals['correct'] += (
                logits.argmax(-1) == labels
            ).to(torch.float64).sum().item()
            totals['brier'] += (
                (probabilities - one_hot) ** 2
            ).sum(-1).sum().item()
            totals['expected_pt_mse'] += (
                (predicted_pts - actual_pts) ** 2
            ).mean(-1).sum().item()
    grp.train()
    if totals['samples'] == 0:
        raise RuntimeError('GRP validation set produced no samples')
    samples = totals['samples']
    return {
        'samples': samples,
        'nll': totals['nll'] / samples,
        'accuracy': totals['correct'] / samples,
        'brier': totals['brier'] / samples,
        'expected_pt_mse': totals['expected_pt_mse'] / samples,
    }

def train():
    cfg = config['grp']
    state_file = cfg['state_file']
    batch_size = cfg['control']['batch_size']
    save_every = cfg['control']['save_every']
    max_steps = cfg['control'].get('max_steps', 0)
    best_state_file = Path(
        cfg['control'].get(
            'best_state_file',
            str(Path(state_file).with_name('best_grp.pth')),
        )
    )
    metrics_file = Path(
        cfg['control'].get(
            'metrics_file',
            str(Path(state_file).with_name('grp_training_metrics.json')),
        )
    )
    early_stopping_patience = int(
        cfg['control'].get('early_stopping_patience', 0)
    )
    early_stopping_min_delta = float(
        cfg['control'].get('early_stopping_min_delta', 0.0)
    )
    selection_metric = str(
        cfg['control'].get('selection_metric', 'nll')
    )
    if selection_metric not in {'nll', 'brier', 'expected_pt_mse'}:
        raise ValueError(f'unsupported GRP selection metric: {selection_metric}')

    device = torch.device(cfg['control']['device'])
    torch.backends.cudnn.benchmark = cfg['control']['enable_cudnn_benchmark']
    if device.type == 'cuda':
        logging.info(f'device: {device} ({torch.cuda.get_device_name(device)})')
    else:
        logging.info(f'device: {device}')

    grp = GRP(**cfg['network']).to(device)
    optimizer = optim.AdamW(grp.parameters())

    if path.exists(state_file):
        state = torch.load(state_file, weights_only=True, map_location=device)
        timestamp = datetime.fromtimestamp(
            state.get('timestamp', 0.0)
        ).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f'loaded: {timestamp}')
        grp.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        steps = state['steps']
    else:
        steps = 0

    def sync_device_before_exit():
        # ROCm on Windows (torch 2.9.1+rocm7.2.1, RX 9070 XT): the process can
        # hang forever at interpreter exit when the device was touched but its
        # stream was never synchronized (e.g. the early returns below).
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    progress = {}
    if metrics_file.exists():
        with metrics_file.open(encoding='utf-8') as file:
            progress = json.load(file)
    if progress.get('early_stopped') and best_state_file.exists():
        logging.info(
            f"early stopping already reached at step {progress.get('final_step', steps):,}"
        )
        sync_device_before_exit()
        return
    if should_stop_training(steps, max_steps):
        logging.info(f'max steps already reached: {steps:,}')
        sync_device_before_exit()
        return

    lr = cfg['optim']['lr']
    optimizer.param_groups[0]['lr'] = lr

    file_index = cfg['dataset']['file_index']
    train_globs = cfg['dataset']['train_globs']
    val_globs = cfg['dataset']['val_globs']
    if path.exists(file_index):
        index = torch.load(file_index, weights_only=True)
        train_file_list = index['train_file_list']
        val_file_list = index['val_file_list']
    else:
        logging.info('building file index...')
        train_file_list = []
        val_file_list = []
        for pat in train_globs:
            train_file_list.extend(glob(pat, recursive=True))
        for pat in val_globs:
            val_file_list.extend(glob(pat, recursive=True))
        train_file_list.sort(reverse=True)
        val_file_list.sort(reverse=True)
        Path(file_index).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'train_file_list': train_file_list, 'val_file_list': val_file_list}, file_index)
    writer = SummaryWriter(cfg['control']['tensorboard_dir'])

    train_file_data = GrpFileDatasetsIter(
        file_list = train_file_list,
        file_batch_size = cfg['dataset']['file_batch_size'],
        cycle = True,
    )
    train_data_loader = iter(DataLoader(
        dataset = train_file_data,
        batch_size = batch_size,
        drop_last = True,
        num_workers = 1,
        collate_fn = collate,
    ))

    logging.info(f'train file list size: {len(train_file_list):,}')
    logging.info(f'val file list size: {len(val_file_list):,}')

    approx_passes = steps * batch_size / (len(train_file_list) * 10)
    logging.info(f'total steps: {steps:,} est. passes: {approx_passes:8.3f}')

    def build_state():
        return {
            'model': grp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'steps': steps,
            'timestamp': datetime.now().timestamp(),
        }

    pts = torch.tensor(
        config.get('env', {}).get('pts', [6.0, 3.0, 0.0]),
        dtype=torch.float64,
        device=device,
    )
    best_metric_value = float(
        progress.get(
            'best_metric_value',
            progress.get('best_val_loss', float('inf')),
        )
    )
    stale_evaluations = int(progress.get('stale_evaluations', 0))
    history = list(progress.get('history', []))
    if not best_state_file.exists():
        best_metric_value = float('inf')
        stale_evaluations = 0

    train_loss = 0.0
    train_correct = 0.0
    train_batches = 0

    def checkpoint_and_validate():
        nonlocal best_metric_value
        nonlocal stale_evaluations
        nonlocal train_loss
        nonlocal train_correct
        nonlocal train_batches

        state = build_state()
        save_checkpoint(state, state_file)
        validation = evaluate_validation(
            grp,
            val_file_list,
            file_batch_size=cfg['dataset']['file_batch_size'],
            batch_size=batch_size,
            device=device,
            pts=pts,
        )
        best_metric_value, stale_evaluations, improved = update_early_stopping(
            best_metric_value,
            validation[selection_metric],
            stale_evaluations,
            early_stopping_min_delta,
        )
        if improved or not best_state_file.exists():
            save_checkpoint(state, best_state_file)

        train_metrics = {
            'loss': train_loss / max(1, train_batches),
            'accuracy': train_correct / max(1, train_batches),
            'batches': train_batches,
        }
        entry = {
            'step': steps,
            'train': train_metrics,
            'validation': validation,
            'improved': improved,
            'stale_evaluations': stale_evaluations,
        }
        history.append(entry)
        early_stopped = should_early_stop(
            stale_evaluations,
            early_stopping_patience,
        )
        payload = {
            'version': 1,
            'best_step': next(
                (
                    item['step']
                    for item in reversed(history)
                    if item.get('improved')
                ),
                steps,
            ),
            'selection_metric': selection_metric,
            'best_metric_value': best_metric_value,
            'stale_evaluations': stale_evaluations,
            'early_stopped': early_stopped,
            'final_step': steps,
            'history': history,
        }
        write_metrics(payload, metrics_file)

        writer.add_scalars(
            'loss',
            {'train': train_metrics['loss'], 'val': validation['nll']},
            steps,
        )
        writer.add_scalars(
            'acc',
            {'train': train_metrics['accuracy'], 'val': validation['accuracy']},
            steps,
        )
        writer.add_scalar('val_brier', validation['brier'], steps)
        writer.add_scalar(
            'val_expected_pt_mse',
            validation['expected_pt_mse'],
            steps,
        )
        writer.add_scalar('lr', lr, steps)
        writer.flush()

        train_loss = 0.0
        train_correct = 0.0
        train_batches = 0
        return early_stopped

    def progress_total():
        next_save = (steps // save_every + 1) * save_every
        target = min(next_save, max_steps) if max_steps > 0 else next_save
        return max(1, target - steps)

    pb = tqdm(total=progress_total(), desc='TRAIN')
    for inputs, rank_by_players in train_data_loader:
        inputs = inputs.to(dtype=torch.float64, device=device)
        rank_by_players = rank_by_players.to(dtype=torch.int64, device=device)

        logits = grp.forward_packed(inputs)
        labels = grp.get_label(rank_by_players)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.inference_mode():
            train_loss += loss.item()
            train_correct += (
                logits.argmax(-1) == labels
            ).to(torch.float64).mean().item()
            train_batches += 1

        steps += 1
        pb.update(1)

        reached_checkpoint = steps % save_every == 0
        reached_target = should_stop_training(steps, max_steps)
        if reached_checkpoint or reached_target:
            pb.close()
            early_stopped = checkpoint_and_validate()
            approx_passes = steps * batch_size / (len(train_file_list) * 10)
            logging.info(
                f'total steps: {steps:,} est. passes: {approx_passes:8.3f}'
            )
            if early_stopped or reached_target:
                break
            pb = tqdm(total=progress_total(), desc='TRAIN')
    pb.close()
    writer.close()
    sync_device_before_exit()

if __name__ == '__main__':
    try:
        train()
    except KeyboardInterrupt:
        pass
    finally:
        # ROCm on Windows: also cover error exits (e.g. an invalid
        # control.state_file) so the traceback is followed by a real process
        # exit instead of a hang; see sync_device_before_exit().
        import sys
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
