def seed_everything(seed: int, deterministic: bool = False) -> None:
    import random

    import numpy as np
    import torch

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def train():
    import prelude

    import logging
    import sys
    import os
    import gc
    import gzip
    import json
    import shutil
    import random
    import torch
    from os import path
    from glob import glob
    from datetime import datetime
    from itertools import chain
    from torch import optim, nn
    from torch.amp import GradScaler
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from common import submit_param, parameter_count, drain, filtered_trimmed_lines, tqdm
    from player import TestPlayer
    from dataloader import FileDatasetsIter, worker_init_fn
    from lr_scheduler import LinearWarmUpCosineAnnealingLR
    from model import Brain, DQN, AuxNet
    from checkpoint import export_deployment, initialize_from_baseline
    from online_training import (
        mixed_batches,
        repeating_batches,
        reached_max_steps,
        should_freeze_brain,
        should_resume_optimizer,
        restore_scheduler_state,
        should_snapshot,
        step_optimizer,
    )
    from libriichi.consts import obs_shape, NUM_PLAYERS
    from config import config

    # Valid obs-encoding versions per ruleset variant. The active variant is
    # determined by the loaded native extension (libriichi), not by config:
    # - yonma (4p): versions 1-4, where 4 is the native 1012-channel ABI
    # - sanma (3p): version 4 is the MahjongCopilot 775-channel deployment ABI,
    #   version 5 is the native 780-channel encoding
    expected_versions = {
        4: {1, 2, 3, 4},
        3: {4, 5},
    }[NUM_PLAYERS]
    default_display_pts = {
        4: [90, 45, 0, -135],
        3: [90, 0, -90],
    }[NUM_PLAYERS]

    version = config['control']['version']
    if version not in expected_versions:
        raise ValueError(
            f'control.version must be one of {sorted(expected_versions)} '
            f'for the active {NUM_PLAYERS}-player native extension, got {version}'
        )
    random_seed = config['control'].get('random_seed')
    deterministic = config['control'].get('deterministic', False)
    if random_seed is not None:
        seed_everything(random_seed, deterministic=deterministic)

    online = config['control']['online']
    batch_size = config['control']['batch_size']
    opt_step_every = config['control']['opt_step_every']
    count_optimizer_updates = config['control'].get('count_optimizer_updates', False)
    if count_optimizer_updates and opt_step_every != 1:
        raise ValueError('successful-update accounting requires opt_step_every=1')
    save_every = config['control']['save_every']
    test_every = config['control']['test_every']
    submit_every = config['control']['submit_every']
    test_games = config['test_play']['games']
    if test_games % NUM_PLAYERS != 0:
        raise ValueError(f'test_play.games must be divisible by {NUM_PLAYERS}')
    min_q_weight = config['cql']['min_q_weight']
    next_rank_weight = config['aux']['next_rank_weight']
    online_training_cfg = config.get('online_training', {})
    expert_ratio = online_training_cfg.get('expert_ratio', 0.0) if online else 0.0
    phase_start_step = online_training_cfg.get('phase_start_step', 0)
    brain_freeze_updates = online_training_cfg.get('brain_freeze_updates', 0)
    max_steps = config['control'].get('max_steps', 0)
    snapshot_every = config['control'].get('snapshot_every', 0)
    snapshot_dir = config['control'].get('snapshot_dir', '')
    resume_optimizer = config['control'].get('resume_optimizer', True)
    assert save_every % opt_step_every == 0
    assert test_every % save_every == 0
    if snapshot_every > 0:
        assert snapshot_dir

    device = torch.device(config['control']['device'])
    torch.backends.cudnn.benchmark = (
        config['control']['enable_cudnn_benchmark'] and not deterministic
    )
    torch.backends.cudnn.deterministic = deterministic
    enable_amp = config['control']['enable_amp']
    enable_compile = config['control']['enable_compile']

    pts = config['env']['pts']
    if len(pts) != NUM_PLAYERS:
        raise ValueError(f'env.pts must contain {NUM_PLAYERS} values, got {pts}')
    gamma = config['env']['gamma']
    file_batch_size = config['dataset']['file_batch_size']
    reserve_ratio = config['dataset']['reserve_ratio']
    num_workers = config['dataset']['num_workers']
    num_epochs = config['dataset']['num_epochs']
    enable_augmentation = config['dataset']['enable_augmentation']
    augmented_first = config['dataset']['augmented_first']
    eps = config['optim']['eps']
    betas = config['optim']['betas']
    weight_decay = config['optim']['weight_decay']
    max_grad_norm = config['optim']['max_grad_norm']
    # Held-out validation loss: val_globs must point at files that are NOT in
    # the training globs. val_loss_every = 0 (default) disables the eval.
    val_loss_every = config['control'].get('val_loss_every', 0)
    val_globs = config['dataset'].get('val_globs', [])
    val_max_batches = config['control'].get('val_max_batches', 50)
    # The per-batch mask assertion forces a GPU sync every step; keep a
    # tripwire at a coarser cadence (default: once per save window).
    mask_check_every = max(1, config['control'].get('mask_check_every', save_every))

    mortal = Brain(version=version, **config['resnet']).to(device)
    dqn = DQN(version=version).to(device)
    aux_net = AuxNet().to(device)
    all_models = (mortal, dqn, aux_net)
    if enable_compile:
        for m in all_models:
            m.compile()

    logging.info(f'version: {version}')
    logging.info(f'obs shape: {obs_shape(version)}')
    logging.info(f'mortal params: {parameter_count(mortal):,}')
    logging.info(f'dqn params: {parameter_count(dqn):,}')
    logging.info(f'aux params: {parameter_count(aux_net):,}')

    mortal.freeze_bn(config['freeze_bn']['mortal'])

    decay_params = []
    no_decay_params = []
    for model in all_models:
        params_dict = {}
        to_decay = set()
        for mod_name, mod in model.named_modules():
            for name, param in mod.named_parameters(prefix=mod_name, recurse=False):
                params_dict[name] = param
                if isinstance(mod, (nn.Linear, nn.Conv1d)) and name.endswith('weight'):
                    to_decay.add(name)
        decay_params.extend(params_dict[name] for name in sorted(to_decay))
        no_decay_params.extend(params_dict[name] for name in sorted(params_dict.keys() - to_decay))
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params},
    ]
    optimizer = optim.AdamW(param_groups, lr=1, weight_decay=0, betas=betas, eps=eps)
    scheduler = LinearWarmUpCosineAnnealingLR(optimizer, **config['optim']['scheduler'])
    scaler = GradScaler(device.type, enabled=enable_amp)
    test_player = TestPlayer()
    best_perf = {
        'avg_rank': float(NUM_PLAYERS),
        'avg_pt': float('-inf'),
    }

    steps = 0
    state_file = config['control']['state_file']
    best_state_file = config['control']['best_state_file']
    deployment_file = config['control'].get('deployment_file')
    best_deployment_file = config['control'].get('best_deployment_file')
    for output_file in filter(
        None,
        (state_file, best_state_file, deployment_file, best_deployment_file),
    ):
        if parent := path.dirname(path.abspath(output_file)):
            os.makedirs(parent, exist_ok=True)
    if path.exists(state_file):
        state = torch.load(state_file, weights_only=True, map_location=device)
        missing_keys = [
            key for key in (
                'mortal', 'current_dqn', 'aux_net', 'optimizer', 'scheduler',
                'scaler', 'steps', 'best_perf', 'timestamp', 'config',
            )
            if key not in state
        ]
        if missing_keys:
            missing = ', '.join(missing_keys)
            raise ValueError(
                f'control.state_file {state_file!r} is not a resumable training state '
                f'(missing keys: {missing}). Deployment-format checkpoints '
                '(config/mortal/current_dqn only, e.g. a copy of a baselines/*.pth file) cannot '
                'be resumed: remove or rename this file to initialize from control.init_from, '
                'or restore the real training state.'
            )
        timestamp = datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f'loaded: {timestamp}')
        mortal.load_state_dict(state['mortal'])
        dqn.load_state_dict(state['current_dqn'])
        aux_net.load_state_dict(state['aux_net'])
        source_online = state['config']['control'].get('online', False)
        if state['config']['control'].get('count_optimizer_updates', False) != count_optimizer_updates:
            raise ValueError('cannot change step numbering semantics while resuming a state')
        if should_resume_optimizer(online, source_online, resume_optimizer):
            optimizer.load_state_dict(state['optimizer'])
            restore_scheduler_state(
                scheduler,
                state['scheduler'],
                config['optim']['scheduler'],
            )
        scaler.load_state_dict(state['scaler'])
        best_perf = state['best_perf']
        steps = state['steps']
    elif baseline_file := config['control'].get('init_from'):
        initialize_from_baseline(baseline_file, mortal, dqn, map_location=device)
        logging.info(f'initialized Brain/DQN from {baseline_file}')

    optimizer.zero_grad(set_to_none=True)

    def take_optimizer_step():
        if max_grad_norm > 0:
            scaler.unscale_(optimizer)
            params = chain.from_iterable(group['params'] for group in optimizer.param_groups)
            clip_grad_norm_(params, max_grad_norm)
        updated = step_optimizer(optimizer, scaler, count_successful=count_optimizer_updates)
        optimizer.zero_grad(set_to_none=True)
        return updated

    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    if device.type == 'cuda':
        logging.info(f'device: {device} ({torch.cuda.get_device_name(device)})')
    else:
        logging.info(f'device: {device}')

    if online:
        submit_param(mortal, dqn, is_idle=True)
        logging.info('param has been submitted')

    writer = SummaryWriter(config['control']['tensorboard_dir'])
    stats = {
        'dqn_loss': 0,
        'cql_loss': 0,
        'next_rank_loss': 0,
    }
    all_q = torch.zeros((save_every, batch_size), device=device, dtype=torch.float32)
    all_q_target = torch.zeros((save_every, batch_size), device=device, dtype=torch.float32)
    idx = 0
    brain_frozen = None

    def update_brain_freeze():
        nonlocal brain_frozen
        freeze = online and should_freeze_brain(
            steps,
            phase_start_step,
            brain_freeze_updates,
        )
        if freeze == brain_frozen:
            return
        mortal.requires_grad_(not freeze)
        mortal.freeze_bn(config['freeze_bn']['mortal'])
        brain_frozen = freeze
        logging.info(f'brain frozen: {brain_frozen}')

    def load_configured_dataset():
        player_names_set = set()
        for filename in config['dataset']['player_names_files']:
            with open(filename, encoding='utf-8') as f:
                player_names_set.update(filtered_trimmed_lines(f))
        player_names = list(player_names_set)
        logging.info(f'loaded {len(player_names):,} players')

        file_index = config['dataset']['file_index']
        if path.exists(file_index):
            index = torch.load(file_index, weights_only=True)
            file_list = index['file_list']
        else:
            logging.info('building file index...')
            file_list = []
            for pat in config['dataset']['globs']:
                file_list.extend(glob(pat, recursive=True))
            if player_names_set:
                filtered = []
                for filename in tqdm(file_list, unit='file'):
                    with gzip.open(filename, 'rt') as f:
                        start = json.loads(next(f))
                        if not set(start['names']).isdisjoint(player_names_set):
                            filtered.append(filename)
                file_list = filtered
            file_list.sort(reverse=True)
            torch.save({'file_list': file_list}, file_index)
        return player_names, file_list

    def make_data_loader(file_list, player_names):
        files = list(file_list)
        if num_workers > 1:
            random.shuffle(files)
        file_data = FileDatasetsIter(
            version = version,
            observation_encoder = config['dataset'].get('observation_encoder'),
            # Backward-compatible alias: existing native_version=5 sanma
            # configs continue to select the historical compat_v4 path.
            native_version = config['dataset'].get('native_version'),
            reference_path = config.get('compat', {}).get('libriichi3p') or None,
            file_list = files,
            pts = pts,
            file_batch_size = file_batch_size,
            reserve_ratio = reserve_ratio,
            player_names = player_names,
            num_epochs = num_epochs,
            enable_augmentation = enable_augmentation,
            augmented_first = augmented_first,
        )
        return DataLoader(
            dataset = file_data,
            batch_size = batch_size,
            drop_last = True,
            num_workers = num_workers,
            pin_memory = True,
            worker_init_fn = worker_init_fn,
        )

    def evaluate_val_loss():
        """Forward-only loss on a held-out file set; strength-blind but cheap.

        Same sample format as training, no gradient, model in eval mode. The
        RNG is seeded for the duration so the val file/buffer shuffles are
        stable across evaluations, then restored for training.
        """
        val_files = []
        for pattern in val_globs:
            val_files.extend(glob(pattern, recursive=True))
        val_files.sort()
        if not val_files:
            logging.warning('val_globs set but no val files found; skipping val loss')
            return
        rng_state = random.getstate()
        random.seed(20260831)
        try:
            loader = make_data_loader(val_files, [])
            mortal.eval(); dqn.eval(); aux_net.eval()
            v_dqn = v_rank = 0.0
            batches = 0
            with torch.inference_mode():
                for batch in loader:
                    obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks = batch
                    obs = obs.to(dtype=torch.float32, device=device)
                    actions = actions.to(dtype=torch.int64, device=device)
                    masks = masks.to(dtype=torch.bool, device=device)
                    steps_to_done = steps_to_done.to(dtype=torch.int64, device=device)
                    kyoku_rewards = kyoku_rewards.to(dtype=torch.float64, device=device)
                    player_ranks = player_ranks.to(dtype=torch.int64, device=device)
                    with torch.autocast(device.type, enabled=enable_amp):
                        phi = mortal(obs)
                        q_out = dqn(phi, masks)
                        q = q_out[range(batch_size), actions]
                        q_target_mc = (gamma ** steps_to_done * kyoku_rewards).to(torch.float32)
                        v_dqn += (0.5 * mse(q, q_target_mc)).item()
                        next_rank_logits, = aux_net(phi)
                        v_rank += ce(next_rank_logits, player_ranks).item()
                    batches += 1
                    if batches >= val_max_batches:
                        break
            if batches:
                writer.add_scalar('val/dqn_loss', v_dqn / batches, steps)
                writer.add_scalar('val/next_rank_loss', v_rank / batches, steps)
                logging.info(
                    f'val loss: dqn {v_dqn/batches:.4f} rank {v_rank/batches:.4f} '
                    f'({batches} batches)'
                )
        finally:
            random.setstate(rng_state)
            mortal.train(); dqn.train(); aux_net.train()

    def build_state():
        return {
            'mortal': mortal.state_dict(),
            'current_dqn': dqn.state_dict(),
            'aux_net': aux_net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'steps': steps,
            'timestamp': datetime.now().timestamp(),
            'best_perf': best_perf,
            'config': config,
        }

    def save_state(state=None):
        if state is None:
            state = build_state()
        torch.save(state, state_file)
        if deployment_file:
            export_deployment(deployment_file, mortal, dqn, config)
        if should_snapshot(steps, phase_start_step, snapshot_every):
            os.makedirs(snapshot_dir, exist_ok=True)
            snapshot_file = path.join(snapshot_dir, f'mortal_step{steps}.pth')
            if path.exists(snapshot_file):
                logging.info(f'snapshot already exists: {snapshot_file}')
            else:
                torch.save(state, snapshot_file)
                logging.info(f'snapshot saved: {snapshot_file}')
        return state

    expert_player_names = []
    expert_file_list = []
    if online and expert_ratio > 0:
        expert_player_names, expert_file_list = load_configured_dataset()
        if not expert_file_list:
            raise RuntimeError('expert replay is enabled but no expert files were found')

    def sync_device_before_exit():
        # ROCm on Windows (torch 2.9.1+rocm7.2.1, RX 9070 XT): the process can
        # hang forever at interpreter exit when the device was touched but its
        # stream was never synchronized. The normal training path synchronizes
        # implicitly (.item(), torch.save); the early return below did not.
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    update_brain_freeze()
    if reached_max_steps(steps, max_steps):
        logging.info(f'max steps already reached: {steps:,}')
        writer.close()
        sync_device_before_exit()
        return

    expert_factory = lambda: make_data_loader(expert_file_list, expert_player_names)
    if online and expert_ratio > 0 and config.get('online_training', {}).get('reuse_expert_iterator', False):
        expert_stream = repeating_batches(expert_factory)
        expert_factory = lambda: expert_stream

    def train_epoch():
        nonlocal steps
        nonlocal idx

        player_names = []
        if online:
            player_names = ['trainee']
            dirname = drain()
            file_list = list(map(lambda p: path.join(dirname, p), os.listdir(dirname)))
        else:
            player_names, file_list = load_configured_dataset()
        logging.info(f'file list size: {len(file_list):,}')

        before_next_test_play = (test_every - steps % test_every) % test_every
        logging.info(f'total steps: {steps:,} (~{before_next_test_play:,})')

        data_loader = make_data_loader(file_list, player_names)
        if online and expert_ratio > 0:
            batches = mixed_batches(
                data_loader,
                expert_factory,
                expert_ratio,
                start_update_index=steps - phase_start_step,
            )
        else:
            source = 'online' if online else 'expert'
            batches = ((source, batch) for batch in data_loader)

        pb = tqdm(total=save_every, desc='TRAIN', initial=steps % save_every)

        def train_batch(
            obs,
            actions,
            masks,
            steps_to_done,
            kyoku_rewards,
            player_ranks,
            use_cql,
        ):
            nonlocal steps
            nonlocal idx
            nonlocal pb

            update_brain_freeze()
            obs = obs.to(dtype=torch.float32, device=device)
            actions = actions.to(dtype=torch.int64, device=device)
            masks = masks.to(dtype=torch.bool, device=device)
            steps_to_done = steps_to_done.to(dtype=torch.int64, device=device)
            kyoku_rewards = kyoku_rewards.to(dtype=torch.float64, device=device)
            player_ranks = player_ranks.to(dtype=torch.int64, device=device)
            if steps % mask_check_every == 0:
                assert masks[range(batch_size), actions].all()

            q_target_mc = gamma ** steps_to_done * kyoku_rewards
            q_target_mc = q_target_mc.to(torch.float32)

            with torch.autocast(device.type, enabled=enable_amp):
                phi = mortal(obs)
                q_out = dqn(phi, masks)
                q = q_out[range(batch_size), actions]
                dqn_loss = 0.5 * mse(q, q_target_mc)
                cql_loss = 0
                if use_cql:
                    cql_loss = q_out.logsumexp(-1).mean() - q.mean()

                next_rank_logits, = aux_net(phi)
                next_rank_loss = ce(next_rank_logits, player_ranks)

                loss = sum((
                    dqn_loss,
                    cql_loss * min_q_weight,
                    next_rank_loss * next_rank_weight,
                ))
            scaler.scale(loss / opt_step_every).backward()
            if count_optimizer_updates and not take_optimizer_step():
                logging.info('AMP skipped optimizer update; step and scheduler budgets unchanged')
                return False

            with torch.inference_mode():
                stats['dqn_loss'] += dqn_loss
                stats['cql_loss'] += cql_loss
                stats['next_rank_loss'] += next_rank_loss
                all_q[idx] = q
                all_q_target[idx] = q_target_mc

            steps += 1
            idx += 1
            if not count_optimizer_updates and idx % opt_step_every == 0:
                take_optimizer_step()
            scheduler.step()
            pb.update(1)

            if online and steps % submit_every == 0:
                submit_param(mortal, dqn, is_idle=False)
                logging.info('param has been submitted')

            if steps % save_every == 0:
                pb.close()

                # downsample to reduce tensorboard event size
                all_q_1d = all_q.cpu().numpy().flatten()[::128]
                all_q_target_1d = all_q_target.cpu().numpy().flatten()[::128]

                writer.add_scalar('loss/dqn_loss', stats['dqn_loss'] / save_every, steps)
                if not online or expert_ratio > 0:
                    writer.add_scalar('loss/cql_loss', stats['cql_loss'] / save_every, steps)
                writer.add_scalar('loss/next_rank_loss', stats['next_rank_loss'] / save_every, steps)
                writer.add_scalar('hparam/lr', scheduler.get_last_lr()[0], steps)
                writer.add_histogram('q_predicted', all_q_1d, steps)
                writer.add_histogram('q_target', all_q_target_1d, steps)
                writer.flush()

                for k in stats:
                    stats[k] = 0
                idx = 0

                before_next_test_play = (test_every - steps % test_every) % test_every
                logging.info(f'total steps: {steps:,} (~{before_next_test_play:,})')

                state = save_state()

                if val_loss_every > 0 and steps % val_loss_every == 0:
                    evaluate_val_loss()

                if online and steps % submit_every != 0:
                    submit_param(mortal, dqn, is_idle=False)
                    logging.info('param has been submitted')

                if steps % test_every == 0:
                    stat = test_player.test_play(test_games // NUM_PLAYERS, mortal, dqn, device)
                    mortal.train()
                    dqn.train()

                    display_pts = config['test_play'].get('pts', default_display_pts)
                    avg_pt = stat.avg_pt(display_pts) # for display only, never used in training
                    better = avg_pt >= best_perf['avg_pt'] and stat.avg_rank <= best_perf['avg_rank']
                    if better:
                        past_best = best_perf.copy()
                        best_perf['avg_pt'] = avg_pt
                        best_perf['avg_rank'] = stat.avg_rank

                    logging.info(f'avg rank: {stat.avg_rank:.6}')
                    logging.info(f'avg pt: {avg_pt:.6}')
                    writer.add_scalar('test_play/avg_ranking', stat.avg_rank, steps)
                    writer.add_scalar('test_play/avg_pt', avg_pt, steps)
                    rank_labels = ['1st', '2nd', '3rd', '4th'][:NUM_PLAYERS]
                    rank_rates = [
                        stat.rank_1_rate,
                        stat.rank_2_rate,
                        stat.rank_3_rate,
                        stat.rank_4_rate,
                    ][:NUM_PLAYERS]
                    writer.add_scalars('test_play/ranking', dict(zip(rank_labels, rank_rates)), steps)
                    writer.add_scalars('test_play/behavior', {
                        'agari': stat.agari_rate,
                        'houjuu': stat.houjuu_rate,
                        'fuuro': stat.fuuro_rate,
                        'riichi': stat.riichi_rate,
                    }, steps)
                    writer.add_scalars('test_play/agari_point', {
                        'overall': stat.avg_point_per_agari,
                        'riichi': stat.avg_point_per_riichi_agari,
                        'fuuro': stat.avg_point_per_fuuro_agari,
                        'dama': stat.avg_point_per_dama_agari,
                    }, steps)
                    writer.add_scalar('test_play/houjuu_point', stat.avg_point_per_houjuu, steps)
                    writer.add_scalar('test_play/point_per_round', stat.avg_point_per_round, steps)
                    writer.add_scalars('test_play/key_step', {
                        'agari_jun': stat.avg_agari_jun,
                        'houjuu_jun': stat.avg_houjuu_jun,
                        'riichi_jun': stat.avg_riichi_jun,
                    }, steps)
                    writer.add_scalars('test_play/riichi', {
                        'agari_after_riichi': stat.agari_rate_after_riichi,
                        'houjuu_after_riichi': stat.houjuu_rate_after_riichi,
                        'chasing_riichi': stat.chasing_riichi_rate,
                        'riichi_chased': stat.riichi_chased_rate,
                    }, steps)
                    writer.add_scalar('test_play/riichi_point', stat.avg_riichi_point, steps)
                    writer.add_scalars('test_play/fuuro', {
                        'agari_after_fuuro': stat.agari_rate_after_fuuro,
                        'houjuu_after_fuuro': stat.houjuu_rate_after_fuuro,
                    }, steps)
                    writer.add_scalar('test_play/fuuro_num', stat.avg_fuuro_num, steps)
                    writer.add_scalar('test_play/fuuro_point', stat.avg_fuuro_point, steps)
                    writer.flush()

                    if better:
                        state['best_perf'] = best_perf
                        torch.save(state, state_file)
                        if deployment_file:
                            export_deployment(deployment_file, mortal, dqn, config)
                        if best_deployment_file:
                            export_deployment(best_deployment_file, mortal, dqn, config)
                        if best_state_file:
                            logging.info(
                                'a new record has been made, '
                                f'pt: {past_best["avg_pt"]:.4} -> {best_perf["avg_pt"]:.4}, '
                                f'rank: {past_best["avg_rank"]:.4} -> {best_perf["avg_rank"]:.4}, '
                                f'saving to {best_state_file}'
                            )
                            shutil.copy(state_file, best_state_file)
                    if online:
                        # BUG: This is a bug with unknown reason. When training
                        # in online mode, the process will get stuck here. This
                        # is the reason why `main` spawns a sub process to train
                        # in online mode instead of going for training directly.
                        sys.exit(0)
                pb = tqdm(total=save_every, desc='TRAIN')

            finished = reached_max_steps(steps, max_steps)
            if (
                steps % save_every != 0
                and (finished or should_snapshot(
                    steps,
                    phase_start_step,
                    snapshot_every,
                ))
            ):
                save_state()
            return finished

        finished = False
        for source, batch in batches:
            if train_batch(*batch, use_cql=source == 'expert'):
                finished = True
                break
        pb.close()

        if online:
            submit_param(mortal, dqn, is_idle=True)
            logging.info('param has been submitted')
        return finished

    while True:
        finished = train_epoch()
        gc.collect()
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()
        if finished or not online:
            # only run one epoch for offline for easier control
            break
    writer.close()
    sync_device_before_exit()

def main():
    import os
    import sys
    import time
    from subprocess import Popen
    from config import config
    from online_training import should_restart_online

    # do not set this env manually
    is_sub_proc_key = 'MORTAL_IS_SUB_PROC'
    online = config['control']['online']
    if not online or os.environ.get(is_sub_proc_key, '0') == '1':
        train()
        return

    cmd = (sys.executable, __file__)
    env = {
        is_sub_proc_key: '1',
        **os.environ.copy(),
    }
    while True:
        child = Popen(
            cmd,
            stdin = sys.stdin,
            stdout = sys.stdout,
            stderr = sys.stderr,
            env = env,
        )
        code = child.wait()
        if code != 0:
            sys.exit(code)
        max_steps = config['control'].get('max_steps', 0)
        if not should_restart_online(code, max_steps):
            return
        time.sleep(3)

if __name__ == '__main__':
    try:
        main()
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
