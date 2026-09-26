import gzip
import logging
import random
import torch
import numpy as np
from torch.utils.data import IterableDataset
from model import GRP
from reward_calculator import RewardCalculator
from libriichi.consts import NUM_PLAYERS
from libriichi.dataset import GameplayLoader
from config import config

class FileDatasetsIter(IterableDataset):
    """Streams training samples from mjai replay logs.

    ``observation_encoder`` selects the sanma deployment observation source:

    - ``compat_v4`` replays native metadata through the historical reference;
    - ``native_v4`` encodes the frozen v4 ABI directly in Rust;
    - ``native`` uses the requested native version directly (yonma default).

    ``native_version`` remains as a backward-compatible alias for
    ``compat_v4`` so existing sanma configurations retain their old behavior.
    """

    def __init__(
        self,
        version,
        file_list,
        pts,
        oracle = False,
        file_batch_size = 20, # hint: around 660 instances per file
        reserve_ratio = 0,
        player_names = None,
        excludes = None,
        num_epochs = 1,
        enable_augmentation = False,
        augmented_first = False,
        native_version = None,
        reference_path = None,
        observation_encoder = None,
    ):
        super().__init__()
        if observation_encoder is None:
            if native_version is not None or NUM_PLAYERS == 3:
                observation_encoder = 'compat_v4'
            else:
                observation_encoder = 'native'
        if observation_encoder not in {'native', 'compat_v4', 'native_v4'}:
            raise ValueError(
                'observation_encoder must be native, compat_v4, or native_v4; '
                f'got {observation_encoder!r}'
            )
        if native_version is not None and observation_encoder != 'compat_v4':
            raise ValueError(
                'native_version is a compat_v4 alias and cannot be combined '
                f'with observation_encoder={observation_encoder!r}'
            )

        self.version = version
        self.observation_encoder = observation_encoder
        self.reference_path = reference_path
        self.use_compat_replay = observation_encoder == 'compat_v4'
        self.native_version = native_version
        if self.use_compat_replay:
            # lazy import: only the sanma compat replay path needs the
            # historical libriichi3p reference binary; a yonma environment
            # must never have to provide it.
            from libriichi3p_compat import NATIVE_VERSION, REFERENCE_VERSION
            if NUM_PLAYERS != 3 or version != REFERENCE_VERSION:
                raise ValueError(
                    f'compat_v4 requires sanma version {REFERENCE_VERSION}, '
                    f'got players={NUM_PLAYERS}, version={version}'
                )
            if oracle:
                raise ValueError('historical libriichi3p compatibility does not expose oracle observations')
            if native_version is None:
                native_version = NATIVE_VERSION
            if native_version != NATIVE_VERSION:
                raise ValueError(
                    f'native sanma metadata must use version {NATIVE_VERSION}, got {native_version}'
                )
            self.native_version = native_version
        elif observation_encoder == 'native_v4':
            if NUM_PLAYERS != 3 or version != 4:
                raise ValueError(
                    f'native_v4 requires sanma version 4, got '
                    f'players={NUM_PLAYERS}, version={version}'
                )
            if oracle:
                raise ValueError('native_v4 oracle observations are not supported')
            self.native_version = None
        else:
            self.native_version = None
        self.file_list = file_list
        self.pts = pts
        self.oracle = oracle
        self.file_batch_size = file_batch_size
        self.reserve_ratio = reserve_ratio
        self.player_names = player_names
        self.excludes = excludes
        self.num_epochs = num_epochs
        self.enable_augmentation = enable_augmentation
        self.augmented_first = augmented_first
        self.iterator = None

    def build_iter(self):
        # do not put it in __init__, it won't work on Windows
        self.grp = GRP(**config['grp']['network'])
        grp_state = torch.load(config['grp']['state_file'], weights_only=True, map_location=torch.device('cpu'))
        self.grp.load_state_dict(grp_state['model'])
        self.reward_calc = RewardCalculator(self.grp, self.pts)

        for _ in range(self.num_epochs):
            yield from self.load_files(self.augmented_first)
            if self.enable_augmentation:
                yield from self.load_files(not self.augmented_first)

    def load_files(self, augmented):
        # shuffle the file list for each epoch
        random.shuffle(self.file_list)

        loader_kwargs = dict(
            version = self.native_version if self.use_compat_replay else self.version,
            oracle = self.oracle,
            player_names = self.player_names,
            excludes = self.excludes,
            augmented = augmented,
            # Compat replay captures the deployment obs/masks separately;
            # avoid generating and immediately discarding native v5 tensors.
            encode_observations = not self.use_compat_replay,
        )
        try:
            self.loader = GameplayLoader(**loader_kwargs)
            self._metadata_only_loader = self.use_compat_replay
        except TypeError as error:
            if 'encode_observations' not in str(error):
                raise
            # Keep old pre-optimization native extensions usable. They still
            # produce v5 tensors, which the compat path discards as before.
            logging.warning(
                'native GameplayLoader lacks encode_observations; '
                'falling back to full metadata pass'
            )
            loader_kwargs.pop('encode_observations')
            self.loader = GameplayLoader(**loader_kwargs)
            self._metadata_only_loader = False
        self.buffer = []

        for start_idx in range(0, len(self.file_list), self.file_batch_size):
            old_buffer_size = len(self.buffer)
            self.populate_buffer(
                self.file_list[start_idx:start_idx + self.file_batch_size],
                augmented=augmented,
            )
            buffer_size = len(self.buffer)

            reserved_size = int((buffer_size - old_buffer_size) * self.reserve_ratio)
            if reserved_size > buffer_size:
                continue

            random.shuffle(self.buffer)
            yield from self.buffer[reserved_size:]
            del self.buffer[reserved_size:]
        random.shuffle(self.buffer)
        yield from self.buffer
        self.buffer.clear()

    def populate_buffer(self, file_list, augmented=False):
        data = self.loader.load_gz_log_files(file_list)
        if self.use_compat_replay:
            self._populate_buffer_compat(file_list, data, augmented)
        else:
            self._populate_buffer_direct(file_list, data)

    def _populate_buffer_direct(self, file_list, data):
        if len(data) != len(file_list):
            raise RuntimeError('native loader returned an unexpected file count')
        for filename, games in zip(file_list, data):
            for game_index, game in enumerate(games):
                # per move
                obs = game.take_obs()
                if self.oracle:
                    invisible_obs = game.take_invisible_obs()
                actions = game.take_actions()
                masks = game.take_masks()
                at_kyoku = game.take_at_kyoku()
                dones = game.take_dones()
                apply_gamma = game.take_apply_gamma()
                event_indices = game.take_event_indices()
                at_kan_select = game.take_at_kan_select()

                # per game
                grp = game.take_grp()
                player_id = game.take_player_id()

                game_size = len(obs)
                counts = {
                    'obs': len(obs),
                    'actions': len(actions),
                    'masks': len(masks),
                    'at_kyoku': len(at_kyoku),
                    'dones': len(dones),
                    'apply_gamma': len(apply_gamma),
                    'event_indices': len(event_indices),
                    'at_kan_select': len(at_kan_select),
                }
                if len(set(counts.values())) != 1:
                    raise RuntimeError(
                        f'native sample counts differ file={filename} '
                        f'game={game_index} player={player_id}: {counts}'
                    )
                if not actions:
                    continue

                grp_feature = grp.take_feature()
                rank_by_player = grp.take_rank_by_player()
                kyoku_rewards = self.reward_calc.calc_delta_pt(player_id, grp_feature, rank_by_player)
                assert len(kyoku_rewards) >= at_kyoku[-1] + 1 # usually they are equal, unless there is no action in the last kyoku

                final_scores = grp.take_final_scores()
                scores_seq = np.concatenate((grp_feature[:, 3:] * 1e4, [final_scores]))
                rank_by_player_seq = (-scores_seq).argsort(-1, kind='stable').argsort(-1, kind='stable')
                player_ranks = rank_by_player_seq[:, player_id]

                steps_to_done = np.zeros(game_size, dtype=np.int64)
                for i in reversed(range(game_size)):
                    if not dones[i]:
                        steps_to_done[i] = steps_to_done[i + 1] + int(apply_gamma[i])

                for i in range(game_size):
                    action = int(actions[i])
                    mask = masks[i]
                    if action < 0 or action >= len(mask) or not bool(mask[action]):
                        raise RuntimeError(
                            f'native action is illegal file={filename} '
                            f'game={game_index} player={player_id} sample={i} '
                            f'event={event_indices[i]} kan_select={at_kan_select[i]} '
                            f'action={action} legal='
                            f'{np.flatnonzero(np.asarray(mask, dtype=bool)).tolist()}'
                        )
                    entry = [
                        obs[i],
                        actions[i],
                        masks[i],
                        steps_to_done[i],
                        kyoku_rewards[at_kyoku[i]],
                        player_ranks[at_kyoku[i] + 1],
                    ]
                    if self.oracle:
                        entry.insert(1, invisible_obs[i])
                    self.buffer.append(entry)

    def _populate_buffer_compat(self, file_list, data, augmented):
        # lazy import: sanma-only reference replay (see __init__)
        from libriichi3p_compat import (
            CompatibilityError,
            capture_gameplay,
            prepare_replay_events,
        )

        if len(data) != len(file_list):
            raise RuntimeError('native loader returned an unexpected file count')
        for filename, games in zip(file_list, data):
            with gzip.open(filename, 'rt', encoding='utf-8') as file:
                raw_log = file.read()
            prepared_events = prepare_replay_events(raw_log, augmented=augmented)

            for game_index, game in enumerate(games):
                # Metadata-only native loading deliberately does not build
                # v5 observations or masks; actions are authoritative there.
                actions = game.take_actions()
                if getattr(self, '_metadata_only_loader', False):
                    native_obs_count = len(actions)
                    native_masks_count = len(actions)
                else:
                    native_obs_count = len(game.take_obs())
                    native_masks_count = len(game.take_masks())
                at_kyoku = game.take_at_kyoku()
                dones = game.take_dones()
                apply_gamma = game.take_apply_gamma()

                if not actions:
                    continue
                try:
                    obs, masks, player_id = capture_gameplay(
                        game,
                        prepared_events=prepared_events,
                        actions=actions,
                        reference_path=self.reference_path,
                    )
                except CompatibilityError as error:
                    logging.warning(
                        "skipping incompatible sanma replay file=%s game=%d: %s",
                        filename,
                        game_index,
                        error,
                    )
                    continue
                if not (
                    native_obs_count
                    == native_masks_count
                    == len(obs)
                    == len(masks)
                    == len(actions)
                ):
                    raise RuntimeError('native and reference sample counts do not match')

                # per game
                grp = game.take_grp()

                game_size = len(obs)

                grp_feature = grp.take_feature()
                rank_by_player = grp.take_rank_by_player()
                kyoku_rewards = self.reward_calc.calc_delta_pt(player_id, grp_feature, rank_by_player)
                assert len(kyoku_rewards) >= at_kyoku[-1] + 1 # usually they are equal, unless there is no action in the last kyoku

                final_scores = grp.take_final_scores()
                scores_seq = np.concatenate((grp_feature[:, 3:] * 1e4, [final_scores]))
                rank_by_player_seq = (-scores_seq).argsort(-1, kind='stable').argsort(-1, kind='stable')
                player_ranks = rank_by_player_seq[:, player_id]

                steps_to_done = np.zeros(game_size, dtype=np.int64)
                for i in reversed(range(game_size)):
                    if not dones[i]:
                        steps_to_done[i] = steps_to_done[i + 1] + int(apply_gamma[i])

                for i in range(game_size):
                    a = int(actions[i])
                    m = masks[i]
                    if a >= len(m) or not bool(m[a]):
                        continue
                    entry = [
                        obs[i],
                        actions[i],
                        masks[i],
                        steps_to_done[i],
                        kyoku_rewards[at_kyoku[i]],
                        player_ranks[at_kyoku[i] + 1],
                    ]
                    self.buffer.append(entry)

    def __iter__(self):
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator

def worker_init_fn(*args, **kwargs):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    per_worker = int(np.ceil(len(dataset.file_list) / worker_info.num_workers))
    start = worker_info.id * per_worker
    end = start + per_worker
    dataset.file_list = dataset.file_list[start:end]
