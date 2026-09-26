import importlib.machinery
import importlib.util
import json
import logging
import os
import platform
import subprocess
import sys
import sysconfig
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

try:
    import orjson

    def json_loads(data):
        return orjson.loads(data)

    def json_dumps_compact(obj) -> str:
        return orjson.dumps(obj).decode("utf-8")
except ImportError:  # pragma: no cover - stdlib fallback
    def json_loads(data):
        return json.loads(data)

    def json_dumps_compact(obj) -> str:
        return json.dumps(obj, separators=(",", ":"))


REFERENCE_VERSION = 4
NATIVE_VERSION = 5
ACTION_SPACE = 44
OBS_SHAPE = (775, 34)
KAN_SELECT_ROW = 635
MAHJONGCOPILOT_COMMIT = "fff6b48"
NUKIDORA_ACTION = 40


class CompatibilityError(RuntimeError):
    pass


def _mask_nukidora_for_native_rules(masks, native_mask=None):
    if native_mask is None:
        return list(masks)

    native_mask = np.asarray(native_mask, dtype=np.bool_)
    if native_mask.shape != (ACTION_SPACE,):
        raise CompatibilityError(
            f"unexpected native action mask shape: {native_mask.shape}"
        )
    if native_mask[NUKIDORA_ACTION]:
        return list(masks)

    adapted = []
    for raw_mask in masks:
        mask = np.asarray(raw_mask, dtype=np.bool_).copy()
        if mask.shape != (ACTION_SPACE,):
            raise CompatibilityError(
                f"unexpected reference action mask shape: {mask.shape}"
            )
        mask[NUKIDORA_ACTION] = False
        adapted.append(mask)
    return adapted


def reference_filename():
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    machine = (platform.machine() or sysconfig.get_platform()).lower()
    if "amd64" in machine or "x86_64" in machine:
        machine = "x86_64"
    elif "arm64" in machine or "aarch64" in machine:
        machine = "aarch64"
    else:
        raise CompatibilityError(f"unsupported architecture: {platform.machine()}")

    system = platform.system()
    suffix = {
        "Windows": "pc-windows-msvc.pyd",
        "Linux": "unknown-linux-gnu.so",
        "Darwin": "apple-darwin.so",
    }.get(system)
    if suffix is None:
        raise CompatibilityError(f"unsupported operating system: {system}")
    return f"libriichi3p-{python_version}-{machine}-{suffix}"


def _as_reference_file(value, filename):
    candidate = Path(value).expanduser().resolve()
    return candidate / filename if candidate.is_dir() else candidate


def resolve_reference_path(path=None):
    filename = reference_filename()
    if path is not None:
        candidate = _as_reference_file(path, filename)
        if not candidate.is_file():
            raise CompatibilityError(f"libriichi3p reference not found: {candidate}")
        return candidate

    for key in ("MORTAL_LIBRIICHI3P", "LIBRIICHI3P_REFERENCE"):
        if value := os.environ.get(key):
            candidate = _as_reference_file(value, filename)
            if candidate.is_file():
                return candidate

    project_root = Path(__file__).resolve().parents[1]
    cache_path = project_root / ".cache" / "libriichi3p" / filename
    if cache_path.is_file():
        return cache_path

    repo_candidates = []
    if value := os.environ.get("MORTAL_MAHJONGCOPILOT_REPO"):
        repo_candidates.append(Path(value).expanduser().resolve())
    repo_candidates.append(project_root.parent / "MahjongCopilot")

    for repo in repo_candidates:
        if (repo / ".git").exists():
            object_name = f"{MAHJONGCOPILOT_COMMIT}:libriichi3p/{filename}"
            result = subprocess.run(
                ["git", "-C", str(repo), "show", object_name],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if result.returncode == 0:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(result.stdout)
                return cache_path

        direct = repo / "libriichi3p" / filename
        if direct.is_file():
            return direct

    raise CompatibilityError(
        "unable to locate the historical libriichi3p binary; set "
        "MORTAL_LIBRIICHI3P to the binary (or its containing directory), "
        "or set MORTAL_MAHJONGCOPILOT_REPO to a MahjongCopilot clone"
    )


def _validate_reference(module):
    actual = (
        module.consts.MAX_VERSION,
        module.consts.ACTION_SPACE,
        tuple(module.consts.obs_shape(REFERENCE_VERSION)),
    )
    expected = (REFERENCE_VERSION, ACTION_SPACE, OBS_SHAPE)
    if actual != expected:
        raise CompatibilityError(
            f"incompatible libriichi3p ABI: expected {expected}, got {actual}"
        )


def load_reference(path=None):
    reference_path = resolve_reference_path(path)
    existing = sys.modules.get("libriichi3p")
    if existing is not None:
        _validate_reference(existing)
        existing_path = Path(existing.__file__).resolve()
        if existing_path != reference_path:
            raise CompatibilityError(
                f"libriichi3p is already loaded from {existing_path}, "
                f"but the pinned binary is {reference_path}; restart Python"
            )
        return existing

    loader = importlib.machinery.ExtensionFileLoader(
        "libriichi3p", str(reference_path)
    )
    spec = importlib.util.spec_from_loader("libriichi3p", loader)
    if spec is None:
        raise CompatibilityError(f"cannot create module spec for {reference_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["libriichi3p"] = module
    try:
        loader.exec_module(module)
        _validate_reference(module)
    except Exception:
        sys.modules.pop("libriichi3p", None)
        raise
    return module


def adapt_event(event):
    event = dict(event)
    event_type = event.get("type")
    if event_type == "start_game":
        event["names"] = list(event.get("names", []))[:3] + [""]
    elif event_type == "start_kyoku":
        event["scores"] = list(event["scores"])[:3] + [0]
        event["tehais"] = list(event["tehais"])[:3] + [["?"] * 13]
    elif event_type in {"hora", "ryukyoku"} and event.get("deltas") is not None:
        event["deltas"] = list(event["deltas"])[:3] + [0]
    return event


def _augment_tile(tile):
    if not isinstance(tile, str) or len(tile) < 2:
        return tile
    if tile[1] == "p":
        return tile[0] + "s" + tile[2:]
    if tile[1] == "s":
        return tile[0] + "p" + tile[2:]
    return tile


def _augment_event(event):
    event = dict(event)
    event_type = event.get("type")
    if event_type == "start_kyoku":
        event["bakaze"] = _augment_tile(event["bakaze"])
        event["dora_marker"] = _augment_tile(event["dora_marker"])
        event["tehais"] = [
            [_augment_tile(tile) for tile in hand] for hand in event["tehais"]
        ]
    elif event_type in {"tsumo", "dahai", "nukidora"}:
        event["pai"] = _augment_tile(event["pai"])
    elif event_type in {"chi", "pon", "daiminkan", "kakan"}:
        event["pai"] = _augment_tile(event["pai"])
        event["consumed"] = [_augment_tile(tile) for tile in event["consumed"]]
    elif event_type == "ankan":
        event["consumed"] = [_augment_tile(tile) for tile in event["consumed"]]
    elif event_type == "dora":
        event["dora_marker"] = _augment_tile(event["dora_marker"])
    elif event_type == "hora" and event.get("ura_markers") is not None:
        event["ura_markers"] = [
            _augment_tile(tile) for tile in event["ura_markers"]
        ]
    return event


def prepare_replay_events(raw_log, *, augmented=False):
    """Parse and adapt a log once for all POV reference replays."""
    prepared = []
    for line in raw_log.splitlines():
        if not line.strip():
            continue
        event = json_loads(line)
        if augmented:
            event = _augment_event(event)
        prepared.append(json_dumps_compact(adapt_event(event)))
    return prepared


def augment_log(raw_log):
    return "\n".join(
        json_dumps_compact(_augment_event(json_loads(line)))
        for line in raw_log.splitlines()
        if line.strip()
    )


def _should_skip_nukidora(event, player_id):
    return (
        event.get("type") == "nukidora"
        and event.get("actor") is not None
        and event["actor"] != player_id
    )


def _can_recover_nukidora(native_state, reference_furiten):
    native_cans = getattr(native_state, "last_cans", None)
    return (
        getattr(native_cans, "can_ron_agari", False)
        and reference_furiten
    )


def _nukidora_ron_event(event):
    converted = dict(event)
    converted["type"] = "dahai"
    converted["tsumogiri"] = False
    return converted


class _ScriptedCaptureEngine:
    def __init__(self):
        self.name = "libriichi3p-replay"
        self.is_oracle = False
        self.version = REFERENCE_VERSION
        self.enable_quick_eval = False
        self.enable_rule_based_agari_guard = False
        self._event_index = None
        self._script = deque()
        self.captures = {}

    def begin_event(self, event_index, script):
        self._event_index = event_index
        self._script = deque(script)

    def finish_event(self):
        if self._script:
            remaining = [(sample, action) for sample, action, _ in self._script]
            raise CompatibilityError(
                f"reference bot did not request decisions {remaining} "
                f"at event {self._event_index}"
            )

    def react_batch(self, obs, masks, invisible_obs):
        del invisible_obs
        actions = []
        q_values = []
        returned_masks = []

        for feature, raw_mask in zip(obs, masks):
            feature = np.asarray(feature, dtype=np.float32).copy()
            mask = np.asarray(raw_mask, dtype=np.bool_).copy()
            selected = None

            if self._script:
                sample_index, expected_action, expected_kan_select = self._script[0]
                if feature.shape != OBS_SHAPE or mask.shape != (ACTION_SPACE,):
                    raise CompatibilityError('unexpected reference shapes during capture')
                kan_select = bool(feature[KAN_SELECT_ROW, 0])
                if (
                    kan_select == expected_kan_select
                    and 0 <= expected_action < mask.size
                    and mask[expected_action]
                ):
                    self._script.popleft()
                    self.captures[sample_index] = (feature, mask)
                    selected = expected_action

            if selected is None:
                legal = np.flatnonzero(mask)
                if legal.size == 0:
                    raise CompatibilityError(
                        f"reference bot produced an empty mask at event {self._event_index}"
                    )
                selected = 43 if mask[43] else int(legal[-1])

            q = np.zeros(mask.size, dtype=np.float32)
            q[selected] = 1.0
            actions.append(int(selected))
            q_values.append(q.tolist())
            returned_masks.append(mask.tolist())

        return actions, q_values, returned_masks, [True] * len(actions)


def capture_replay(
    raw_log=None,
    *,
    player_id,
    event_indices,
    actions,
    at_kan_select,
    reference_path=None,
    prepared_events=None,
):
    if not 0 <= player_id < 3:
        raise ValueError(f"invalid sanma player id: {player_id}")
    if not (len(event_indices) == len(actions) == len(at_kan_select)):
        raise ValueError("decision metadata lengths do not match")
    if prepared_events is None:
        if raw_log is None:
            raise ValueError("raw_log is required when prepared_events is absent")
        prepared_events = prepare_replay_events(raw_log)
    elif raw_log is not None:
        raise ValueError("pass either raw_log or prepared_events, not both")

    events = list(prepared_events)
    scripts = defaultdict(list)
    for sample_index, (event_index, action, kan_select) in enumerate(
        zip(event_indices, actions, at_kan_select)
    ):
        if not 0 <= event_index < len(events):
            raise ValueError(f"event index out of range: {event_index}")
        scripts[event_index].append((sample_index, int(action), bool(kan_select)))
    for script in scripts.values():
        # Mortal batches the optional kan-selection observation before the
        # primary action observation, while gameplay metadata stores them after.
        script.sort(key=lambda entry: not entry[2])

    module = load_reference(reference_path)
    engine = _ScriptedCaptureEngine()
    bot = module.mjai.Bot(engine, player_id)

    for event_index, event_json in enumerate(events):
        engine.begin_event(event_index, scripts.get(event_index, ()))
        bot.react(event_json)
        engine.finish_event()

    if len(engine.captures) != len(actions):
        missing = sorted(set(range(len(actions))) - engine.captures.keys())
        raise CompatibilityError(f"missing reference observations for samples {missing}")

    observations = []
    masks = []
    for sample_index in range(len(actions)):
        feature, mask = engine.captures[sample_index]
        if feature.shape != OBS_SHAPE or mask.shape != (ACTION_SPACE,):
            raise CompatibilityError(
                f"unexpected reference shapes: obs={feature.shape}, mask={mask.shape}"
            )
        observations.append(feature)
        masks.append(mask)
    return observations, masks


def capture_gameplay(game, raw_log=None, actions=None, *, reference_path=None, prepared_events=None):
    event_indices = game.take_event_indices()
    at_kan_select = game.take_at_kan_select()
    player_id = game.take_player_id()
    observations, masks = capture_replay(
        raw_log,
        player_id=player_id,
        event_indices=event_indices,
        actions=actions,
        at_kan_select=at_kan_select,
        reference_path=reference_path,
        prepared_events=prepared_events,
    )
    return observations, masks, player_id


class ReferenceBot:
    def __init__(self, engine, player_id, *, reference_path=None):
        if not 0 <= player_id < 3:
            raise ValueError(f"invalid sanma player id: {player_id}")
        reference = load_reference(reference_path)
        self.bot = reference.mjai.Bot(engine, player_id)

    @property
    def state(self):
        return self.bot.state

    def react(self, line, *, can_act=True):
        event = json_loads(line)
        event.pop("meta", None)
        adapted = json_dumps_compact(adapt_event(event))
        return self.bot.react(adapted, can_act=can_act)


class _BatchRequest:
    def __init__(self, obs, masks, invisible_obs, native_mask=None):
        self.obs = list(obs)
        self.masks = list(masks)
        self.invisible_obs = None if invisible_obs is None else list(invisible_obs)
        self.native_mask = native_mask
        self.done = threading.Event()
        self.result = None
        self.error = None


class _BatchContext:
    def __init__(self, delegate, active_workers):
        self.delegate = delegate
        self.active_workers = active_workers
        self.requests = []
        self.condition = threading.Condition()

    def submit(self, obs, masks, invisible_obs, native_mask=None):
        request = _BatchRequest(obs, masks, invisible_obs, native_mask)
        with self.condition:
            self.requests.append(request)
            self.condition.notify_all()
        request.done.wait()
        if request.error is not None:
            raise request.error
        return request.result

    def worker_done(self):
        with self.condition:
            self.active_workers -= 1
            self.condition.notify_all()

    def _evaluate(self, requests):
        lengths = [len(request.obs) for request in requests]
        obs = [value for request in requests for value in request.obs]
        masks = [
            value
            for request in requests
            for value in _mask_nukidora_for_native_rules(
                request.masks,
                request.native_mask,
            )
        ]

        try:
            if all(request.invisible_obs is None for request in requests):
                invisible_obs = None
            elif any(request.invisible_obs is None for request in requests):
                raise CompatibilityError("mixed oracle and non-oracle model requests")
            else:
                invisible_obs = [
                    value for request in requests for value in request.invisible_obs
                ]
            result = self.delegate.react_batch(obs, masks, invisible_obs)
            if len(result) != 4:
                raise CompatibilityError("model engine returned an invalid batch result")
            actions, q_values, returned_masks, is_greedy = result
            if not (
                len(actions)
                == len(q_values)
                == len(returned_masks)
                == len(is_greedy)
                == len(obs)
            ):
                raise CompatibilityError("model engine returned mismatched batch lengths")

            offset = 0
            for request, length in zip(requests, lengths):
                end = offset + length
                request.result = (
                    actions[offset:end],
                    q_values[offset:end],
                    returned_masks[offset:end],
                    is_greedy[offset:end],
                )
                offset = end
        except Exception as error:
            for request in requests:
                request.error = error
        finally:
            for request in requests:
                request.done.set()

    def run(self):
        while True:
            with self.condition:
                self.condition.wait_for(
                    lambda: self.active_workers == 0
                    or (
                        self.requests
                        and len(self.requests) >= self.active_workers
                    )
                )
                if self.active_workers == 0:
                    return
                requests = self.requests
                self.requests = []
            self._evaluate(requests)


class _ModelProxy:
    def __init__(self, delegate):
        if getattr(delegate, "version", None) != REFERENCE_VERSION:
            raise CompatibilityError(
                f"self-play model must use deployment version {REFERENCE_VERSION}"
            )
        if getattr(delegate, "is_oracle", False):
            raise CompatibilityError("reference self-play does not support oracle models")
        self.name = getattr(delegate, "name", "mortal-3p")
        self.version = REFERENCE_VERSION
        self.is_oracle = False
        self.enable_quick_eval = False
        self.enable_rule_based_agari_guard = getattr(
            delegate, "enable_rule_based_agari_guard", False
        )
        self._context = None
        self._thread = threading.local()
        self._lock = threading.Lock()

    def set_context(self, context):
        with self._lock:
            if self._context is not None:
                raise CompatibilityError("overlapping compatibility batches")
            self._context = context

    def clear_context(self):
        with self._lock:
            self._context = None

    def set_native_mask(self, native_mask):
        self._thread.native_mask = native_mask

    def clear_native_mask(self):
        self._thread.native_mask = None

    def react_batch(self, obs, masks, invisible_obs):
        with self._lock:
            context = self._context
        if context is None:
            raise CompatibilityError("model request outside a compatibility batch")
        return context.submit(
            obs,
            masks,
            invisible_obs,
            native_mask=getattr(self._thread, "native_mask", None),
        )


class CompatMjaiEngine:
    engine_type = "mjai-log"

    def __init__(self, delegate, *, reference_path=None, max_workers=64):
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self.delegate = delegate
        self.reference = load_reference(reference_path)
        self.proxy = _ModelProxy(delegate)
        self.name = getattr(delegate, "name", "mortal-3p")
        self.player_ids = None
        self.bots = {}
        self.processed_events = {}
        self.max_workers = max_workers

    def set_player_ids(self, player_ids):
        player_ids = [int(player_id) for player_id in player_ids]
        if any(not 0 <= player_id < 3 for player_id in player_ids):
            raise ValueError(f"invalid sanma player ids: {player_ids}")
        self.player_ids = player_ids

    def start_game(self, game_index):
        if self.player_ids is None:
            raise CompatibilityError("player ids must be set before starting games")
        player_id = self.player_ids[game_index]
        self.bots[game_index] = self.reference.mjai.Bot(self.proxy, player_id)
        self.processed_events[game_index] = 0

    @staticmethod
    def _event_json(event, *, nukidora_as_dahai=False):
        event = dict(event)
        event.pop("meta", None)
        if nukidora_as_dahai and event.get("type") == "nukidora":
            event["type"] = "dahai"
            event["tsumogiri"] = False
        return json_dumps_compact(adapt_event(event))

    @staticmethod
    def _event_json_for_bot(bot, event, player_id):
        nukidora_as_dahai = (
            event.get("type") == "nukidora"
            and event.get("actor") == player_id
            and getattr(bot.state, "self_riichi_accepted", False)
        )
        return CompatMjaiEngine._event_json(
            event,
            nukidora_as_dahai=nukidora_as_dahai,
        )

    def _recover_nukidora_furiten(self, game_state, bot, events):
        if events[-1].get("type") != "nukidora":
            return None
        native_state = getattr(game_state, "state", None)
        if not _can_recover_nukidora(
            native_state,
            getattr(bot.state, "at_furiten", False),
        ):
            return None

        game_index = int(game_state.game_index)
        player_id = self.player_ids[game_index]
        repaired = self.reference.mjai.Bot(self.proxy, player_id)
        skipped = 0

        for event in events[:-1]:
            is_harmful_nukidora = (
                _should_skip_nukidora(event, player_id)
            )
            if is_harmful_nukidora:
                skipped += 1
                continue
            repaired.react(
                self._event_json_for_bot(repaired, event, player_id),
                can_act=False,
            )

        reaction = repaired.react(self._event_json_for_bot(
            repaired,
            _nukidora_ron_event(events[-1]),
            player_id,
        ))
        if reaction is not None:
            self.bots[game_index] = repaired
            logging.warning(
                "recovered historical nukidora furiten mismatch for game %d "
                "(player %d, skipped %d event(s))",
                game_index,
                player_id,
                skipped,
            )
        return reaction

    def _react_one(self, game_state, context):
        game_index = int(game_state.game_index)
        try:
            if game_index not in self.bots:
                self.start_game(game_index)
            events = json_loads(game_state.events_json)
            start = self.processed_events[game_index]
            if start >= len(events):
                raise CompatibilityError(
                    f"game {game_index} did not provide any new events: "
                    f"processed={start}, received={len(events)}, "
                    f"last={events[-1] if events else None}, "
                    f"player_id={self.player_ids[game_index]}"
                )

            bot = self.bots[game_index]
            for event in events[start:-1]:
                bot.react(
                    self._event_json_for_bot(bot, event, self.player_ids[game_index]),
                    can_act=False,
                )
            native_mask = None
            native_state = getattr(game_state, "state", None)
            if native_state is not None:
                try:
                    _, native_mask = native_state.encode_obs(NATIVE_VERSION, False)
                except Exception as error:
                    raise CompatibilityError(
                        "native state could not provide an action mask"
                    ) from error

            self.proxy.set_native_mask(native_mask)
            try:
                reaction = bot.react(
                    self._event_json_for_bot(
                        bot,
                        events[-1],
                        self.player_ids[game_index],
                    )
                )
                if reaction is None:
                    reaction = self._recover_nukidora_furiten(game_state, bot, events)
            finally:
                self.proxy.clear_native_mask()
            if reaction is None:
                raise CompatibilityError(
                    f"reference bot returned no reaction for game {game_index}: "
                    f"player_id={self.player_ids[game_index]}, "
                    f"last={events[-1]}, "
                    f"native_cans={getattr(game_state, 'state', None) and game_state.state.last_cans}, "
                    f"reference_furiten={getattr(bot.state, 'at_furiten', None)}"
                )
            self.processed_events[game_index] = len(events)
            return reaction
        finally:
            context.worker_done()

    def _react_chunk(self, game_states):
        if not game_states:
            return []

        context = _BatchContext(self.delegate, len(game_states))
        self.proxy.set_context(context)
        try:
            with ThreadPoolExecutor(max_workers=len(game_states)) as executor:
                futures = [
                    executor.submit(self._react_one, game_state, context)
                    for game_state in game_states
                ]
                context.run()
                return [future.result() for future in futures]
        finally:
            self.proxy.clear_context()

    def react_batch(self, game_states):
        game_states = list(game_states)
        reactions = []
        for start in range(0, len(game_states), self.max_workers):
            reactions.extend(
                self._react_chunk(game_states[start : start + self.max_workers])
            )
        return reactions

    def end_kyoku(self, game_index):
        bot = self.bots.get(game_index)
        if bot is not None:
            bot.react('{"type":"end_kyoku"}', can_act=False)
        self.processed_events[game_index] = 0

    def end_game(self, game_index, scores):
        del scores
        bot = self.bots.pop(game_index, None)
        if bot is not None:
            bot.react('{"type":"end_game"}', can_act=False)
        self.processed_events.pop(game_index, None)
