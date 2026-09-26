from fractions import Fraction


def step_optimizer(optimizer, scaler, *, count_successful=False):
    previous_scale = scaler.get_scale() if count_successful else None
    scaler.step(optimizer)
    scaler.update()
    return not count_successful or scaler.get_scale() >= previous_scale


def batch_source(update_index, expert_ratio):
    if update_index < 0:
        raise ValueError('update_index must be non-negative')

    ratio = Fraction(str(expert_ratio)).limit_denominator(1000)
    if ratio < 0 or ratio > 1:
        raise ValueError('expert_ratio must be between 0 and 1')

    position = update_index % ratio.denominator
    online_batches = ratio.denominator - ratio.numerator
    return 'online' if position < online_batches else 'expert'


def repeating_batches(factory):
    """Keep a loader alive until exhausted; fail closed on an empty epoch."""
    while True:
        iterator = iter(factory())
        try:
            first = next(iterator)
        except StopIteration as error:
            raise RuntimeError('expert dataset produced no batches') from error
        yield first
        yield from iterator


def mixed_batches(
    online_batches,
    expert_batches_factory,
    expert_ratio,
    start_update_index=0,
):
    online_iter = iter(online_batches)
    expert_iter = None
    update_index = start_update_index

    while True:
        source = batch_source(update_index, expert_ratio)
        if source == 'online':
            try:
                batch = next(online_iter)
            except StopIteration:
                return
        else:
            if expert_iter is None:
                expert_iter = iter(expert_batches_factory())
            try:
                batch = next(expert_iter)
            except StopIteration:
                expert_iter = iter(expert_batches_factory())
                try:
                    batch = next(expert_iter)
                except StopIteration as exc:
                    raise RuntimeError('expert dataset produced no batches') from exc

        yield source, batch
        update_index += 1


def choose_weighted(items, weights, sample):
    if not items or len(items) != len(weights):
        raise ValueError('items and weights must have the same non-zero length')
    if not 0 <= sample < 1:
        raise ValueError('sample must be in [0, 1)')
    if any(weight < 0 for weight in weights):
        raise ValueError('weights must be non-negative')

    total = sum(weights)
    if total <= 0:
        raise ValueError('at least one weight must be positive')

    threshold = sample * total
    cumulative = 0
    for item, weight in zip(items, weights):
        cumulative += weight
        if threshold < cumulative:
            return item
    return items[-1]


def weighted_quota_index(weights, counts):
    if not weights or len(weights) != len(counts):
        raise ValueError('weights and counts must have the same non-zero length')
    if any(weight < 0 for weight in weights) or any(count < 0 for count in counts):
        raise ValueError('weights and counts must be non-negative')
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError('at least one weight must be positive')

    next_total = sum(counts) + 1
    deficits = [
        next_total * weight / total_weight - count
        for weight, count in zip(weights, counts)
    ]
    return max(range(len(deficits)), key=deficits.__getitem__)


def should_freeze_brain(step, start_step, freeze_updates):
    return step < start_step + freeze_updates


def should_snapshot(step, start_step, every):
    phase_updates = step - start_step
    return every > 0 and phase_updates > 0 and phase_updates % every == 0


def reached_max_steps(step, max_steps):
    return max_steps > 0 and step >= max_steps


def should_restart_online(exit_code, max_steps):
    return exit_code == 0 and max_steps <= 0


def resolve_online_resume_numbering(current_step, phase_start_step, target_step):
    """Resolve the (start, target) step window for a resuming online state.

    The online phase trains exactly ``target_step - phase_start_step``
    updates beyond its seed. When the on-disk state predates the phase's
    step numbering — it was seeded from an earlier offline snapshot — the
    phase window must be rebased onto the state's actual step; otherwise
    training would run from the stale step all the way to the old target
    (silent over-training). Returns ``(start_step, target_step, rebased)``.
    """
    online_updates = target_step - phase_start_step
    if online_updates < 0:
        raise ValueError('online target must not precede the phase start')
    if current_step < phase_start_step:
        return current_step, current_step + online_updates, True
    return phase_start_step, target_step, False


def should_resume_optimizer(online, source_online, resume_optimizer):
    return resume_optimizer and (not online or source_online)


def restore_scheduler_state(scheduler, saved_state, configured_state):
    saved_max_steps = int(saved_state.get('max_steps', scheduler.max_steps))
    scheduler.load_state_dict(saved_state)
    configured_max_steps = int(
        configured_state.get('max_steps', saved_max_steps)
    )
    if configured_max_steps <= saved_max_steps:
        return
    for name in (
        'init',
        'peak',
        'final',
        'warm_up_steps',
        'max_steps',
        'offset',
        'epoch_size',
    ):
        if name in configured_state:
            setattr(scheduler, name, configured_state[name])
