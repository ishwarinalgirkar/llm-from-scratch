"""Learning rate schedules (stub implementations)."""

def linear_warmup_decay(step: int, warmup_steps: int, total_steps: int, base_lr: float):
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)
    else:
        return base_lr * max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
