"""Resume the original H3 denoising loop without changing its math."""

import hashlib
import inspect
import os
from pathlib import Path
import runpy
import sys

import torch
from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3DenoiseLoopWrapper


checkpoint = Path(os.environ["H3_REFERENCE_CHECKPOINT"])
stop_at = 24
original = MiniMaxH3DenoiseLoopWrapper.loop_step
fields = ("latents", "audio_latents")
saved = None


def code_hash(callable_object):
    return hashlib.sha256(inspect.getsource(callable_object).encode()).hexdigest()


def schedule_state(scheduler):
    if type(scheduler).__name__ != "MiniMaxH3Scheduler" or scheduler.order != 1:
        raise RuntimeError("Unexpected H3 scheduler; checkpoint layout is unverified")
    if getattr(scheduler.config, "stochastic_sampling", False):
        raise RuntimeError("This checkpoint adapter requires the deterministic base scheduler")
    return {
        "sigmas": scheduler.sigmas.cpu(),
        "timesteps": scheduler.timesteps.cpu(),
        "index": scheduler.step_index,
        "begin": scheduler.begin_index,
        "code": code_hash(type(scheduler).step),
    }


def run_step(self, components, block_state, i, t):
    schedulers = (components.scheduler, components.audio_scheduler)
    if i == 0:
        if len(block_state.timesteps) != 49:
            raise RuntimeError("Unexpected reference denoising schedule")
        for scheduler in schedulers:
            schedule_state(scheduler)
        print("H3_REFERENCE_CHECKPOINT_HOOK_ACTIVE", flush=True)

    if saved is not None:
        if i < stop_at:
            return components, block_state
        if i == stop_at:
            for scheduler, state in zip(schedulers, saved["schedules"], strict=True):
                current = schedule_state(scheduler)
                if any(not torch.equal(current[key], state[key]) for key in ("sigmas", "timesteps")):
                    raise RuntimeError("Checkpoint schedule mismatch")
                if current["code"] != state["code"] or current["begin"] != state["begin"] or state["index"] != i:
                    raise RuntimeError("Checkpoint scheduler version or index mismatch")
                scheduler._step_index = state["index"]
            for field in fields:
                target = getattr(block_state, field)
                source = saved[field]
                if (target.shape, target.dtype) != (source.shape, source.dtype):
                    raise RuntimeError(f"Checkpoint tensor mismatch: {field}")
                target.copy_(source)
            torch.set_rng_state(saved["cpu_rng"])
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
            print(f"H3_REFERENCE_RESUMED step={i}", flush=True)

    components, block_state = original(self, components, block_state, i=i, t=t)
    if saved is None and i + 1 == stop_at:
        data = {field: getattr(block_state, field).detach().cpu() for field in fields}
        data.update(
            schedules=[schedule_state(scheduler) for scheduler in schedulers],
            cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),
            loop=code_hash(MiniMaxH3DenoiseLoopWrapper.__call__),
            step=stop_at,
            identity=os.environ["H3_REFERENCE_IDENTITY"],
        )
        if any(state["index"] != stop_at for state in data["schedules"]):
            raise RuntimeError("Cannot save mismatched audio/video scheduler indices")
        temporary = checkpoint.with_suffix(".part")
        torch.save(data, temporary)
        temporary.replace(checkpoint)
        print(f"H3_REFERENCE_CHECKPOINT_SAVED step={stop_at}", flush=True)
        raise SystemExit(0)
    return components, block_state


if checkpoint.exists():
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        saved["step"] != stop_at
        or saved["loop"] != code_hash(MiniMaxH3DenoiseLoopWrapper.__call__)
        or saved["identity"] != os.environ["H3_REFERENCE_IDENTITY"]
    ):
        raise RuntimeError("Checkpoint does not match the original reference configuration")

MiniMaxH3DenoiseLoopWrapper.loop_step = run_step
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
