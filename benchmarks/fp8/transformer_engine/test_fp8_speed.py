"""
Test to compare pure TE FP8 performance vs Accelerate's FP8 integration.

This script measures training speed to verify that Accelerate's FP8 integration
doesn't add overhead compared to using Transformer Engine directly.
"""

import time
import torch
import transformer_engine.common.recipe as te_recipe
import transformer_engine.pytorch as te
from fp8_utils import get_training_utilities, get_named_parameters
from transformer_engine.common.recipe import DelayedScaling
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import FP8RecipeKwargs, set_seed
from accelerate.utils.transformer_engine import convert_model

MODEL_NAME = "bert-base-cased"

def train_baseline():
    """Baseline: Pure TE with manual fp8_autocast."""
    print("=" * 60)
    print("BASELINE: Pure TE with manual fp8_autocast")
    print("=" * 60)
    set_seed(42)
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = get_training_utilities(MODEL_NAME)
    
    # Convert to TE
    old_named_params = get_named_parameters(model)
    with torch.no_grad():
        convert_model(model)
    new_named_params = get_named_parameters(model)
    mapping = {p: new_named_params[n] for n, p in old_named_params.items()}
    for param_group in optimizer.param_groups:
        param_group["params"] = [mapping[p] for p in param_group["params"]]
    
    FP8_RECIPE_KWARGS = {"fp8_format": te_recipe.Format.HYBRID, "amax_history_len": 32, "amax_compute_algo": "max"}
    fp8_recipe = DelayedScaling(**FP8_RECIPE_KWARGS)
    
    model.to("cuda")
    model.train()
    
    # Warmup
    print("Warming up...")
    for i, batch in enumerate(train_dataloader):
        if i >= 10:
            break
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                batch = {k: v.to("cuda") for k, v in batch.items()}
                outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Timed run
    print("Running timed benchmark...")
    torch.cuda.synchronize()
    start = time.time()
    steps = 0
    for batch in train_dataloader:
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                batch = {k: v.to("cuda") for k, v in batch.items()}
                outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        steps += 1
    torch.cuda.synchronize()
    baseline_time = time.time() - start
    
    print(f"Steps: {steps}")
    print(f"Total time: {baseline_time:.2f}s")
    print(f"Time per step: {baseline_time/steps:.3f}s")
    return baseline_time, steps

def train_integration():
    """Accelerate integration: Using Accelerate's FP8 support."""
    print("\n" + "=" * 60)
    print("ACCELERATE: Using Accelerate's FP8 integration")
    print("=" * 60)
    FP8_RECIPE_KWARGS = {"fp8_format": "HYBRID", "amax_history_len": 32, "amax_compute_algo": "max"}
    kwargs_handlers = [FP8RecipeKwargs(backend="TE", **FP8_RECIPE_KWARGS)]
    AcceleratorState()._reset_state(True)
    accelerator = Accelerator(mixed_precision="fp8", kwargs_handlers=kwargs_handlers)
    set_seed(42)
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = get_training_utilities(
        MODEL_NAME, accelerator=accelerator
    )
    
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    model.train()
    
    # Warmup
    print("Warming up...")
    for i, batch in enumerate(train_dataloader):
        if i >= 10:
            break
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
    
    # Timed run
    print("Running timed benchmark...")
    torch.cuda.synchronize()
    start = time.time()
    steps = 0
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        steps += 1
    torch.cuda.synchronize()
    accel_time = time.time() - start
    
    print(f"Steps: {steps}")
    print(f"Total time: {accel_time:.2f}s")
    print(f"Time per step: {accel_time/steps:.3f}s")
    return accel_time, steps

if __name__ == "__main__":
    baseline_time, baseline_steps = train_baseline()
    accel_time, accel_steps = train_integration()
    
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Baseline (Pure TE):  {baseline_time/baseline_steps:.3f}s per step")
    print(f"Accelerate:          {accel_time/accel_steps:.3f}s per step")
    
    if abs(baseline_time - accel_time) / baseline_time < 0.05:
        speedup_pct = abs(baseline_time / accel_time - 1) * 100
        print(f"Speedup:             {baseline_time/accel_time:.2f}x ({speedup_pct:.1f}% difference)")
        print("✅ Accelerate performance is equivalent to pure TE!")
    elif accel_time < baseline_time:
        speedup = baseline_time / accel_time
        print(f"Speedup:             {speedup:.2f}x")
        print(f"✅ Accelerate is {(speedup - 1)*100:.1f}% faster!")
    else:
        slowdown = (accel_time / baseline_time - 1) * 100
        print(f"Slowdown:            {slowdown:.1f}%")
        print(f"❌ Accelerate is {slowdown:.1f}% slower than pure TE!")
        print("\nThis indicates an issue with Accelerate's FP8 integration overhead.")
