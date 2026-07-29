from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsfm_fairness_audit.sen1_supervised_campaign import (  # noqa: E402
    _finish_scaled_optimizer_step,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Small CUDA-only gate for the Sen1 AMP overflow recovery contract. "
            "This does not read data or launch a campaign."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the AMP overflow runtime gate.")
    device = torch.device("cuda")
    model = torch.nn.Linear(4, 1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=128.0,
        growth_interval=10_000,
        enabled=True,
    )
    inputs = torch.ones((2, 4), device=device)
    targets = torch.zeros((2, 1), device=device)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
    scale_before = float(scaler.get_scale())
    scaler.scale(loss).backward()
    first_parameter_name, first_parameter = next(model.named_parameters())
    first_parameter.grad.view(-1)[0] = float("inf")
    before = [parameter.detach().clone() for parameter in model.parameters()]
    overflow = _finish_scaled_optimizer_step(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        amp=True,
        scale_before=scale_before,
        mode="S1",
        seed=101,
        epoch=5,
        batch_index=0,
        training_stage="cuda_runtime_gate",
        sample_ids=["synthetic-runtime-gate"],
    )
    after_overflow = [
        parameter.detach().clone() for parameter in model.parameters()
    ]
    if not overflow["amp_overflow"] or not overflow["optimizer_step_skipped"]:
        raise RuntimeError("GradScaler did not report the injected overflow.")
    if not all(
        torch.equal(previous, current)
        for previous, current in zip(before, after_overflow)
    ):
        raise RuntimeError("Optimizer parameters changed on a skipped AMP step.")
    if first_parameter_name not in overflow["amp_overflow_record"][
        "overflow_parameter_names"
    ]:
        raise RuntimeError("Overflow parameter name was not recorded.")

    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        next_loss = torch.nn.functional.mse_loss(model(inputs), targets)
    next_scale_before = float(scaler.get_scale())
    scaler.scale(next_loss).backward()
    finite = _finish_scaled_optimizer_step(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        amp=True,
        scale_before=next_scale_before,
        mode="S1",
        seed=101,
        epoch=5,
        batch_index=1,
        training_stage="cuda_runtime_gate",
        sample_ids=["synthetic-runtime-gate"],
    )
    if finite["amp_overflow"] or not any(
        not torch.equal(previous, current)
        for previous, current in zip(after_overflow, model.parameters())
    ):
        raise RuntimeError("A normal batch did not update after overflow recovery.")
    payload = {
        "schema": "geobwer.sen1floods11.amp_cuda_gate.v1",
        "status": "pass",
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "device": str(device),
        "overflow_record": overflow["amp_overflow_record"],
        "parameters_unchanged_on_overflow": True,
        "next_finite_batch_updated_parameters": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("SEN1_AMP_OVERFLOW_CUDA_GATE=PASS")
    print(f"OUTPUT={args.output}")


if __name__ == "__main__":
    main()
