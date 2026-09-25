print("=== SFT resume: recast loaded optimizer state to the parameters' training dtype ===")
print("Objective : Verify medvision_bm.sft.sft_utils.recast_optimizer_state")
print("            (1) floating-point moment tensors (exp_avg, exp_avg_sq) whose dtype differs from")
print("                the target are recast to it, on the owning parameter's device,")
print("            (2) `step` tensors are left untouched (fused AdamW needs them fp32 as loaded),")
print("            (3) state already in the target dtype is left as-is and the count reports 0,")
print("            (4) the return value counts exactly the tensors that were recast.")
print("Background: with FSDP use_orig_params + accelerate's fp32 upcast, Trainer loads optimizer")
print("            state while the optimizer still holds stale low-precision parameter views, so")
print("            Optimizer.load_state_dict casts the moments to bf16; the first step then fails in")
print("            fused AdamW with 'Tensors of the same index must be on the same device and dtype'.")
print("Data      : tiny torch model on CPU; no FSDP, no GPU required.")

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path("src").resolve()))

import torch

from medvision_bm.sft.sft_utils import recast_optimizer_state  # noqa: E402

torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=False)

# Build a state the way a stale-view load would leave it: moments in bf16, step fp32 scalar.
for p in model.parameters():
    opt.state[p] = {
        "step": torch.tensor(1000.0, dtype=torch.float32),
        "exp_avg": torch.zeros_like(p, dtype=torch.bfloat16),
        "exp_avg_sq": torch.zeros_like(p, dtype=torch.bfloat16),
    }
n_params = sum(1 for _ in model.parameters())

# (1)+(4): recast to fp32 -> 2 moment tensors per parameter recast
n = recast_optimizer_state(opt, torch.float32)
assert n == 2 * n_params, n
for p in model.parameters():
    st = opt.state[p]
    assert st["exp_avg"].dtype == torch.float32 and st["exp_avg_sq"].dtype == torch.float32
    assert st["exp_avg"].device == p.device and st["exp_avg_sq"].device == p.device
    assert st["exp_avg"].shape == p.shape
print("[PASS] (1)(4) bf16 moments recast to fp32 on the parameter's device; count = 2 x n_params")

# (2): step untouched
for p in model.parameters():
    assert opt.state[p]["step"].dtype == torch.float32 and opt.state[p]["step"].item() == 1000.0
print("[PASS] (2) step tensors untouched")

# (3): idempotent
assert recast_optimizer_state(opt, torch.float32) == 0
print("[PASS] (3) already-matching state is left alone (0 recast)")

# The state must now be usable by a real AdamW step without a dtype error.
loss = model(torch.randn(5, 4)).pow(2).mean()
loss.backward()
opt.step()
print("[PASS] optimizer.step() runs on the recast state")

print("\nALL PASS")
