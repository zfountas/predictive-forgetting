# Sample KV Cache Data

Place pre-computed KV cache log files here for Figure 5 plotting.

Expected format: PyTorch tensor files named `log_<i>` where each file contains
a list of steps, each step being a list of layer dictionaries with keys:
- `k_cache`: [B, H, S, D] tensor
- `k_update`: [B, H, S, D] tensor
- `v_cache`: [B, H, S, D] tensor
- `v_update`: [B, H, S, D] tensor
- `mask`: [B, S] tensor

For the bootstrap analysis (grand average plot), place per-log CSV files in
`../stats_cache/` with columns: `layer`, `ratio`, `ci_width`.
