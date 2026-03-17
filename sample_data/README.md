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

**Data availability:** The pre-computed KV cache data (100 example logs) is
publicly available on Figshare:

https://doi.org/10.6084/m9.figshare.31534807

The full dataset (~60GB, N=1318 logs) is available upon reasonable request
to the corresponding author (zafeirios.fountas@huawei.com).
