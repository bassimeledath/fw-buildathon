# Team plan (5 people, build 10:30 to 4:30, submit by 3:30)

One rule: **the test set locks at 11:00.** After that, nobody changes `data/test*` or the
reward. Every number we report comes from that locked set at temperature 0, with the same prompt.

The names are a suggestion; swap roles to fit skills. Each role owns one folder of results in
`runs/` and one slide.

| Role | Owner | Owns | Done when |
|---|---|---|---|
| **1. RL lead and integrator** | Bassim | Calibrate the train band with the base snapshot (k=8). GRPO runs plus a small knob sweep (rank 32/64, lr 1e-4/5e-5). Picks the final checkpoint. Final call on scope. | Best checkpoint chosen by 1:30, with its curve and the knobs tried, including the losers. |
| **2. Eval and stats** | | 10:30 to 11:00: generate a bigger test set (about 300 cases), then lock it. Base model and 2 frontier baselines at temperature 0 (`ase.episodes`, `ase.calibrate --k 1`). Base pass curve at k=8 (`ase.passk`), which gives the quadrant. `ase.stats` on every comparison. The two charts. | A leaderboard table plus the McNemar/bootstrap chart by 2:30. |
| **3. Harness baselines and routing** | | Harness A (strong prompt plus the "ask when results conflict" rule, with JSON-schema outputs), Harness B (A × 5 votes), Harness C (KB catalog in the prompt), all on the locked set. The routing bar (best per-category router). | Rows in the leaderboard with cost per case, losses included, by 1:30. |
| **4. Second technique and failure analysis** | | Rejection-sampling SFT. Sample the base 8 times on train states only, keep the full-credit draws, then serverless SFT (cookbook `sft_loop`). Read 20 wrong fixes from the base and the trained model, and write the "intrinsic failure" examples. | An RS-SFT checkpoint scored on the locked set. 3 before/after transcripts for the demo. |
| **5. Business case, cost and submission** | | The business case with sourced numbers and every assumption labeled. Deploy the best LoRA (on-demand, live merge) and measure p50 latency and cost per resolved case. Break-even against kimi-k3. README, disclosure, the 60 s video. Make the repo public and check it in an incognito window. | Submitted by 3:30. |

## Timeline
| Time | Everyone | Notes |
|---|---|---|
| 10:30 | Setup (README). Each person runs `ase.sanity` to confirm their env works. | Fireworks keys: one per person if possible, so rate limits don't collide. |
| 11:00 | **Test set locked** (role 2) | Roles 1 and 4 start sampling the base on *train* states only. |
| 11:30 | Baselines done (2). Harness A running (3). | |
| 12:00 | **GRPO launched** (1). RS-SFT data harvested (4). | Training runs in the background, about 40 s per step. |
| 1:30 | Checkpoints scored. Harness rows done. Deploy starts (5). | Freeze new experiments here. |
| 2:30 | Stats, charts and leaderboard final (2). Video recorded (5 and 4). | |
| 3:00 | Dry run of the 60 s pitch, in STORY.md order. | |
| 3:30 | **Submit** (5). Keep updating until 4:30 if needed. | |

## Hand-offs
- **2 → everyone at 11:00:** the locked `data/test_*` files plus the commit hash. Results must cite that hash.
- **1 → 2 and 5 at 1:30:** the best snapshot path (from `runs/<run>/result.json`).
- **3 → 2:** a `runs/` folder per harness, in the same format as `ase.episodes` / `ase.calibrate`, so `ase.stats` runs on it.
- **4 → 5:** 3 transcripts plus 1 sentence each, for the video.

## Cut list if behind
Drop, in this order: Harness C, then the knob sweep, then RS-SFT. **Never cut:** the locked test
set, the base vs trained stats, the cost per case, the video.
