# Documentation Audit Log

Evidence and justification for every change applied during the Phase 1 audit (2026-02-11).

---

## Fix A1: README.md — Remove stale issue links

**File:** `README.md` line 38
**Change:** `"Open design issues: [#7](../../issues/7)–[#14](../../issues/14)."` → `"Research phase — documentation and architecture specification complete. Implementation not yet started."`
**Evidence:** `gh issue list --state closed` confirms all issues #7–#14 are CLOSED. No open issues remain.
**Justification:** Stale links to closed issues mislead readers about project status.

---

## Fix F1: HYDRA_SPEC.md — Remove 7 "moved to" stubs

**File:** `HYDRA_SPEC.md`
**Change:** Removed sections "Training Pipeline", "Loss Functions", "PID-Lagrangian λ Auto-Tuning", "Failure Modes & Mitigations", "Monitoring Metrics", "Implementation Roadmap", "Open Questions" — each was a 2-line stub saying "> This section has been moved to [TRAINING.md](...)"
**Evidence:** Each stub's target content exists in TRAINING.md. The "Related Documents" section at the top of HYDRA_SPEC.md already cross-references TRAINING.md.
**Justification:** Redundant navigational noise. Readers already have the cross-reference table; stub sections add no information and clutter the document outline.

---

## Fix H3/E1: HYDRA_SPEC.md — Training VRAM estimate

**File:** `HYDRA_SPEC.md` line 21
**Change:** `"<20GB active"` → `"<4GB active"`
**Evidence:** INFRASTRUCTURE.md lines 414–421 contain the itemized VRAM budget:
- Model (bf16): ~33–34 MB
- Optimizer state (AdamW, bf16): ~130 MB
- Opponent cache (5 × 33 MB): ~165 MB
- PPO minibatch (on-GPU): ~200–400 MB
- PPO rollout buffer: **CPU pinned memory**, NOT VRAM (lines 421, 433, 658)
- Total: ~3.7 GB (line 419), "never exceeds 4 GB" (line 761)
**Justification:** The old "<20GB" was based on a stale estimate that assumed the PPO rollout buffer lived in VRAM. The architecture explicitly uses CPU pinned memory with async transfers.

---

## Fix D1/D2/D3: MORTAL_ANALYSIS.md — Remove stub sections

**File:** `MORTAL_ANALYSIS.md`
**Change:** Removed "3-Player (Sanma) Adaptations" (stub cross-ref to ECOSYSTEM.md), "Ecosystem" (stub cross-ref to REFERENCES.md/ECOSYSTEM.md), and "Integration Projects" table (Akagi, Riki, kanachan — already covered in ECOSYSTEM.md with more detail).
**Evidence:** ECOSYSTEM.md contains all integration projects with license, star count, and Hydra relevance. MORTAL_ANALYSIS.md's table was a subset with less information.
**Justification:** Duplicate content. MORTAL_ANALYSIS.md should focus on Mortal itself, not the broader ecosystem.

---

## Fix D4: COMMUNITY_INSIGHTS.md — Remove duplicated Mortal architecture deep dive

**File:** `COMMUNITY_INSIGHTS.md` §10 (old numbering §9)
**Change:** Replaced full Mermaid architecture diagram + version evolution table + distributed training details + 1v3 evaluation protocol (~40 lines) with a 1-line cross-reference: `> See [MORTAL_ANALYSIS.md](MORTAL_ANALYSIS.md) for the full architecture analysis...`
**Evidence:** Every piece of content (DQN head v1–v4, training loss components, distributed protocol, 1v3 duplicate evaluation) exists in MORTAL_ANALYSIS.md §1 with source citations to specific commit `0cff2b5`.
**Justification:** Verbatim duplication. Two copies = two places to get out of sync.

---

## Fix F2: ECOSYSTEM.md — Remove empty "Data Pipeline Architecture" section

**File:** `ECOSYSTEM.md` §6
**Change:** Deleted 4-line section that contained only: `## 6. Data Pipeline Architecture` + `> See [INFRASTRUCTURE.md § Data Pipeline](...) and [INFRASTRUCTURE.md § System Architecture](...) for detailed pipeline diagrams.`
**Evidence:** Zero substantive content beyond a cross-reference link.
**Justification:** A section with no content beyond a link is navigational noise. The cross-reference table at the top of relevant docs already covers this.

---

## Fix F3: REFERENCES.md — Remove "Technical References" section

**File:** `REFERENCES.md`
**Change:** Removed ~25-line section listing generic links (PyTorch docs, torch.compile tutorial, AMP docs, Rust Book, PyO3 docs, Rayon docs, W&B, MLflow).
**Evidence:** All links are top-level URLs (e.g., `https://pytorch.org/docs/`, `https://doc.rust-lang.org/book/`) that any developer would find via Google in seconds.
**Justification:** Low signal-to-noise. REFERENCES.md should cite domain-specific sources relevant to Hydra, not generic toolchain documentation.

---

## Fix G-Mortal-1: MORTAL_ANALYSIS.md — consts.rs line reference

**File:** `MORTAL_ANALYSIS.md` §1.1
**Change:** `consts.rs:15` → `consts.rs:25`
**Evidence:** Librarian agent verified at commit `0cff2b52982be5b1163aa9a62fb01f03ce91e0d2`:
- Line 15 is the comment `// = 46` (end of ACTION_SPACE calculation)
- Line 25 is the match arm `4 => (1012, 34)` inside `obs_shape()`
Source: https://github.com/Equim-chan/Mortal/blob/0cff2b52982be5b1163aa9a62fb01f03ce91e0d2/libriichi/src/consts.rs#L25
**Justification:** Wrong line number would send readers to wrong code.

---

## Fix G-Mortal-2: MORTAL_ANALYSIS.md — train.py line reference

**File:** `MORTAL_ANALYSIS.md` §1.2
**Change:** `train.py:237` → `train.py:236-238`
**Evidence:** Librarian agent verified at commit `0cff2b5`:
- L236: `cql_loss = 0`
- L237: `if not online:`
- L238: `cql_loss = q_out.logsumexp(-1).mean() - q.mean()`
The CQL formula spans all three lines (init + guard + computation). Citing only L237 (the `if` guard) misses the actual formula.
Source: https://github.com/Equim-chan/Mortal/blob/0cff2b52982be5b1163aa9a62fb01f03ce91e0d2/mortal/train.py#L236-L238
**Justification:** Incomplete line reference.

---

## Fix URL: COMMUNITY_INSIGHTS.md — modern-jan.com URLs (4 occurrences)

**File:** `COMMUNITY_INSIGHTS.md` lines 123, 124, 147
**Change:** `modern-jan.com/2023/09/06/luckyj_article_ja/` → `modern-jan.com/blog/luckyj_article_ja/` (4 occurrences)
**Evidence:** HTTP 302 redirect confirmed: the old date-based WordPress URL redirects to the new `/blog/` path. The destination page loads with the correct LuckyJ article content.
**Justification:** Use canonical URL to avoid depending on redirect permanence.

---

## Fix Suphx-1: HYDRA_SPEC.md — Block count rationale

**File:** `HYDRA_SPEC.md` line 140 (Key Design Choices table)
**Change:** `"Matches Suphx depth, proven sufficient for Mahjong complexity"` → `"Suphx uses 50 blocks; 40 balances depth with parameter budget for Hydra's 256ch width"`
**Evidence:** Suphx paper (arXiv:2003.13590) Figures 4-5 explicitly annotate "Repeat 50x" on the residual block. Suphx uses 50 blocks with 256 filters; Hydra uses 40 blocks with 256 channels. The old claim "matches Suphx depth" was false — 40 ≠ 50.
**Justification:** Factual error. Hydra intentionally uses fewer blocks; the rationale should explain why.

---

## Fix Suphx-2: REFERENCES.md — Suphx architecture description

**File:** `REFERENCES.md` line 13
**Change:** `"~31-layer CNN (15 residual blocks), 134 channels"` → `"50 residual blocks, 256 filters, separate models per action type with 838 input channels (discard/riichi) and 958 input channels (chow/pong/kong) (Table 2, Figures 4-5)"`
**Evidence:** Suphx paper (arXiv:2003.13590):
- Figures 4-5: "Repeat 50x" on residual blocks, "3x1 conv 256" throughout
- Table 2: Input dimensions are 838×34×1 (discard/riichi) and 958×34×1 (chow/pong/kong)
- "134" does NOT appear in the architecture description — it has no basis in the paper
**Justification:** The old description was completely wrong on all three counts (blocks, channels, input features).

---

## Fix Suphx-3: TRAINING.md — Oracle guiding contribution

**File:** `TRAINING.md` line 187
**Change:** `"~0.12 dan"` → `"~0.06 dan"` for oracle contribution; `"~0.71 dan"` → `"~0.63 dan"` for full pipeline; baseline numbers corrected; added caveat about offline vs online evaluation
**Evidence:** Suphx paper (arXiv:2003.13590) Figure 8 box plot median readings:
- SL baseline: ~7.66 stable dan
- RL-1 (SL+RL+GRP): ~8.23
- RL-2 (SL+RL+GRP+Oracle): ~8.29
- Oracle contribution = RL-2 − RL-1 ≈ 0.06 dan
- Full pipeline = RL-2 − SL ≈ 0.63 dan
Note: Figure 8 is offline evaluation against weaker opponents, not online Tenhou play. Final online system reached 8.74 dan (Table 4).
**Justification:** The old values (~0.12, ~0.71) appear to have been misread from the box plot or conflated with other comparisons.

---

## Fix G1: TRAINING.md — PPO forgetting claim

**File:** `TRAINING.md` line 258
**Change:** `"This avoids the catastrophic forgetting that Mortal experiences"` → Nuanced version acknowledging PPO self-play can also suffer forgetting from distributional shift, mitigated by league opponent pool and KL anchoring.
**Evidence:** COMMUNITY_INSIGHTS.md §7 (Self-Play Training Best Practices) and Mortal Discussion #64 both document PPO catastrophic forgetting in self-play. The old claim was oversimplified — PPO avoids *replay buffer staleness* but not *distributional shift*.
**Justification:** Misleading oversimplification. PPO self-play is not immune to catastrophic forgetting.

---

## Fix B1: CHECKPOINTING.md — Opponent pool cross-reference

**File:** `CHECKPOINTING.md` line 183
**Change:** `"canonical definition in [INFRASTRUCTURE.md § Phase 3](...)"` → `"canonical definition in [TRAINING.md § Phase 3](...)"`
**Evidence:** TRAINING.md §Phase 3 (lines 216–240) contains the authoritative opponent pool weights (50% latest, 30% best, 20% BC anchor). INFRASTRUCTURE.md §Phase 3 (line 660) contains implementation details (GPU cache, FIFO eviction) but explicitly defers to TRAINING.md for composition weights (line 660: "composition weights defined in [TRAINING.md § Phase 3]").
**Justification:** Cross-reference pointed to the wrong document. TRAINING.md is the single source of truth for training parameters.

---

## Fix INFRA-1: INFRASTRUCTURE.md — VRAM budget table

**File:** `INFRASTRUCTURE.md` lines 413–421
**Change:** Replaced old table (which showed ~77 GB PPO rollout in VRAM) with corrected breakdown showing CPU pinned memory for rollout buffer, ~3.7 GB total VRAM.
**Evidence:** Same document, lines 658 (double-buffered rollout with pinned memory), line 643 (Mermaid diagram labels rollout as "Memory (Pinned)"), line 694 (~3.7 GB total), line 761 ("never exceeds 4 GB"). Internal consistency verified across all four independent references.
**Justification:** The old estimate incorrectly placed the full rollout buffer in VRAM.

---

## Fix INFRA-2: INFRASTRUCTURE.md — PPO hyperparameters deduplication

**File:** `INFRASTRUCTURE.md` lines 676–690
**Change:** Removed duplicate PPO hyperparameter table (12 rows), replaced with cross-reference to TRAINING.md § Phase 3 as the authoritative source.
**Evidence:** TRAINING.md lines 245–256 contain the complete PPO hyperparameter table with all parameters (clip ε, entropy coef, GAE λ, γ, value clip, LR, minibatch size, update epochs, gradient clip, init). INFRASTRUCTURE.md now says "See TRAINING.md § Phase 3 for the authoritative PPO hyperparameter table" (line 678).
**Justification:** Single Source of Truth. Two copies of the same table = two places to diverge.

---

## Fix INFRA-3: INFRASTRUCTURE.md — pyo3 version

**File:** `INFRASTRUCTURE.md` line 72
**Change:** `pyo3 | 0.22+` → `pyo3 | 0.28+`
**Evidence:** PyO3 v0.28.0 released 01 Feb 2025 (GitHub releases page). Latest stable series as of Feb 2026. 0.22 is two major versions behind.
**Justification:** Stale version pin for a project that hasn't started implementation.

---

## Fix MISC: Section renumbering in COMMUNITY_INSIGHTS.md

**File:** `COMMUNITY_INSIGHTS.md`
**Change:** Renumbered sections after LuckyJ from §4→§5 through §11→§12, and §12→§13, to account for LuckyJ being §4 (was previously unnumbered).
**Evidence:** LuckyJ section header was `## LuckyJ (Tencent AI Lab)` with no number; all subsequent sections had numbers. Added `## 4.` prefix and incremented subsequent sections.
**Justification:** Consistent numbering throughout the document.

---

## Fix MISC: Volatility note in COMMUNITY_INSIGHTS.md

**File:** `COMMUNITY_INSIGHTS.md` line 4
**Change:** Added source volatility note: `> **Source volatility note:** Several references link to personal blogs (note.com, hatenablog, Ghost, nicovideo blomaga, modern-jan.com) that may go offline. All critical data points (statistics, architecture details, p-values) are reproduced inline so this document remains self-contained even if external links rot. Last verified: 2026-02-11.`
**Evidence:** The document cites 8+ personal blog/platform URLs that could disappear. All critical data points (LuckyJ p-values, NAGA stats, danger multipliers) are already reproduced inline in tables.
**Justification:** Defensive documentation. Acknowledges link rot risk and confirms self-containment.

---

## Fix MISC: Latency table caveat in HYDRA_SPEC.md

**File:** `HYDRA_SPEC.md` lines 467–470
**Change:** Added `(Estimated Targets)` to heading, added note: `> **Note:** These are design targets based on comparable architectures (Mortal, Suphx), not measured values. Actual benchmarks will be established during Milestone 2.` Changed column headers from `RTX 3070` / `RTX 4090` to `RTX 3070 (est.)` / `RTX 4090 (est.)`.
**Evidence:** No implementation exists — no benchmarks could have been measured. Values are projections.
**Justification:** Presenting estimates as measured values is misleading. Readers need to know these are targets.

---

## Fix MISC: ABLATION_PLAN.md — Phase 1 readiness gate reference

**File:** `ABLATION_PLAN.md` line 32
**Change:** Added `SL loss plateaued` to the Phase 1 readiness gate criteria list, and added cross-reference to INFRASTRUCTURE.md § Phase 1.
**Evidence:** INFRASTRUCTURE.md § Phase 1 readiness gate (line ~510) includes "SL loss plateaued" as a criterion alongside the metrics listed.
**Justification:** Incomplete criteria list could cause ablation baseline to be selected prematurely.

---

## Fix MISC: OPPONENT_MODELING.md — Remove stale cross-ref

**File:** `OPPONENT_MODELING.md`
**Change:** Removed `> See [HYDRA_SPEC.md § Safety Channels](HYDRA_SPEC.md#safety-channels-6183) for the channel-level summary diagram.`
**Evidence:** HYDRA_SPEC.md does not contain a "Safety Channels (61-83)" section or anchor. The anchor `#safety-channels-6183` is broken.
**Justification:** Broken cross-reference.

---

## Fix MISC: ECOSYSTEM.md — Star count column removal

**File:** `ECOSYSTEM.md` § Synthetic Data table
**Change:** Removed "Stars" column from the Synthetic Data table (mjx 202, mahjax 22, mjai.app 109).
**Evidence:** Star counts are volatile — they change daily and were already stale.
**Justification:** Volatile data in a reference document. License, speed, and notes columns provide the information readers need.

---

## Fix MISC: REFERENCES.md — Training Data Sources dedup

**File:** `REFERENCES.md`
**Change:** Replaced 7-row Training Data Sources table with cross-reference to ECOSYSTEM.md § Data Sources and archive/DATA_SOURCES.md.
**Evidence:** ECOSYSTEM.md § Data Sources & Datasets already contains the same data with more detail. archive/DATA_SOURCES.md has full converter/tool information.
**Justification:** Duplicate content across three locations.

---

## Fix MISC: REFERENCES.md — Mortal Discussion #3 correction

**File:** `REFERENCES.md` line 220
**Change:** `Discussion #3` → `(source code)` — the MC-returns-vs-TD insight was confirmed from source code, not from Discussion #3 (which is actually a PR about torch.autocast keyword arguments).
**Evidence:** GitHub PR #3 is titled "Keyword argument in torch.autocast()" — it's a code fix, not a discussion about Q-targets. The MC returns insight comes from `train.py` Q-target computation.
**Justification:** Wrong citation. PR #3 has nothing to do with MC returns.

---

## Fix MISC: MAJSOUL_INTEGRATION.md — Star count column removal

**File:** `archive/MAJSOUL_INTEGRATION.md`
**Change:** Removed "Stars" column from integration tables.
**Evidence:** Same rationale as ECOSYSTEM.md — star counts are volatile.
**Justification:** Volatile data in archived reference document.

---

## Correction applied during audit: LuckyJ p-value precision

**File:** `COMMUNITY_INSIGHTS.md` line 123
**Change:** `p=0.029` → `p=0.02883`
**Evidence:** modern-jan.com LuckyJ article reports the exact p-value as 0.02883. The 0.029 rounding loses precision unnecessarily.
**Justification:** Exact value available; use it.

---

## Claims verified as CORRECT (no change needed)

| Claim | Evidence | Source |
|-------|----------|--------|
| Mortal commit `0cff2b5` exists | GitHub API confirms | https://github.com/Equim-chan/Mortal/commit/0cff2b5 |
| `model.py:L10-28` (SE attention) | Verified at pinned commit | Librarian agent (Mortal source session) |
| `model.py:L233-249` (GRP 24-way) | Verified at pinned commit | Librarian agent (Mortal source session) |
| `obs_repr.rs:L149-164` (dual-scale scores) | Verified at pinned commit | Librarian agent (Mortal source session) |
| `obs_repr.rs:L451/L457` (shanten discards) | Verified at pinned commit | Librarian agent (Mortal source session) |
| `invisible.rs:L152-245` (oracle encoding) | Verified at pinned commit | Librarian agent (Mortal source session) |
| NAGA: 4 CNNs, confidence estimation | DMV article confirms | https://dmv.nico/en/articles/mahjong_ai_naga/ |
| NAGA: guided backprop, 5 variants | DMV article confirms | https://dmv.nico/en/articles/mahjong_ai_naga/ |
| NAGA: ~9-dan stable | DMV article confirms | https://dmv.nico/en/articles/mahjong_ai_naga/ |
| Kanachan: 12L/768d and 24L/1024d | Config source confirms | https://github.com/Cryolite/kanachan/.../config.py |
| LuckyJ: 1,321 games to 10-dan | TechNode + haobofu.github.io | Multiple sources |
| LuckyJ: 10.68 stable dan | TechNode + haobofu.github.io | Multiple sources |
| LuckyJ vs NAGA: p=0.00003 | modern-jan.com | Exact match |
| RVR authors: Li, Wu, Fu, Fu, Zhao, Xing | Semantic Scholar | Confirmed |
| RVR: no specific speedup multiplier | Paper abstract | Confirmed |
| ACH: ICLR 2022, correct link | OpenReview | Confirmed |
| OLSS: ICML 2023, correct PMLR link | DBLP + PMLR | Confirmed |
| All external URLs accessible | Fetched successfully | dmv.nico, IEEE CoG, Phoenix paper, note.com, hatenablog |

---

## CRITICAL FIX: Mortal "10-dan Tenhou" fabrication (applied during audit)

**Files:** `README.md` line 8, `HYDRA_SPEC.md` line 22, `REFERENCES.md` line 203
**Change:**
- README.md: `"despite 10-dan Tenhou rating"` → removed entirely
- HYDRA_SPEC.md: `"Tenhou 10-dan+"` → `"Surpass community-estimated ~7-dan play strength"`
- REFERENCES.md: `"| Mortal | Tenhou | 10-dan | 2023 |"` → `"| Mortal | — | **No ranked play** | — |"` with citation to mjai-reviewer FAQ
**Evidence:** mjai-reviewer FAQ (https://github.com/Equim-chan/mjai-reviewer/blob/master/faq.md) — Equim-chan (Mortal's developer) states:
> "Tenhou rejected my AI account request for Mortal because Mortal was developed by an individual rather than a company."
> "I am not running any AI in ranked lobbies and will not do so until an official permission is granted."
Mortal README contains zero mentions of Tenhou ranked play.
**Justification:** Mortal has NEVER played ranked on Tenhou. The "10-dan" rating attributed to Mortal was fabricated — it has no basis in any source. The developer himself confirms the rejection.

---

## Fix: NAGA "26,598 games" flagged as unverifiable (applied during audit)

**Files:** `REFERENCES.md` line 58 and 200, `COMMUNITY_INSIGHTS.md` lines 59 and 114
**Change:** Added `"(source unverified — this number does not appear in the DMV article or any locatable public source)"` to all occurrences of 26,598. Removed comparison `"vs Suphx's 5,373 and NAGA's 26,598"` from LuckyJ section since NAGA number is unverified.
**Evidence:** DMV article (https://dmv.nico/en/articles/mahjong_ai_naga/) — full text fetched. Contains NO mention of 26,598 or any specific game count. Web search for "26,598" + "NAGA" + "mahjong" across DuckDuckGo and Bing returned zero results.
**Justification:** Unverifiable claim should be flagged, not presented as fact.

---

## Correction applied during audit: Suphx input channels

**File:** `REFERENCES.md` line 13
**Change:** `"826-843 input channels"` → `"838 input channels (discard/riichi) and 958 input channels (chow/pong/kong)"`
**Evidence:** Suphx paper (arXiv:2003.13590) Table 2: Input dimensions are explicitly 838×34×1 (discard/riichi models) and 958×34×1 (chow/pong/kong models). The 120-channel difference (958-838) encodes which tiles can form a call. The previous "826-843" was an approximation error.
**Justification:** Exact values available from Table 2; use them.

---

## Correction applied during audit: Star count precision reverted

**Files:** `REFERENCES.md` lines 52, 65
**Change:** `"1K+"` reverted to `"1.3K+"` (Mortal, 1,341 stars) and `"1.1K+"` (mjai-reviewer, 1,172 stars)
**Evidence:** GitHub API: Mortal stargazerCount=1,341, mjai-reviewer stargazerCount=1,172.
**Justification:** "1K+" understates both repos. The original values were more accurate.

---

## Fix: Suphx oracle dropout method misattribution (applied during audit)

**Files:** `TRAINING.md` line 171, `INFRASTRUCTURE.md` line 607, `OPPONENT_MODELING.md` line 385
**Change:** `"group-level scalar multiplication (following Suphx, arXiv:2003.13590 Section 3.3)"` → `"group-level deterministic scaling"` with a design note clarifying that Suphx uses a different method.
**Evidence:** Suphx paper (arXiv:2003.13590) Section 3.3, Equation 5:
> "δₜ is the dropout matrix at the t-th iteration whose elements are Bernoulli variables with P(δₜ(i,j) = 1) = γₜ. We gradually decay γₜ from 1 to 0."
Suphx uses **element-wise Bernoulli dropout** (each feature independently zeroed with probability 1-γₜ). Hydra uses **deterministic group-level scalar multiplication** (two groups × one scalar each, decaying 1.0→0.0). These are fundamentally different:
- Suphx: stochastic, binary {0,1} per element, single γₜ for all oracle features
- Hydra: deterministic, continuous [0,1] per group, separate schedules for opponent/wall
**Justification:** Misattribution — claiming to "follow Suphx" when using a different mechanism is misleading. Hydra's approach is a valid design choice but should not be attributed to Suphx.
