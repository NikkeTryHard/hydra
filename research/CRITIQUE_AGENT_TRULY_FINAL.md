# TRULY FINAL Critique Review: Sign-Off or Kill

## CONTEXT
Design: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md
This design has been through 5 review rounds. All known bugs are fixed. This is the LAST review.

## YOUR JOB
Read the design. Answer these 5 questions. If ALL answers are satisfactory, say "SIGN OFF." If any answer is "this will fail," say "KILL" and explain why.

## Q1: Is the budget internally consistent?
Phases: -1(150h) + 0(50h) + 1(200h) + 2(800h) + 3(800h) = 2000h.
- Does each phase's game count follow from its wall-hours and throughput?
- Is 150h enough for Phase -1 benchmarks?
- Is 800h enough for Phase 3 ExIt to matter (given teacher only processes 1-5% of states)?

## Q2: Does DRDA-wrapped ACH actually work as described?
The policy is: pi = softmax(l_base + y_theta / tau_drda)
ACH trains y_theta with gate c and global eta.
- Does this composition make mathematical sense?
- When base is updated (every 25-50h), does the Hedge accumulator in y_theta need resetting?
- What happens if tau_drda is set wrong? What's the safe range?

## Q3: Is the MVP viable?
If we only build: ACH + Oracle + CT-SMC + AFBS + Hand-EV (5 components):
- What dan level do you estimate?
- Is this enough to beat Mortal?
- What's the simplest possible system that reaches 9+ dan?

## Q4: What's the #1 thing that will go wrong during implementation?
Not theory. Not math. The actual engineering nightmare. What breaks first?

## Q5: Final win probability against LuckyJ (10.68 dan)?
Previous estimates ranged 30-70%. Given the current design with all fixes, Phase -1 reserve, DRDA-wrapped ACH, 3-tier nets, and honest budget:
- Your number (0-100%)
- One sentence justification
- What single thing would move this number up 10 percentage points?

## REFERENCES
- Design: https://raw.githubusercontent.com/NikkeTryHard/hydra/master/research/design/HYDRA_FINAL.md
- ACH: https://openreview.net/pdf?id=DTXZqTNV5nW
- OLSS: https://proceedings.mlr.press/v202/liu23k/liu23k.pdf
- DRDA: https://proceedings.iclr.cc/paper_files/paper/2025/file/1b3ceb8a495a63ced4a48f8429ccdcd8-Paper-Conference.pdf
- Suphx: https://arxiv.org/pdf/2003.13590
- KataGo: https://arxiv.org/pdf/1902.10565
- Repo: https://github.com/NikkeTryHard/hydra
