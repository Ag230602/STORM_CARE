# STORM-CARE-FM — AAAI 2027 Experiment Code Bundle

Companion code to `AAAI2027_experiment_plan.md`. Written against the audit
reports and README only — I could not see the repository source, so each file
is honest about its status:

| File | Plan item | Status |
|---|---|---|
| `scripts/compute_significance.py` | E4 | **Runs as-is** (pandas/numpy/scipy). Storm-level bootstrap CIs + Wilcoxon + Holm; emits the "claimable" flag the manuscript rule uses. |
| `scripts/check_dose_response.py` | E3 | **Runs as-is** given sweep outputs. Point + bootstrap monotonicity verdict; prints the abstract go/no-go. |
| `scripts/grep_orphaned_numbers.py` | sync checklist | **Runs as-is.** Exit-code-1 gate for dead numbers (97.7%, ep 8, +29.9, stale splits, …). Wire into a `make submission` target. |
| `scripts/run_physics_weight_sweep.py` | E5 | **Near-runnable**: shells out to `model.physics.train` like the audited commands. Needs a `--physics-weight-scale` flag added to the trainer (one line in config + argparse) and column-name confirmation. `--dry-run` prints commands. |
| `scripts/audit_test_coverage.py` | E1 step 1 | **Scaffold**: 2 adapter functions (`load_test_storm_ids`, `enumerate_windows`) must call the repo's split file and the *same* ERA5-completeness predicate the corrected benchmark uses. Refactor that predicate into a shared function — do not duplicate it. |
| `scripts/run_full_test_benchmark.py` | E1 steps 2–4 | **Scaffold**: 3 adapters (`build_datasets`, `train_model`, `evaluate_model`) call existing corrected trainers/eval. Orchestration, frozen-fallback mode, tuning-budget logging, and the output contract for E4 are done. |
| `model/counterfactual/scenarios_dose_response.py` | E3 | **Template**: scenario table + ramp reference implementation + `register()` hook. Must be wired to the real scenario registry and the real exposure channel slice. Includes the timing-only control that defends against the "same magnitude, different label" circularity. |

## Order of operations

1. Drop files into the repo preserving paths.
2. **Day 1, cheap:** wire `scenarios_dose_response.py` (verify the three
   operators inject different perturbations!), rerun
   `model.counterfactual.run`, emit a per-sequence outcomes CSV (add ~3 lines
   in `run.py` to write per-sequence rows before averaging), then
   `python scripts/check_dose_response.py --input <per_sequence.csv>`.
   This decides the abstract's dose–response sentence.
3. Wire the two E1 adapters in `audit_test_coverage.py`; run it to learn the
   real benchmark size before committing GPU time.
4. Start E2 (no new code: `model.foundation.pretrain` without `--demo`, on
   GPU, after fixing the IRMA ERA5 tag match) in parallel.
5. Wire `run_full_test_benchmark.py` adapters; run `--mode frozen` first so
   there is always a complete result, then `--mode full` if time allows.
6. Pipe every comparison CSV through `compute_significance.py`.
7. Rerun `scripts/run_ablations.py` (E6, existing repo script) after E1/E2.
8. Before every manuscript freeze: `python scripts/grep_orphaned_numbers.py`.

## Contracts the scaffolds enforce (please keep)

- The benchmark refuses to run without the coverage manifest, so coverage
  and evaluation can never disagree.
- Checkpoint selection is validation-only; the test set is touched once.
- `--mode frozen` configs must be pre-registered (copy the known-good
  Irma/Ian rerun configs), so the compute fallback is not silent tuning.
- Comparison sentences in the paper require `claimable_at_0.05 == True`.
- Numbers flow script → metrics/ → tables/ → sync_manuscript.py; nothing
  is hand-edited.
