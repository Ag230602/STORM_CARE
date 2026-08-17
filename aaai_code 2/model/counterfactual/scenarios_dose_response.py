"""E3 — Evacuation dose-response scenario definitions.

TEMPLATE: adapt to the actual scenario registry in
model/counterfactual/scenarios.py. I could not see that file, so this module
defines the scenarios in a neutral dict form plus a register() hook; wire it
into however scenarios.py exposes its registry (dict, dataclass list, or
factory functions).

DESIGN REQUIREMENTS (from the audit + the manuscript's margin note):
  1. Each lead-time variant must inject a genuinely DIFFERENT perturbation
     into the warm-up state history — different timing and/or magnitude.
     If all three variants apply the same latent shift with different labels,
     monotonicity is baked into the inputs and the dose-response check is
     circular. The mirror diagnostics + the input-vs-output delta table are
     the guard against this.
  2. Perturbations are applied to the WARM-UP HISTORY (pre-encoding), never
     to decoded outputs, and never via z_override (removed API).
  3. Magnitudes below reuse the audited earlier_evacuation input delta
     (-0.12 on the exposure-relevant channel at the final warm-up step) as
     the 24h anchor, and scale timing/magnitude around it. Adjust the channel
     names to the real state-vector layout.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Scenario table.
#   steps_before_end : how many warm-up steps before the end of the warm-up
#                      window the evacuation signal starts (6h per step).
#   magnitude        : signed shift applied to the exposure-relevant input
#                      channel(s), ramped linearly from onset to window end
#                      (a step function is also defensible — pick one and
#                      document it in the audit).
# 12h / 24h / 36h => 2 / 4 / 6 six-hour steps of additional warning.
# ---------------------------------------------------------------------------
DOSE_RESPONSE_SCENARIOS = {
    "earlier_evacuation_12h": {"steps_before_end": 2, "magnitude": -0.06,
                               "target": "exposure"},
    "earlier_evacuation_24h": {"steps_before_end": 4, "magnitude": -0.12,
                               "target": "exposure"},   # audited anchor
    "earlier_evacuation_36h": {"steps_before_end": 6, "magnitude": -0.18,
                               "target": "exposure"},
    "delayed_evacuation_12h": {"steps_before_end": 2, "magnitude": +0.06,
                               "target": "exposure"},
    "delayed_evacuation_24h": {"steps_before_end": 4, "magnitude": +0.12,
                               "target": "exposure"},
}

# NOTE ON INTERPRETATION: timing and magnitude co-vary above (earlier onset
# AND larger cumulative shift), which mirrors "more warning => more people
# out earlier". If reviewers may object that magnitude alone drives the
# ordering, ALSO run the timing-only control below, where magnitude is fixed
# and only onset moves. If the ordering survives timing-only, the
# dose-response claim is much stronger; report both.
TIMING_ONLY_CONTROL = {
    "earlier_evac_t2_fixedmag": {"steps_before_end": 2, "magnitude": -0.12,
                                 "target": "exposure"},
    "earlier_evac_t4_fixedmag": {"steps_before_end": 4, "magnitude": -0.12,
                                 "target": "exposure"},
    "earlier_evac_t6_fixedmag": {"steps_before_end": 6, "magnitude": -0.12,
                                 "target": "exposure"},
}


def apply_ramped_shift(state_history, scenario: dict):
    """Reference implementation of the warm-up perturbation.

    state_history : tensor [T_warmup, n_nodes, n_channels] (adapt indexing)
    Applies a linear ramp of `magnitude` on the exposure channel(s) starting
    `steps_before_end` steps before the end of warm-up. Returns a COPY.

    TODO(Adrija): replace `exposure_channel_slice` with the real channel
    index/slice for the exposure-relevant observation group, matching how
    the existing earlier_evacuation scenario selects channels.
    """
    import torch  # local import so the table above is importable without torch
    out = state_history.clone()
    T = out.shape[0]
    k = int(scenario["steps_before_end"])
    onset = max(0, T - k)
    exposure_channel_slice = slice(None)  # TODO: real slice here
    for i, t in enumerate(range(onset, T)):
        frac = (i + 1) / max(1, T - onset)          # linear ramp 0 -> 1
        out[t, :, exposure_channel_slice] = torch.clamp(
            out[t, :, exposure_channel_slice] + frac * scenario["magnitude"],
            min=0.0, max=1.0,
        )
    return out


def register(registry: dict, include_timing_control: bool = True) -> dict:
    """Merge these scenarios into the existing scenario registry.

    Call from scenarios.py, e.g.:
        from .scenarios_dose_response import register
        SCENARIOS = register(SCENARIOS)
    """
    merged = dict(registry)
    merged.update(DOSE_RESPONSE_SCENARIOS)
    if include_timing_control:
        merged.update(TIMING_ONLY_CONTROL)
    return merged
