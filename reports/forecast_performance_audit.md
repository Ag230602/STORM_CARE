# Forecast Performance Claim Audit

## Verdict
- STORM-CARE does not currently have a supported all-horizon forecast-superiority claim against Persistence.
- On the Irma/Ian ERA5-complete case study, GNO+DynGNN loses to Persistence at 6/12/24/48 h.
- On the same case study, GNO+DynGNN beats Transformer at 6/12/24/48 h and beats LSTM at 12/24/48 h, but loses to LSTM at 6 h.
- CLIPER is only available in the storm-level HURDAT2 baseline table; no STORM-CARE neural model is evaluated under that exact protocol.
- The foundation checkpoint metrics are validation-demo numbers and must not be compared as a test-set superiority claim.

## Case-Study Horizon Verdicts
| baseline    | 6             | 12            | 24            | 48            |
|:------------|:--------------|:--------------|:--------------|:--------------|
| LSTM        | not_supported | supported_win | supported_win | supported_win |
| Persistence | not_supported | not_supported | not_supported | not_supported |
| Transformer | supported_win | supported_win | supported_win | supported_win |

## Required Manuscript Claim
Use: "The corrected benchmark reports calibrated/probabilistic forecasts and a reproducible neural baseline study. Persistence remains the strongest short-horizon and overall case-study baseline; learned-model superiority is not claimed."

Avoid: "STORM-CARE outperforms Persistence/CLIPER/LSTM/Transformer/GNO at all horizons."

Source table: `tables/table_forecast_performance_audit.csv`
