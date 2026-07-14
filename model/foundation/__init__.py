"""
STORM-CARE Foundation Model — Package

Self-supervised pretraining on HURDAT2 · IBTrACS · ERA5 · Vulnerability data.

Quick-start
-----------
  from model.foundation import FoundationConfig, FoundationModel, PretrainRunner

  cfg    = FoundationConfig().apply_demo_overrides()
  runner = PretrainRunner(cfg)
  runner.run()
"""
from .config            import FoundationConfig
from .data_pipeline     import (
    MultiSourceDataPipeline,
    StormRecord,
    parse_hurdat2_full,
    parse_ibtracs,
    compute_storm_features,
)
from .graph_construction import (
    GlobalStormGraph,
    build_global_storm_graph,
    build_window_graph,
    EDGE_TEMPORAL_NEXT,
    EDGE_TEMPORAL_SKIP,
    EDGE_INTER_STORM,
    EDGE_SELF_LOOP,
    N_EDGE_TYPES,
)
from .architecture      import (
    FoundationModel,
    StormTokenizer,
    ERA5PatchEncoder,
    VulnerabilityEncoder,
    FoundationBackbone,
    FutureStateHead,
    MaskedReconstructionHead,
    ContrastiveHead,
    MultiHorizonHead,
)
from .objectives        import (
    CombinedPretrainingObjective,
    FutureStateLoss,
    MaskedReconstructionLoss,
    ContrastiveEvolutionLoss,
    MultiHorizonLoss,
    sample_mask,
)
from .evaluation        import FoundationEvaluator
from .pretrain          import PretrainRunner, StormSequenceDataset

__all__ = [
    "FoundationConfig",
    "MultiSourceDataPipeline",
    "StormRecord",
    "parse_hurdat2_full",
    "parse_ibtracs",
    "compute_storm_features",
    "GlobalStormGraph",
    "build_global_storm_graph",
    "build_window_graph",
    "FoundationModel",
    "StormTokenizer",
    "ERA5PatchEncoder",
    "VulnerabilityEncoder",
    "FoundationBackbone",
    "CombinedPretrainingObjective",
    "sample_mask",
    "FoundationEvaluator",
    "PretrainRunner",
    "StormSequenceDataset",
]
