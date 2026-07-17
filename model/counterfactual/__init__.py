"""model.counterfactual — Module 5: Counterfactual Reasoning Engine"""
from .config    import CounterfactualConfig
from .engine    import CounterfactualEngine
from .scenarios import SCENARIO_INTERVENTIONS, SCENARIO_DESCRIPTIONS

__all__ = ["CounterfactualConfig", "CounterfactualEngine",
           "SCENARIO_INTERVENTIONS", "SCENARIO_DESCRIPTIONS"]
