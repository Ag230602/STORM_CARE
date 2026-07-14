"""model.counterfactual — Module 5: Counterfactual Reasoning Engine"""
from .config    import CounterfactualConfig
from .engine    import CounterfactualEngine
from .scenarios import SCENARIOS, SCENARIO_DESCRIPTIONS

__all__ = ["CounterfactualConfig", "CounterfactualEngine",
           "SCENARIOS", "SCENARIO_DESCRIPTIONS"]
