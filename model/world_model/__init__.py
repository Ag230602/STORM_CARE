"""model.world_model — Module 4: RSSM World Model"""
from .config       import WorldModelConfig
from .architecture import WorldModel
from .train        import WorldModelTrainer

__all__ = ["WorldModelConfig", "WorldModel", "WorldModelTrainer"]
