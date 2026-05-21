"""Lightning callbacks for SkyScapesNet."""
from skyscapesnet.callbacks.class_weights import ScheduledClassWeightsCallback
from skyscapesnet.callbacks.stitching import StitchingCallback

__all__ = ["ScheduledClassWeightsCallback", "StitchingCallback"]
