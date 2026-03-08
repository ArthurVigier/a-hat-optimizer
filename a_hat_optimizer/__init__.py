"""
a_hat_optimizer — Extract and exploit the agency direction (Â) from LLM hidden states.

Â (a-hat) is a geometric direction in the hidden state space of LLMs that
predicts when the model should invoke a tool. Extracted via PCA on the
difference between tool-call and non-tool-call hidden states, Â achieves
AUC > 0.94 across model sizes from 1.7B to 8B parameters.

Usage:
    from a_hat_optimizer import AHat

    # Auto-extract from a HuggingFace model
    a_hat = AHat.from_model("Qwen/Qwen3-8B")
    
    # Or from pre-computed direction
    a_hat = AHat.from_file("a_hat_direction.npy")
    
    # Predict: should the model call a tool?
    should_call, confidence = a_hat.predict(hidden_state)
"""

__version__ = "0.1.0"

from a_hat_optimizer.core import AHat
from a_hat_optimizer.calibrator import AHatCalibrator

def __getattr__(name):
    if name == "AHatExtractor":
        from a_hat_optimizer.extractor import AHatExtractor
        return AHatExtractor
    if name == "HiddenStateHook":
        from a_hat_optimizer.hook import HiddenStateHook
        return HiddenStateHook
    raise AttributeError(f"module 'a_hat_optimizer' has no attribute '{name}'")

__all__ = [
    "AHat",
    "AHatExtractor",
    "AHatCalibrator",
    "HiddenStateHook",
]
