"""
core.py — The main AHat class.

Simple, modular, one import away from usage.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class AHat:
    """
    Agency direction detector for LLM hidden states.
    
    Â is a unit vector in the hidden state space. The projection of a hidden
    state onto Â predicts whether the model should invoke a tool at that step.
    
    Attributes:
        direction: unit vector (numpy, float32)
        threshold: decision boundary (projection > threshold → tool call)
        hidden_dim: dimensionality of the hidden states
        metadata: optional dict with extraction info (model, layer, AUC, etc.)
    """

    def __init__(
        self,
        direction: np.ndarray,
        threshold: float = 0.0,
        metadata: Optional[dict] = None,
    ):
        self.direction = (direction / (np.linalg.norm(direction) + 1e-12)).astype(np.float32)
        self.threshold = threshold
        self.hidden_dim = len(self.direction)
        self.metadata = metadata or {}
        
        # Lazy torch cache
        self._direction_torch: Optional[torch.Tensor] = None

    # ── Prediction ──────────────────────────────────────────────────────

    def predict(self, h: Union[np.ndarray, "torch.Tensor"]) -> tuple[bool, float]:
        """
        Predict whether a tool call is needed.
        
        Args:
            h: hidden state vector (numpy or torch, any dtype)
            
        Returns:
            (should_call_tool, confidence)
            confidence > 0 means tool call recommended
            confidence < 0 means no tool call
        """
        try:
            import torch as _torch
            if isinstance(h, _torch.Tensor):
                return self._predict_torch(h)
        except ImportError:
            pass
        
        h_flat = np.asarray(h).flatten().astype(np.float32)
        if h_flat.shape[0] != self.hidden_dim:
            raise ValueError(
                f"Hidden state dim {h_flat.shape[0]} != expected {self.hidden_dim}"
            )
        proj = float(np.dot(h_flat, self.direction))
        confidence = proj - self.threshold
        return confidence > 0, confidence

    def predict_batch(self, H: Union[np.ndarray, "torch.Tensor"]) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict for a batch of hidden states.
        
        Args:
            H: (batch, hidden_dim) array or tensor
            
        Returns:
            (should_call: bool array, confidences: float array)
        """
        try:
            import torch as _torch
            if isinstance(H, _torch.Tensor):
                H_np = H.detach().cpu().float().numpy()
            else:
                H_np = H.astype(np.float32)
        except ImportError:
            H_np = np.asarray(H).astype(np.float32)
        
        projections = H_np @ self.direction
        confidences = projections - self.threshold
        should_call = confidences > 0
        return should_call, confidences

    def _predict_torch(self, h: "torch.Tensor") -> tuple[bool, float]:
        """Torch-native prediction (stays on device)."""
        import torch
        if self._direction_torch is None or self._direction_torch.device != h.device:
            self._direction_torch = torch.from_numpy(self.direction).to(h.device, h.dtype)
        proj = float(torch.dot(h.flatten(), self._direction_torch))
        confidence = proj - self.threshold
        return confidence > 0, confidence

    # ── Factory methods ─────────────────────────────────────────────────

    @classmethod
    def from_model(
        cls,
        model_name_or_path: str,
        layer: Optional[int] = None,
        n_samples: int = 15,
        device: str = "cuda",
        dtype=None,
        **kwargs,
    ) -> "AHat":
        """
        Auto-extract Â from a HuggingFace model.
        
        Loads the model, runs contrastive prompts, extracts the mean-diff
        direction, and calibrates the threshold. One-liner setup.
        
        Args:
            model_name_or_path: HuggingFace model ID or local path
            layer: layer to extract from (default: middle layer)
            n_samples: number of contrastive pairs (default: 15)
            device: "cuda" or "cpu"
            dtype: torch dtype (default: bfloat16 if cuda, float32 if cpu)
            **kwargs: passed to AutoModelForCausalLM.from_pretrained
            
        Returns:
            Calibrated AHat instance
        """
        from a_hat_optimizer.extractor import AHatExtractor
        
        extractor = AHatExtractor(
            model_name_or_path=model_name_or_path,
            layer=layer,
            device=device,
            dtype=dtype,
            **kwargs,
        )
        result = extractor.extract(n_samples=n_samples)
        
        a_hat = cls(
            direction=result["direction"],
            threshold=result.get("threshold", 0.0),
            metadata={
                "model": model_name_or_path,
                "layer": result.get("layer"),
                "auc": result.get("auc"),
                "n_samples": n_samples,
                "extraction_method": "mean_diff",
            },
        )
        
        # Cleanup
        extractor.cleanup()
        
        return a_hat

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "AHat":
        """
        Load Â from a saved file.
        
        Supports:
            - .npy file (direction only, threshold=0)
            - .npz file (direction + threshold)
            - directory with direction.npy + config.json
        """
        path = Path(path)
        
        if path.is_dir():
            direction = np.load(path / "direction.npy")
            config_path = path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                threshold = config.get("threshold", 0.0)
                metadata = config
            else:
                threshold = 0.0
                metadata = {}
            return cls(direction, threshold, metadata)
        
        if path.suffix == ".npz":
            data = np.load(path)
            direction = data["direction"]
            threshold = float(data.get("threshold", 0.0))
            return cls(direction, threshold)
        
        # .npy — direction only
        direction = np.load(path)
        return cls(direction, threshold=0.0)

    @classmethod
    def from_traces(
        cls,
        hidden_states: np.ndarray,
        labels: np.ndarray,
        calibrate: bool = True,
    ) -> "AHat":
        """
        Extract Â from pre-collected hidden states and tool-call labels.
        
        Args:
            hidden_states: (n_steps, hidden_dim) array
            labels: (n_steps,) binary array (1=tool call, 0=no tool)
            calibrate: if True, calibrate threshold on the same data
            
        Returns:
            AHat instance
        """
        from a_hat_optimizer.extractor import extract_direction_from_data
        from a_hat_optimizer.calibrator import AHatCalibrator
        
        direction, auc = extract_direction_from_data(hidden_states, labels)
        
        threshold = 0.0
        if calibrate:
            calibrator = AHatCalibrator(direction)
            threshold = calibrator.calibrate(hidden_states, labels)
        
        return cls(
            direction=direction,
            threshold=threshold,
            metadata={"auc": auc, "n_samples": len(labels), "extraction_method": "mean_diff"},
        )

    # ── Serialization ───────────────────────────────────────────────────

    def save(self, path: Union[str, Path]):
        """
        Save Â to disk.
        
        Creates a directory with:
            direction.npy — the unit vector
            config.json — threshold, metadata, and extraction info
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        np.save(path / "direction.npy", self.direction)
        
        config = {
            "threshold": self.threshold,
            "hidden_dim": self.hidden_dim,
            **self.metadata,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)
        
        logger.info(f"AHat saved to {path}")

    # ── Threshold adjustment ────────────────────────────────────────────

    def set_threshold(self, threshold: float):
        """Manually set the decision threshold."""
        self.threshold = threshold
        logger.info(f"Threshold set to {threshold:.4f}")

    def auto_calibrate(
        self,
        hidden_states: np.ndarray,
        labels: np.ndarray,
        strategy: str = "midpoint",
    ):
        """
        Auto-calibrate threshold from labeled data.
        
        Args:
            hidden_states: (n, hidden_dim)
            labels: (n,) binary (1=tool, 0=no tool)
            strategy: "midpoint", "f1", "youden", or "percentile"
        """
        from a_hat_optimizer.calibrator import AHatCalibrator
        calibrator = AHatCalibrator(self.direction)
        self.threshold = calibrator.calibrate(hidden_states, labels, strategy=strategy)
        logger.info(f"Threshold auto-calibrated: θ={self.threshold:.4f} (strategy={strategy})")

    # ── Info ─────────────────────────────────────────────────────────────

    def info(self) -> dict:
        """Return summary info about this AHat instance."""
        return {
            "hidden_dim": self.hidden_dim,
            "threshold": self.threshold,
            "direction_norm": float(np.linalg.norm(self.direction)),
            **self.metadata,
        }

    def __repr__(self):
        auc = self.metadata.get("auc", "?")
        model = self.metadata.get("model", "?")
        return f"AHat(dim={self.hidden_dim}, θ={self.threshold:.4f}, AUC={auc}, model={model})"
