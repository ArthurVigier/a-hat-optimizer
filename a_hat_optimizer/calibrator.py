"""
calibrator.py — Threshold calibration strategies for Â.

Four strategies:
  - midpoint: halfway between tool and no-tool means (default, simple)
  - f1: threshold that maximizes F1 score
  - youden: threshold that maximizes Youden's J (sensitivity + specificity - 1)
  - percentile: threshold at a specific percentile of the no-tool projections
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class AHatCalibrator:
    """
    Calibrates the decision threshold for Â.
    
    Usage:
        calibrator = AHatCalibrator(direction)
        threshold = calibrator.calibrate(hidden_states, labels, strategy="f1")
    """

    def __init__(self, direction: np.ndarray):
        self.direction = direction / (np.linalg.norm(direction) + 1e-12)

    def calibrate(
        self,
        hidden_states: np.ndarray,
        labels: np.ndarray,
        strategy: str = "midpoint",
        **kwargs,
    ) -> float:
        """
        Calibrate threshold on labeled data.
        
        Args:
            hidden_states: (n, hidden_dim)
            labels: (n,) binary (1=tool, 0=no tool)
            strategy: "midpoint", "f1", "youden", or "percentile"
            **kwargs: strategy-specific parameters
            
        Returns:
            Calibrated threshold
        """
        projections = hidden_states.astype(np.float32) @ self.direction
        y = labels.astype(int)

        strategies = {
            "midpoint": self._midpoint,
            "f1": self._max_f1,
            "youden": self._youden,
            "percentile": self._percentile,
        }

        if strategy not in strategies:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {list(strategies.keys())}")

        threshold = strategies[strategy](projections, y, **kwargs)
        
        # Log stats
        pred = (projections > threshold).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        tn = ((pred == 0) & (y == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        logger.info(
            f"Calibration ({strategy}): θ={threshold:.4f}, "
            f"precision={precision:.3f}, recall={recall:.3f}, F1={f1:.3f}, "
            f"TP={tp}, FP={fp}, FN={fn}, TN={tn}"
        )

        return float(threshold)

    def _midpoint(self, projections: np.ndarray, labels: np.ndarray, **kwargs) -> float:
        """Midpoint between class means."""
        proj_tool = projections[labels == 1]
        proj_notool = projections[labels == 0]
        return float((proj_tool.mean() + proj_notool.mean()) / 2)

    def _max_f1(self, projections: np.ndarray, labels: np.ndarray, n_thresholds: int = 200, **kwargs) -> float:
        """Threshold that maximizes F1 score."""
        thresholds = np.linspace(projections.min(), projections.max(), n_thresholds)
        best_f1 = 0
        best_t = 0

        for t in thresholds:
            pred = (projections > t).astype(int)
            tp = ((pred == 1) & (labels == 1)).sum()
            fp = ((pred == 1) & (labels == 0)).sum()
            fn = ((pred == 0) & (labels == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        return float(best_t)

    def _youden(self, projections: np.ndarray, labels: np.ndarray, n_thresholds: int = 200, **kwargs) -> float:
        """Threshold that maximizes Youden's J statistic (sensitivity + specificity - 1)."""
        thresholds = np.linspace(projections.min(), projections.max(), n_thresholds)
        best_j = -1
        best_t = 0

        for t in thresholds:
            pred = (projections > t).astype(int)
            tp = ((pred == 1) & (labels == 1)).sum()
            fp = ((pred == 1) & (labels == 0)).sum()
            fn = ((pred == 0) & (labels == 1)).sum()
            tn = ((pred == 0) & (labels == 0)).sum()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            j = sensitivity + specificity - 1
            
            if j > best_j:
                best_j = j
                best_t = t

        return float(best_t)

    def _percentile(self, projections: np.ndarray, labels: np.ndarray, percentile: float = 95, **kwargs) -> float:
        """Threshold at a percentile of the no-tool projections.
        
        percentile=95 means: only 5% of no-tool steps would be wrongly
        classified as tool calls. Conservative, minimizes false positives.
        """
        proj_notool = projections[labels == 0]
        return float(np.percentile(proj_notool, percentile))

    def sweep(
        self,
        hidden_states: np.ndarray,
        labels: np.ndarray,
        n_thresholds: int = 100,
    ) -> dict:
        """
        Full threshold sweep with metrics at each point.
        
        Returns a dict with arrays for plotting precision-recall curves.
        """
        projections = hidden_states.astype(np.float32) @ self.direction
        y = labels.astype(int)
        
        thresholds = np.linspace(projections.min(), projections.max(), n_thresholds)
        results = {
            "thresholds": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "specificity": [],
            "youden_j": [],
        }

        for t in thresholds:
            pred = (projections > t).astype(int)
            tp = ((pred == 1) & (y == 1)).sum()
            fp = ((pred == 1) & (y == 0)).sum()
            fn = ((pred == 0) & (y == 1)).sum()
            tn = ((pred == 0) & (y == 0)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results["thresholds"].append(float(t))
            results["precision"].append(float(precision))
            results["recall"].append(float(recall))
            results["f1"].append(float(f1))
            results["specificity"].append(float(specificity))
            results["youden_j"].append(float(recall + specificity - 1))

        # Find best per metric
        results["best_f1_threshold"] = results["thresholds"][np.argmax(results["f1"])]
        results["best_youden_threshold"] = results["thresholds"][np.argmax(results["youden_j"])]

        return results
