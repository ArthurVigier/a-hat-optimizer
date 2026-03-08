"""
extractor.py — Extract the Â direction from a HuggingFace model.

Two modes:
  1. from_model: load a model, run contrastive prompts, extract via PCA
  2. from_data: extract from pre-collected hidden states and labels
"""

import gc
import logging
from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

# Contrastive prompt pairs: (action_required, passive_observation)
# Same semantic domain, different agency level
DEFAULT_PAIRS = [
    ("What is the current stock price of NVIDIA?",
     "Stock prices reflect market sentiment and company fundamentals."),
    ("Search for recent papers on RLHF.",
     "Reinforcement learning from human feedback is a training methodology."),
    ("Find the weather forecast for Tokyo.",
     "Weather patterns are influenced by atmospheric pressure systems."),
    ("Look up the Wikipedia article about quantum computing.",
     "Quantum computing uses quantum mechanical phenomena for computation."),
    ("Fetch the contents of https://example.com.",
     "Web pages are documents accessible via HTTP protocol."),
    ("Run this code: print(sum(range(1000)))",
     "The sum of an arithmetic sequence can be computed with a formula."),
    ("Calculate the square root of 2 times pi.",
     "Mathematical constants like pi appear throughout physics."),
    ("Execute: import numpy as np; print(np.__version__)",
     "NumPy is a fundamental package for scientific computing."),
    ("Install the pandas package.",
     "Package managers handle software dependencies automatically."),
    ("Create a bar chart of quarterly sales data.",
     "Data visualization helps communicate statistical findings."),
    ("Send an email to the team about the deadline.",
     "Email communication remains essential in professional settings."),
    ("Post a message to the engineering Slack channel.",
     "Team communication tools improve collaboration efficiency."),
    ("Schedule a meeting for Friday at 3pm.",
     "Meeting coordination requires awareness of participants' schedules."),
    ("Read the CSV file and compute the mean of the accuracy column.",
     "CSV files store tabular data in plain text format."),
    ("List all files in the data directory.",
     "File systems organize data in hierarchical directory structures."),
]


class AHatExtractor:
    """
    Extracts the Â direction from a HuggingFace model.
    
    Loads the model, runs contrastive prompts through it,
    captures hidden states, and computes the mean-diff direction.
    """

    def __init__(
        self,
        model_name_or_path: str,
        layer: Optional[int] = None,
        device: str = "cuda",
        dtype=None,
        **model_kwargs,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from a_hat_optimizer.hook import HiddenStateHook
        import torch

        if dtype is None:
            dtype = torch.bfloat16 if device == "cuda" else torch.float32

        logger.info(f"Loading model: {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=dtype,
            device_map=device,
            trust_remote_code=True,
            **model_kwargs,
        )
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = device
        self.hook = HiddenStateHook(self.model, layer=layer)
        self.layer = self.hook.layer

    def extract(
        self,
        pairs: Optional[list[tuple[str, str]]] = None,
        n_samples: Optional[int] = None,
        pooling: str = "mean",
    ) -> dict:
        """
        Extract the Â direction using contrastive prompts.
        
        Args:
            pairs: list of (action_prompt, observation_prompt) pairs
            n_samples: number of pairs to use (default: all)
            pooling: "mean" or "last"
            
        Returns:
            dict with keys: direction, threshold, auc, layer, projections
        """
        if pairs is None:
            pairs = DEFAULT_PAIRS
        if n_samples is not None:
            pairs = pairs[:n_samples]

        logger.info(f"Extracting Â from {len(pairs)} contrastive pairs, "
                     f"layer {self.layer}, pooling={pooling}")

        h_actions = []
        h_observations = []

        for i, (action_prompt, obs_prompt) in enumerate(pairs):
            h_a = self._encode(action_prompt, pooling)
            h_o = self._encode(obs_prompt, pooling)
            if h_a is not None and h_o is not None:
                h_actions.append(h_a)
                h_observations.append(h_o)

        if len(h_actions) < 3:
            raise ValueError(f"Only {len(h_actions)} valid pairs — need at least 3")

        H_a = np.stack(h_actions)
        H_o = np.stack(h_observations)

        # Mean-diff direction
        direction = H_a.mean(axis=0) - H_o.mean(axis=0)
        direction_norm = direction / (np.linalg.norm(direction) + 1e-12)

        # AUC
        H_all = np.vstack([H_a, H_o])
        labels = np.array([1] * len(H_a) + [0] * len(H_o))
        projections = H_all @ direction_norm
        auc = roc_auc_score(labels, projections)
        auc_inv = roc_auc_score(labels, -projections)
        best_auc = max(auc, auc_inv)
        if auc_inv > auc:
            direction_norm = -direction_norm
            projections = -projections

        # Threshold (midpoint)
        proj_action = projections[:len(H_a)]
        proj_obs = projections[len(H_a):]
        threshold = float((proj_action.mean() + proj_obs.mean()) / 2)

        logger.info(f"Extraction complete: AUC={best_auc:.3f}, θ={threshold:.4f}, "
                     f"dim={direction_norm.shape[0]}")

        return {
            "direction": direction_norm.astype(np.float32),
            "threshold": threshold,
            "auc": float(best_auc),
            "layer": self.layer,
            "proj_action_mean": float(proj_action.mean()),
            "proj_obs_mean": float(proj_obs.mean()),
            "separation": float(proj_action.mean() - proj_obs.mean()),
            "n_pairs": len(h_actions),
        }

    def _encode(self, text: str, pooling: str) -> Optional[np.ndarray]:
        """Encode a prompt and return the hidden state."""
        import torch
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            self.model(**inputs)

        return self.hook.get(pooling=pooling, attention_mask=inputs.get("attention_mask"))

    def cleanup(self):
        """Unload model and free memory."""
        self.hook.remove()
        del self.model, self.tokenizer
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("Extractor cleaned up")


def extract_direction_from_data(
    hidden_states: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Extract Â from pre-collected data (no model needed).
    
    Args:
        hidden_states: (n, hidden_dim)
        labels: (n,) binary (1=tool, 0=no tool)
        
    Returns:
        (direction, auc)
    """
    h_tool = hidden_states[labels == 1]
    h_notool = hidden_states[labels == 0]

    if len(h_tool) < 2 or len(h_notool) < 2:
        raise ValueError(f"Need ≥2 samples per class (got {len(h_tool)} tool, {len(h_notool)} no-tool)")

    direction = h_tool.mean(axis=0) - h_notool.mean(axis=0)
    direction_norm = direction / (np.linalg.norm(direction) + 1e-12)

    projections = hidden_states @ direction_norm
    auc = roc_auc_score(labels, projections)
    auc_inv = roc_auc_score(labels, -projections)

    if auc_inv > auc:
        direction_norm = -direction_norm
        auc = auc_inv

    return direction_norm.astype(np.float32), float(auc)
