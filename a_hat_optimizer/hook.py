"""
hook.py — Forward hook for capturing hidden states from HuggingFace models.

Compatible with Llama, Qwen, Mistral, Gemma, Phi, and any model with
a standard transformer layer stack.
"""

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class HiddenStateHook:
    """
    Captures hidden states from a specific layer of a HuggingFace model.
    
    Usage:
        hook = HiddenStateHook(model, layer=16)
        # ... run model.generate() or model() ...
        h = hook.get(pooling="last")  # numpy array
        hook.remove()
    """

    def __init__(
        self,
        model: torch.nn.Module,
        layer: Optional[int] = None,
    ):
        """
        Args:
            model: HuggingFace CausalLM model
            layer: layer index (default: middle layer)
        """
        self.model = model
        self._captured: Optional[torch.Tensor] = None
        self._hook_handle = None

        # Detect architecture
        self.num_layers = self._detect_num_layers(model)
        self.layer = layer if layer is not None else self.num_layers // 2

        # Install hook
        self._install()

    def _detect_num_layers(self, model) -> int:
        """Detect number of layers across architectures."""
        if hasattr(model, "config"):
            for attr in ["num_hidden_layers", "n_layer", "num_layers"]:
                if hasattr(model.config, attr):
                    return getattr(model.config, attr)
        
        layers = self._get_layers(model)
        if layers is not None:
            return len(layers)
        
        return 32  # fallback

    def _get_layers(self, model) -> Optional[torch.nn.ModuleList]:
        """Find the layer stack across architectures."""
        # Standard: model.model.layers (Llama, Qwen, Mistral)
        if hasattr(model, "model"):
            inner = model.model
            if hasattr(inner, "layers"):
                return inner.layers
            if hasattr(inner, "decoder") and hasattr(inner.decoder, "layers"):
                return inner.decoder.layers
        # GPT-2 style: model.transformer.h
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h
        return None

    def _install(self):
        """Install the forward hook."""
        layers = self._get_layers(self.model)
        if layers is None:
            logger.warning("Cannot find model layers — hook not installed")
            return

        if self.layer >= len(layers):
            self.layer = len(layers) // 2
            logger.warning(f"Layer index out of range, falling back to {self.layer}")

        target = layers[self.layer]
        self._hook_handle = target.register_forward_hook(self._hook_fn)
        logger.info(f"Hook installed on layer {self.layer}/{self.num_layers}")

    def _hook_fn(self, module, input, output):
        """Capture the output of the target layer."""
        if isinstance(output, tuple):
            self._captured = output[0].detach()
        elif hasattr(output, "last_hidden_state"):
            self._captured = output.last_hidden_state.detach()
        else:
            self._captured = output.detach()

    def get(
        self,
        pooling: str = "last",
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Optional[np.ndarray]:
        """
        Get the captured hidden state.
        
        Args:
            pooling: "last" (last token), "mean" (mean pooling), or "all" (full sequence)
            attention_mask: for proper mean pooling with padding
            
        Returns:
            numpy array (hidden_dim,) for last/mean, (seq_len, hidden_dim) for all
        """
        if self._captured is None:
            return None

        h = self._captured  # (batch, seq_len, hidden_dim)

        if pooling == "all":
            return h.squeeze(0).cpu().float().numpy()

        if pooling == "last":
            if attention_mask is not None:
                seq_len = int(attention_mask.sum(dim=1)[0].item()) - 1
                return h[0, seq_len, :].cpu().float().numpy()
            return h[0, -1, :].cpu().float().numpy()

        if pooling == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).to(h.device, h.dtype)
                h_mean = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                h_mean = h.mean(dim=1)
            return h_mean.squeeze(0).cpu().float().numpy()

        raise ValueError(f"Unknown pooling: {pooling}. Use 'last', 'mean', or 'all'.")

    def remove(self):
        """Remove the hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
            logger.info("Hook removed")

    def __del__(self):
        self.remove()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove()
