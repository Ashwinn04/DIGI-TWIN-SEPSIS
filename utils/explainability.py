"""
Explainability utilities: Captum (IntegratedGradients, GradientShap) and SHAP.
Saves per-inference attributions as .npz for dashboard consumption.
"""
import os
from typing import Optional, Dict, Any
import numpy as np
import torch


def _model_forward_wrapper(model, features, masks, delta_t):
    if delta_t is not None:
        return model(features, masks, delta_t)
    return model(features, masks)


def compute_integrated_gradients(model, features: torch.Tensor, masks: torch.Tensor,
                                 delta_t: Optional[torch.Tensor] = None,
                                 target: Optional[int] = None, steps: int = 50) -> np.ndarray:
    """
    Compute Integrated Gradients attribution per input feature at each timestep.
    Returns ndarray with shape [seq_len, input_size]. For batch, call per-sample.
    """
    try:
        from captum.attr import IntegratedGradients
    except Exception as e:
        raise RuntimeError(f"Captum not available: {e}")

    model.eval()
    features = features.requires_grad_(True)
    ig = IntegratedGradients(lambda x: _model_forward_wrapper(model, x, masks, delta_t))
    attributions = ig.attribute(inputs=features.unsqueeze(0), target=target, n_steps=steps)
    return attributions.detach().cpu().numpy()[0]


def compute_gradient_shap(model, features: torch.Tensor, masks: torch.Tensor,
                          delta_t: Optional[torch.Tensor] = None,
                          target: Optional[int] = None, samples: int = 50) -> np.ndarray:
    """
    Compute Gradient SHAP attribution per input feature at each timestep.
    """
    try:
        from captum.attr import GradientShap
    except Exception as e:
        raise RuntimeError(f"Captum not available: {e}")

    model.eval()
    baseline = torch.zeros_like(features)
    gs = GradientShap(lambda x: _model_forward_wrapper(model, x, masks, delta_t))
    attributions = gs.attribute(inputs=features.unsqueeze(0), baselines=baseline.unsqueeze(0),
                                n_samples=samples)
    return attributions.detach().cpu().numpy()[0]


def save_attributions_npz(save_dir: str, filename: str, meta: Dict[str, Any], **arrays):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    np.savez_compressed(path, **arrays, **{f"meta_{k}": v for k, v in meta.items()})
    print(f"Attributions saved to {path}")
    return path


def extract_attention_weights(model, features: torch.Tensor, masks: torch.Tensor,
                              delta_t: Optional[torch.Tensor] = None) -> Optional[np.ndarray]:
    """
    Try to obtain attention weights from supported models. Returns [seq_len] or [seq_len] array per sample.
    """
    model.eval()
    try:
        if hasattr(model, 'get_attention_weights'):
            with torch.no_grad():
                weights = model.get_attention_weights(features.unsqueeze(0), masks.unsqueeze(0))
            return weights.detach().cpu().numpy()[0]
    except Exception:
        pass
    # For AttentionTransformerModel forward may return (logits, attn)
    try:
        with torch.no_grad():
            out = _model_forward_wrapper(model, features.unsqueeze(0), masks.unsqueeze(0),
                                         delta_t.unsqueeze(0) if delta_t is not None else None)
        if isinstance(out, tuple) and len(out) == 2:
            return out[1].detach().cpu().numpy()[0]
    except Exception:
        pass
    return None


