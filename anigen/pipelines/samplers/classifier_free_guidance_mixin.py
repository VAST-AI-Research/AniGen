from typing import *


class ClassifierFreeGuidanceSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, **kwargs):
        pred = super()._inference_model(model, x_t, t, cond, **kwargs)
        neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
        return (1 + cfg_strength) * pred - cfg_strength * neg_pred


class AniGenClassifierFreeGuidanceSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance.
    """

    def _inference_model(self, model, x_t, x_t_skl, t, cond, neg_cond, cfg_strength, **kwargs):
        # Allow per-branch kwargs for the negative/unconditional pass via `neg_<name>`.
        # Example: pass `joints_num=J` and `neg_joints_num=0` to apply CFG on joints number.
        neg_overrides = {}
        shared_kwargs = {}
        for k, v in kwargs.items():
            if k.startswith('neg_'):
                neg_overrides[k[4:]] = v
            else:
                shared_kwargs[k] = v

        pred, pred_skl = super()._inference_model(model, x_t, x_t_skl, t, cond, **shared_kwargs)
        neg_kwargs = dict(shared_kwargs)
        neg_kwargs.update(neg_overrides)
        neg_pred, neg_pred_skl = super()._inference_model(model, x_t, x_t_skl, t, neg_cond, **neg_kwargs)
        return (1 + cfg_strength) * pred - cfg_strength * neg_pred, (1 + cfg_strength) * pred_skl - cfg_strength * neg_pred_skl
