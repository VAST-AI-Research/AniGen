from typing import *


class GuidanceIntervalSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance with interval.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
        if cfg_interval[0] <= t <= cfg_interval[1]:
            pred = super()._inference_model(model, x_t, t, cond, **kwargs)
            neg_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
            return (1 + cfg_strength) * pred - cfg_strength * neg_pred
        else:
            return super()._inference_model(model, x_t, t, cond, **kwargs)


class AniGenGuidanceIntervalSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance with interval.
    """

    def _inference_model(self, model, x_t, x_t_skl, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
        if cfg_interval[0] <= t <= cfg_interval[1]:
            pred, pred_skl = super()._inference_model(model, x_t, x_t_skl, t, cond, **kwargs)
            neg_pred, neg_pred_skl = super()._inference_model(model, x_t, x_t_skl, t, neg_cond, **kwargs)
            return (1 + cfg_strength) * pred - cfg_strength * neg_pred, (1 + cfg_strength) * pred_skl - cfg_strength * neg_pred_skl
        else:
            return super()._inference_model(model, x_t, x_t_skl, t, cond, **kwargs)
