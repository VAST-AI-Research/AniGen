from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import AniGenClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import AniGenGuidanceIntervalSamplerMixin
from ...utils.geodesic_noise import maybe_geodesic_smooth_slat_noise


class AniGenFlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """
    def __init__(
        self,
        sigma_min: float,
        geodesic_smooth_noise: bool = False,
        geodesic_smooth_noise_iters: int = 0,
        geodesic_smooth_noise_alpha: float = 0.7,
    ):
        self.sigma_min = sigma_min
        self.geodesic_smooth_noise = geodesic_smooth_noise
        self.geodesic_smooth_noise_iters = geodesic_smooth_noise_iters
        self.geodesic_smooth_noise_alpha = geodesic_smooth_noise_alpha

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps

    def _x0_eps_to_xt(self, x_0, eps, t: float):
            """Forward noising: x_t = (1-t)*x_0 + (sigma(t))*eps.

            Notes:
            - With this parameterization, at t=1 we have x_1 = eps.
            - This helper is used by `sample_inpaint` to build the known (masked) x_t from
                a known x_0 and a known noise/eps.
            """
            t = float(t)
            a = 1.0 - t
            b = float(self.sigma_min + (1.0 - self.sigma_min) * t)
            return a * x_0 + b * eps

    def _inference_model(self, model, x_t, x_t_skl, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        if cond is not None and cond.shape[0] == 1 and x_t.shape[0] > 1:
            cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))
        return model(x_t, x_t_skl, t, cond, **kwargs)

    def _get_model_prediction(self, model, x_t, x_t_skl, t, cond=None, **kwargs):
        pred_v, pred_v_skl = self._inference_model(model, x_t, x_t_skl, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        pred_x_0_skl, pred_eps_skl = self._v_to_xstart_eps(x_t=x_t_skl, t=t, v=pred_v_skl)
        return pred_x_0, pred_eps, pred_v, pred_x_0_skl, pred_eps_skl, pred_v_skl

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        x_t_skl,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        pred_x_0, pred_eps, pred_v, pred_x_0_skl, pred_eps_skl, pred_v_skl = self._get_model_prediction(model, x_t, x_t_skl, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        pred_x_prev_skl = x_t_skl - (t - t_prev) * pred_v_skl
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0, "pred_x_prev_skl": pred_x_prev_skl, "pred_x_0_skl": pred_x_0_skl})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        noise_skl,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            progress_callback: Optional callback(step_index, total_steps) invoked after each step.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = maybe_geodesic_smooth_slat_noise(
            noise, model,
            enabled=self.geodesic_smooth_noise,
            iters=self.geodesic_smooth_noise_iters,
            alpha=self.geodesic_smooth_noise_alpha,
        )
        sample_skl = noise_skl
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "samples_skl": None, "pred_x_t": [], "pred_x_0": [], "pred_x_t_skl": [], "pred_x_0_skl": []})
        for step_idx, (t, t_prev) in enumerate(tqdm(t_pairs, desc="Sampling", disable=not verbose)):
            out = self.sample_once(model, sample, sample_skl, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            sample_skl = out.pred_x_prev_skl
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
            ret.pred_x_t_skl.append(out.pred_x_prev_skl)
            ret.pred_x_0_skl.append(out.pred_x_0_skl)
            if progress_callback is not None:
                progress_callback(step_idx, steps)
        ret.samples = sample
        ret.samples_skl = sample_skl
        return ret

    @torch.no_grad()
    def sample_inpaint(
        self,
        model,
        noise,
        noise_skl,
        cond: Optional[Any] = None,
        *,
        # Standard sampling parameters
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        # Inpainting option A (dense tensors): known x0 + known noise(eps) + mask
        known_x0: Optional[torch.Tensor] = None,
        known_noise: Optional[torch.Tensor] = None,
        inpaint_mask: Optional[torch.Tensor] = None,
        known_x0_skl: Optional[torch.Tensor] = None,
        known_noise_skl: Optional[torch.Tensor] = None,
        inpaint_mask_skl: Optional[torch.Tensor] = None,
        # Inpainting option B (generic): user hook for SparseTensor / custom blending
        inpaint_fn: Optional[Callable[[Any, Any, float], Tuple[Any, Any]]] = None,
        **kwargs,
    ):
        """Euler sampling with inpainting-style constraints.

        This is useful when part of the latent (or sparse features) is known (GT) and
        should stay consistent during denoising, similar to diffusion inpainting.

        Two usage modes:
        1) Dense tensors (torch.Tensor):
           Provide `known_x0` (GT clean target), `known_noise` (eps; note x_1 == eps),
           and `inpaint_mask` (1 keeps GT region, 0 keeps sampled region). The sampler
           will, after every Euler step (at t_prev), blend:
             x_{t_prev} <- mask * known_x_{t_prev} + (1-mask) * x_{t_prev}
           where known_x_t is computed from (known_x0, known_noise) via forward noising.

        2) Generic hook (recommended for SparseTensor):
           Provide `inpaint_fn(sample, sample_skl, t_prev) -> (sample, sample_skl)`.
           This hook runs after every Euler step. You can implement coord-based GT
           overwrites there.

        All CFG-related kwargs (e.g. neg_cond/cfg_strength/cfg_interval) are passed
        through to `sample_once`.
        """

        def _blend_dense(x, known_x, mask):
            if mask is None or known_x is None:
                return x
            if not isinstance(x, torch.Tensor):
                raise TypeError("Dense inpainting mode requires torch.Tensor states; use `inpaint_fn` for SparseTensor.")
            m = mask
            if not isinstance(m, torch.Tensor):
                m = torch.as_tensor(m)
            if m.dtype != torch.float32 and m.dtype != torch.float16 and m.dtype != torch.bfloat16:
                m = m.float()
            m = m.to(device=x.device)
            known_x = known_x.to(device=x.device, dtype=x.dtype)
            return x * (1.0 - m) + known_x * m

        sample = maybe_geodesic_smooth_slat_noise(
            noise, model,
            enabled=self.geodesic_smooth_noise,
            iters=self.geodesic_smooth_noise_iters,
            alpha=self.geodesic_smooth_noise_alpha,
        )
        sample_skl = noise_skl
        t_seq = np.linspace(1, 0, int(steps) + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((float(t_seq[i]), float(t_seq[i + 1])) for i in range(int(steps)))
        ret = edict({"samples": None, "samples_skl": None, "pred_x_t": [], "pred_x_0": [], "pred_x_t_skl": [], "pred_x_0_skl": []})

        for step_idx, (t, t_prev) in enumerate(tqdm(t_pairs, desc="Sampling", disable=not verbose)):
            out = self.sample_once(model, sample, sample_skl, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            sample_skl = out.pred_x_prev_skl

            # Apply inpainting constraint at the *new* time (t_prev).
            if inpaint_fn is not None:
                sample, sample_skl = inpaint_fn(sample, sample_skl, float(t_prev))
            else:
                if known_x0 is not None or known_noise is not None or inpaint_mask is not None:
                    if known_x0 is None or known_noise is None or inpaint_mask is None:
                        raise ValueError("Dense inpainting requires `known_x0`, `known_noise`, and `inpaint_mask`.")
                    known_xt = self._x0_eps_to_xt(known_x0.to(device=sample.device, dtype=sample.dtype), known_noise.to(device=sample.device, dtype=sample.dtype), float(t_prev))
                    sample = _blend_dense(sample, known_xt, inpaint_mask)

                if known_x0_skl is not None or known_noise_skl is not None or inpaint_mask_skl is not None:
                    if known_x0_skl is None or known_noise_skl is None or inpaint_mask_skl is None:
                        raise ValueError("Dense inpainting for skl requires `known_x0_skl`, `known_noise_skl`, and `inpaint_mask_skl`.")
                    known_xt_skl = self._x0_eps_to_xt(known_x0_skl.to(device=sample_skl.device, dtype=sample_skl.dtype), known_noise_skl.to(device=sample_skl.device, dtype=sample_skl.dtype), float(t_prev))
                    sample_skl = _blend_dense(sample_skl, known_xt_skl, inpaint_mask_skl)

            ret.pred_x_t.append(sample)
            ret.pred_x_0.append(out.pred_x_0)
            ret.pred_x_t_skl.append(sample_skl)
            ret.pred_x_0_skl.append(out.pred_x_0_skl)
            if progress_callback is not None:
                progress_callback(step_idx, int(steps))

        ret.samples = sample
        ret.samples_skl = sample_skl
        return ret


class AniGenFlowEulerCfgSampler(AniGenClassifierFreeGuidanceSamplerMixin, AniGenFlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        noise_skl,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        verbose: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            progress_callback: Optional callback(step_index, total_steps) invoked after each step.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, noise_skl, cond, steps, rescale_t, verbose, progress_callback=progress_callback, neg_cond=neg_cond, cfg_strength=cfg_strength, **kwargs)


class AniGenFlowEulerGuidanceIntervalSampler(AniGenGuidanceIntervalSamplerMixin, AniGenFlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        noise_skl,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            progress_callback: Optional callback(step_index, total_steps) invoked after each step.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, noise_skl, cond, steps, rescale_t, verbose, progress_callback=progress_callback, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)
