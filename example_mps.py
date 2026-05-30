"""Apple Silicon entrypoint: bootstrap device shims, then run example.main()."""
import sys
import anigen_mps  # noqa: F401  -- configures env + installs knn/nvdiffrast shims at import


def _install_fp32_upcast():
    """Wrap AnigenImageTo3DPipeline.from_pretrained to upcast fp16 models to fp32 on Mac.

    AniGen builds its flow models and decoders with use_fp16=True; MPS mishandles
    mixed-dtype matmuls. We upcast the whole pipeline to fp32 right after load. CUDA is
    unaffected: anigen_mps.upcast_pipeline_fp32 no-ops when torch.cuda.is_available().
    """
    from anigen.pipelines.anigen_image_to_3d import AnigenImageTo3DPipeline

    _orig = AnigenImageTo3DPipeline.from_pretrained  # staticmethod -> plain function

    def _patched(*args, **kwargs):
        pipeline = _orig(*args, **kwargs)
        anigen_mps.upcast_pipeline_fp32(pipeline)
        return pipeline

    AnigenImageTo3DPipeline.from_pretrained = staticmethod(_patched)


if __name__ == "__main__":
    if "--device" not in sys.argv:
        sys.argv += ["--device", "mps"]
    _install_fp32_upcast()
    import example
    example.main()
