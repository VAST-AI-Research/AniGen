"""Apple Silicon entrypoint: bootstrap device shims, then run example.main()."""
import sys
import anigen_mps  # noqa: F401  -- configures env + installs knn/nvdiffrast shims at import

if __name__ == "__main__":
    if "--device" not in sys.argv:
        sys.argv += ["--device", "mps"]
    import example
    example.main()
