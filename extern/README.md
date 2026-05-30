# Vendored Metal packages

Forks of Pedro Naugusto's Apple Silicon Metal libraries, carrying our fixes
not yet upstreamed. Source: github.com/pawel-mazurkiewicz/<pkg>.
Vendored as plain source (nested .git stripped). Re-sync by re-cloning the fork.

| dir | import name | replaces | vendored fork commit |
|---|---|---|---|
| mtlgemm | flex_gemm | spconv sparse conv | 867aec8234299a7fe1ede7f802c8debe5a939a82 |
| mtldiffrast | mtldiffrast | nvdiffrast | 02e783a0dfcf0ba1acfd0300039bba6b99652883 |
| mtlmesh | cumesh | cumesh mesh post-processing | 39600375ad47a6342fe69d752cf33b1355a2b111 |
| mtlbvh | mtlbvh | CUDA BVH (used by mtlmesh) | cbaef38f905175328fb99648406714895c2da0c5 |

Drop the vendored copy and switch to upstream if/when these fixes merge.
Requires the Xcode Metal toolchain (`xcrun`) to compile `.metal` -> `.metallib`.
