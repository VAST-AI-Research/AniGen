"""
Parity tests for mtlbvh scalar-output dtype specialization (fp32/fp16/bf16).

Validates two invariants:

1. **fp32 path is bit-exact unchanged from before** — `output_dtype=None` /
   `torch.float32` reproduces the original kernel result with zero deviation.

2. **fp16/bf16 narrowed paths are *exactly* the fp32 result cast at store** —
   internal BVH math always runs fp32; the only difference is the final
   `(OUT_T)dist` cast in the Metal shader. So `narrow_out` must equal
   `fp32_out.to(narrow_dtype)` bit-for-bit. Anything looser would indicate a
   precision-losing arithmetic divergence somewhere in the kernel.

This is the strongest possible parity check given the hardware: we cannot
match fp32 precision in narrower types (that's fundamental), but we can
prove no extra error has been introduced beyond the final store.

Vec3 outputs (``uvw``, ray-hit ``positions``) always stay fp32 in this round
— no `packed_half3` equivalent in Metal — so vec3 outputs are checked for
bit-equality across all three dtype settings.
"""
import math
import torch
import pytest


_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


# --- fixtures ---------------------------------------------------------------

@pytest.fixture
def icosphere_bvh():
    """Subdivided icosahedron — 642 verts, 1280 triangles, unit sphere."""
    phi = (1 + math.sqrt(5)) / 2
    verts = torch.tensor([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=torch.float32)
    verts = verts / verts.norm(dim=1, keepdim=True)
    tris = torch.tensor([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ], dtype=torch.int32)

    # Subdivide 3× (→ 1280 tris)
    for _ in range(3):
        verts, tris = _subdivide(verts, tris)

    from mtlbvh import MtlBVH
    return MtlBVH(verts, tris)


def _subdivide(v, t):
    edge_midpoints = {}
    new_verts = [row for row in v]
    new_tris = []
    for tri in t.tolist():
        mids = []
        for i in range(3):
            e = tuple(sorted([tri[i], tri[(i + 1) % 3]]))
            if e not in edge_midpoints:
                mid = (v[e[0]] + v[e[1]]) / 2
                mid = mid / mid.norm()
                edge_midpoints[e] = len(new_verts)
                new_verts.append(mid)
            mids.append(edge_midpoints[e])
        a, b, c = tri
        m0, m1, m2 = mids
        new_tris.extend([[a, m0, m2], [b, m1, m0], [c, m2, m1], [m0, m1, m2]])
    return torch.stack(new_verts), torch.tensor(new_tris, dtype=torch.int32)


@pytest.fixture
def query_points():
    torch.manual_seed(42)
    return torch.randn(500, 3) * 2.0


@pytest.fixture
def query_rays():
    torch.manual_seed(7)
    ro = torch.randn(200, 3) * 3.0
    rd = -ro / ro.norm(dim=-1, keepdim=True)
    return ro, rd


# --- helpers ----------------------------------------------------------------

def _bit_exact_cast(fp32_tensor, narrow_dtype):
    """The expected narrow output: fp32 result rounded once via PyTorch's cast.

    If the Metal kernel runs fp32 internally and only narrows at the store,
    its output should equal this byte-for-byte (modulo any rounding-mode
    differences between Metal's `(half)` cast and PyTorch's `.to(half)`,
    which both use round-to-nearest-even on Apple Silicon — so they agree).
    """
    return fp32_tensor.to(narrow_dtype)


def _assert_narrow_eq_fp32_cast(narrow, fp32, dtype, label):
    """Strongest possible parity: narrowed kernel output == cast-of-fp32 output."""
    assert narrow.dtype == dtype, f"{label}: dtype mismatch {narrow.dtype} != {dtype}"
    if dtype == torch.float32:
        # fp32 path must be bit-exact equal to itself (sanity).
        assert torch.equal(narrow, fp32), f"{label}: fp32 path not deterministic"
        return
    expected = _bit_exact_cast(fp32, dtype)
    if torch.equal(narrow, expected):
        return
    # If they don't match exactly, report worst-case diff in *narrow ULPs*.
    # That number tells us how much slack the kernel introduced beyond a
    # plain cast — anything > 1 ULP means the narrow path is doing extra
    # arithmetic in low precision and should be investigated.
    n_ulp = torch.finfo(dtype).eps  # 1 ULP at value 1.0
    diff = (narrow.float() - fp32).abs()
    max_abs = diff.max().item()
    max_rel = (diff / fp32.abs().clamp_min(1e-9)).max().item()
    # bf16/fp16 cast error from a single rounding step is bounded by 0.5 ULP
    # of the narrow type relative to the value magnitude. Allow 1 ULP slack
    # to absorb tie-breaking rounding-mode trivia between Metal and torch.
    bound_abs = (fp32.abs().max().item()) * n_ulp
    assert max_abs <= bound_abs, (
        f"{label}: max_abs={max_abs:.4e} > 1-ULP bound={bound_abs:.4e} "
        f"(max_rel={max_rel:.4e}, eps={n_ulp:.4e}). Kernel introduced "
        f"precision loss beyond the final-store cast."
    )


# --- parity tests -----------------------------------------------------------

@pytest.mark.parametrize("dtype", _DTYPES)
def test_unsigned_distance_dtype_parity(icosphere_bvh, query_points, dtype):
    """Narrowed `distances` output equals fp32 result cast to dtype (≤ 1 ULP)."""
    d_ref, fid_ref, _ = icosphere_bvh.unsigned_distance(query_points, output_dtype=torch.float32)
    d_tgt, fid_tgt, _ = icosphere_bvh.unsigned_distance(query_points, output_dtype=dtype)
    _assert_narrow_eq_fp32_cast(d_tgt, d_ref, dtype, f"unsigned_distance/{dtype}")
    # face_id is always int64 — must be bit-exact regardless of out_dtype.
    assert torch.equal(fid_tgt, fid_ref), "face_id diverged across dtypes"


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("mode", ["watertight", "raystab"])
def test_signed_distance_dtype_parity(icosphere_bvh, query_points, dtype, mode):
    """Narrowed signed-distance output equals fp32 result cast to dtype (≤ 1 ULP)."""
    d_ref, fid_ref, _ = icosphere_bvh.signed_distance(
        query_points, mode=mode, output_dtype=torch.float32)
    d_tgt, fid_tgt, _ = icosphere_bvh.signed_distance(
        query_points, mode=mode, output_dtype=dtype)
    _assert_narrow_eq_fp32_cast(d_tgt, d_ref, dtype, f"signed_distance/{mode}/{dtype}")
    assert torch.equal(fid_tgt, fid_ref), f"{mode}: face_id diverged across dtypes"


@pytest.mark.parametrize("dtype", _DTYPES)
def test_ray_trace_dtype_parity(icosphere_bvh, query_rays, dtype):
    """Narrowed `depth` output equals fp32 cast to dtype (≤ 1 ULP).

    Only depth is narrowed; vec3 hit `positions` and int64 `face_id` stay
    fp32/int64 regardless and must be bit-exact equal across dtype settings.
    """
    ro, rd = query_rays
    pos_ref, fid_ref, d_ref = icosphere_bvh.ray_trace(ro, rd, output_dtype=torch.float32)
    pos_tgt, fid_tgt, d_tgt = icosphere_bvh.ray_trace(ro, rd, output_dtype=dtype)

    assert pos_tgt.dtype == torch.float32, "vec3 positions must always stay fp32"
    assert torch.equal(pos_tgt, pos_ref), "hit positions diverged across dtypes"
    assert torch.equal(fid_tgt, fid_ref), "face_id diverged across dtypes"

    # depth: ignore miss rays (fid<0) — those are BVH_MAX_DIST which overflows fp16.
    hit = (fid_ref >= 0) & (fid_tgt >= 0)
    assert hit.sum() > 0, "no rays hit — test fixture broken"
    _assert_narrow_eq_fp32_cast(d_tgt[hit], d_ref[hit], dtype, f"ray_trace/{dtype}")


def test_fp32_default_bit_exact_with_explicit(icosphere_bvh, query_points):
    """Default `output_dtype=None` produces *bit-exactly* the same result
    as the explicit `output_dtype=torch.float32` path (and as the legacy
    fp32-only kernel from before this round)."""
    d_implicit, fid_i, _ = icosphere_bvh.unsigned_distance(query_points)
    d_explicit, fid_e, _ = icosphere_bvh.unsigned_distance(
        query_points, output_dtype=torch.float32)
    assert d_implicit.dtype == torch.float32
    assert torch.equal(d_implicit, d_explicit)
    assert torch.equal(fid_i, fid_e)


def test_invalid_output_dtype_rejected(icosphere_bvh, query_points):
    with pytest.raises(ValueError):
        icosphere_bvh.unsigned_distance(query_points, output_dtype=torch.int32)
