#!/usr/bin/env python3
"""Static AST census of sparse-conv / spatial-op construction sites.

Decision gate for the Apple Silicon inference port (Task 7).

The flex_gemm sparse-conv backend supports SUBMANIFOLD conv ONLY
(stride == 1 AND padding is None). Strided ``SparseConv3d`` and any
``SparseInverseConv3d`` raise ``NotImplementedError`` on that backend.

This scanner walks the model + module source files with Python's ``ast``
(NO heavy imports, NO weight downloads), finds every construction of the
sparse-conv and spatial-op classes, resolves the args it can (especially
``stride`` and ``padding``), classifies each site as SUBMANIFOLD / STRIDED /
INVERSE / SPATIAL, and prints a summary table plus a final VERDICT.

Usage:
    .venv-mac/bin/python scripts/probe_sparse_ops.py [PATH ...]

If no PATHs are given, the default decode-path file set is scanned.
"""
from __future__ import annotations

import argparse
import ast
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

# Call names we care about. Match either the bare name or the trailing
# attribute, e.g. both ``SparseConv3d(...)`` and ``sp.SparseConv3d(...)``.
SPARSE_CONV_NAMES = {"SparseConv3d", "SparseInverseConv3d"}
SPATIAL_NAMES = {"SparseDownsample", "SparseUpsample", "SparseSubdivide"}
TARGET_NAMES = SPARSE_CONV_NAMES | SPATIAL_NAMES

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Files that DEFINE the sparse-conv wrappers (not decode-path *usage*). Their
# internal ``spconv.SparseConv3d`` / ``spconv.SparseInverseConv3d`` delegations
# are the backend implementation, reachable only when a MODEL constructs a
# strided/inverse conv. We report their sites but exclude them from the GAP
# verdict (the model-file census decides whether they are ever reached).
DEFINITION_FILES = {
    os.path.join("anigen", "modules", "sparse", "conv", "conv_spconv.py"),
    os.path.join("anigen", "modules", "sparse", "conv", "conv_flex_gemm.py"),
    os.path.join("anigen", "modules", "sparse", "conv", "conv_torchsparse.py"),
}

# Default decode-path coverage set (relative to repo root).
DEFAULT_TARGETS = [
    "anigen/models/structured_latent_vae/",      # SLAT VAE decoder + sub-decoders
    "anigen/modules/sparse/spatial.py",          # spatial ops internals
    "anigen/modules/sparse/conv/conv_spconv.py", # the SparseConv3d/InverseConv3d defs
    "anigen/models/anigen_sparse_structure_vae.py",  # SS (sparse-structure) decoder
]

# Classifications
SUBMANIFOLD = "SUBMANIFOLD"
STRIDED = "STRIDED"
INVERSE = "INVERSE"
SPATIAL = "SPATIAL"
UNKNOWN = "UNKNOWN"  # could not resolve stride/padding -> manual check needed


@dataclass
class Site:
    file: str
    line: int
    call_name: str          # the trailing attribute / bare name, e.g. "SparseConv3d"
    classification: str
    stride_repr: str
    padding_repr: str
    needs_manual_check: bool = False
    note: str = ""
    is_definition: bool = False  # site lives in a backend-definition file, not a model


def _call_name(node: ast.Call) -> Optional[str]:
    """Return the trailing identifier of the call target, or None."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


# SparseConv3d.__init__ positional order (from conv_spconv.py):
#   (in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None)
SPARSECONV3D_POS = ["in_channels", "out_channels", "kernel_size", "stride",
                    "dilation", "padding", "bias", "indice_key"]
# SparseInverseConv3d.__init__ positional order:
#   (in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None)
SPARSEINVCONV3D_POS = ["in_channels", "out_channels", "kernel_size", "stride",
                       "dilation", "bias", "indice_key"]


def _resolve_arg(node: ast.Call, name: str, pos_order: List[str]):
    """Return (found: bool, value_repr: str, is_literal: bool, literal_value).

    Looks first at keyword args, then positional by declared order.
    """
    # keyword
    for kw in node.keywords:
        if kw.arg == name:
            return _eval_node(kw.value)
    # positional
    if name in pos_order:
        idx = pos_order.index(name)
        if idx < len(node.args):
            return _eval_node(node.args[idx])
    return (False, None, False, None)


def _eval_node(value: ast.AST):
    """Try to literal-eval a node. Return (found, repr, is_literal, value)."""
    try:
        lit = ast.literal_eval(value)
        return (True, repr(lit), True, lit)
    except Exception:
        try:
            expr = ast.unparse(value)
        except Exception:
            expr = "<unparseable>"
        return (True, expr, False, None)


def classify_call(node: ast.Call, call_name: str, file: str) -> Site:
    line = node.lineno

    if call_name in SPATIAL_NAMES:
        # Spatial ops: pure gather/scatter, never strided conv. Classified SPATIAL.
        return Site(file=file, line=line, call_name=call_name,
                    classification=SPATIAL, stride_repr="-", padding_repr="-")

    if call_name == "SparseInverseConv3d":
        return Site(file=file, line=line, call_name=call_name,
                    classification=INVERSE, stride_repr="(inverse)",
                    padding_repr="(inverse)",
                    note="SparseInverseConv3d is unsupported by flex_gemm")

    # call_name == "SparseConv3d"
    s_found, s_repr, s_lit, s_val = _resolve_arg(node, "stride", SPARSECONV3D_POS)
    p_found, p_repr, p_lit, p_val = _resolve_arg(node, "padding", SPARSECONV3D_POS)

    # Defaults: stride=1, padding=None.
    if not s_found:
        s_repr, s_lit, s_val = "1 (default)", True, 1
    if not p_found:
        p_repr, p_lit, p_val = "None (default)", True, None

    needs_manual = (s_found and not s_lit) or (p_found and not p_lit)

    if needs_manual:
        return Site(file=file, line=line, call_name=call_name,
                    classification=UNKNOWN, stride_repr=s_repr,
                    padding_repr=p_repr, needs_manual_check=True,
                    note="stride/padding is a non-literal expression; verify manually")

    # Both literal-resolved. Submanifold iff stride==1 and padding is None.
    stride_is_one = (s_val == 1) or (
        isinstance(s_val, (tuple, list)) and all(x == 1 for x in s_val))
    padding_is_none = p_val is None

    if stride_is_one and padding_is_none:
        cls = SUBMANIFOLD
    else:
        cls = STRIDED
    return Site(file=file, line=line, call_name=call_name, classification=cls,
                stride_repr=s_repr, padding_repr=p_repr,
                note="" if cls == SUBMANIFOLD else "strided/padded -> spconv.SparseConv3d (unsupported by flex_gemm)")


def scan_file(path: str) -> List[Site]:
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    try:
        tree = ast.parse(src, filename=path)
    except SyntaxError as e:
        print(f"  !! SyntaxError parsing {path}: {e}", file=sys.stderr)
        return []
    rel = os.path.relpath(path, REPO_ROOT)
    is_def = rel in DEFINITION_FILES
    sites: List[Site] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in TARGET_NAMES:
                site = classify_call(node, name, rel)
                site.is_definition = is_def
                sites.append(site)
    return sites


def collect_files(targets: List[str]) -> List[str]:
    files: List[str] = []
    for t in targets:
        abs_t = t if os.path.isabs(t) else os.path.join(REPO_ROOT, t)
        if os.path.isdir(abs_t):
            for root, _dirs, fnames in os.walk(abs_t):
                if "__pycache__" in root:
                    continue
                for fn in sorted(fnames):
                    if fn.endswith(".py"):
                        files.append(os.path.join(root, fn))
        elif os.path.isfile(abs_t):
            files.append(abs_t)
        else:
            print(f"  !! target not found: {t}", file=sys.stderr)
    # de-dup, stable
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*", help="files/dirs to scan (default: decode path set)")
    args = parser.parse_args(argv)

    targets = args.targets if args.targets else DEFAULT_TARGETS
    files = collect_files(targets)

    all_sites: List[Site] = []
    for f in files:
        all_sites.extend(scan_file(f))

    # Sort for stable output.
    all_sites.sort(key=lambda s: (s.file, s.line))

    # Census table.
    print("=" * 100)
    print("SPARSE-CONV / SPATIAL-OP CENSUS (static AST scan)")
    print("=" * 100)
    print(f"Scanned {len(files)} file(s); found {len(all_sites)} construction site(s).\n")

    hdr = f"{'CLASS':<12} {'CALL':<20} {'STRIDE':<16} {'PADDING':<16} LOCATION"
    print(hdr)
    print("-" * 100)
    for s in all_sites:
        loc = f"{s.file}:{s.line}"
        tag = "  [backend def]" if s.is_definition else ""
        print(f"{s.classification:<12} {s.call_name:<20} {s.stride_repr:<16} {s.padding_repr:<16} {loc}{tag}")
        if s.note:
            print(f"{'':<12} -> {s.note}")
    print("-" * 100)
    print("[backend def] = site inside a conv backend definition file (the wrapper's")
    print("                internal spconv delegation), NOT a model-construction site.")
    print("                These are reachable only if a model builds such a conv;")
    print("                the model-file census below decides the verdict.")

    # Tallies.
    counts = {}
    for s in all_sites:
        counts[s.classification] = counts.get(s.classification, 0) + 1
    print("\nTallies:")
    for k in (SUBMANIFOLD, SPATIAL, STRIDED, INVERSE, UNKNOWN):
        if k in counts:
            print(f"  {k:<12}: {counts[k]}")

    # Gap detection. Only SparseConv3d/SparseInverseConv3d in MODEL files matter
    # for flex_gemm. Backend-definition sites are excluded (see note above).
    model_sites = [s for s in all_sites if not s.is_definition]
    strided = [s for s in model_sites if s.classification == STRIDED]
    inverse = [s for s in model_sites if s.classification == INVERSE]
    unknown = [s for s in model_sites if s.classification == UNKNOWN]

    print()
    if unknown:
        print("MANUAL-CHECK REQUIRED (non-literal stride/padding):")
        for s in unknown:
            print(f"  {s.file}:{s.line} {s.call_name} stride={s.stride_repr} padding={s.padding_repr}")
        print()

    if strided or inverse:
        sites_desc = []
        for s in strided:
            sites_desc.append(f"STRIDED SparseConv3d @ {s.file}:{s.line} (stride={s.stride_repr}, padding={s.padding_repr})")
        for s in inverse:
            sites_desc.append(f"SparseInverseConv3d @ {s.file}:{s.line}")
        print("VERDICT: GAP - " + "; ".join(sites_desc))
        return 1

    if unknown:
        print("VERDICT: INCONCLUSIVE - non-literal stride/padding sites need manual confirmation (see above)")
        return 2

    print("VERDICT: SUBMANIFOLD-SUFFICIENT")
    print("  No strided SparseConv3d and no SparseInverseConv3d constructed on the scanned decode path.")
    print("  All sparse convs are submanifold (stride==1, padding=None); spatial up/downsampling")
    print("  uses SparseSubdivide / SparseDownsample / SparseUpsample (gather/scatter, no conv).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
