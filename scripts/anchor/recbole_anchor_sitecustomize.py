"""recbole-anchor venv shims (Track C, REBUILD_WAVE_SPEC P1.1).

Canonical copy — installed as ``sitecustomize.py`` in
``/projects/PROJECT_ALLOC/envs/recbole-anchor/lib/python3.10/site-packages/``
by scripts/anchor/setup_recbole_env.sh. Keep the two in sync.

RecBole 1.2.1 predates numpy>=1.24 (which removed the deprecated scalar
aliases ``np.float`` / ``np.bool`` / ...) and torch>=2.6 (which flipped
``torch.load``'s default to ``weights_only=True``, breaking RecBole
checkpoint load). Rather than editing installed copies of either library,
this sitecustomize installs two post-import shims:

  * numpy: re-inject the removed scalar aliases as the builtin types
    (exactly what numpy<1.20 aliased them to).
  * torch: wrap ``torch.load`` so ``weights_only`` defaults to ``False``.
    Callers that pass it explicitly keep their value.

The shims fire lazily on first import of numpy / torch, so pip and other
tooling inside the venv pay no import cost. The venv is anchor-only and is
never activated by production pipelines.
"""

import importlib.util
import sys

_NUMPY_ALIASES = {
    "float": float, "int": int, "bool": bool, "object": object,
    "str": str, "complex": complex, "long": int, "unicode": str,
}


def _patch_numpy(np):
    import warnings
    for name, target in _NUMPY_ALIASES.items():
        with warnings.catch_warnings():
            # numpy>=2 emits FutureWarning from __getattr__ for some removed
            # aliases (np.object / np.str); the probe itself should be silent.
            warnings.simplefilter("ignore")
            present = hasattr(np, name)
        if not present:
            setattr(np, name, target)


def _patch_torch(torch):
    orig = torch.load
    if getattr(orig, "_anchor_weights_only_shim", False):
        return
    import functools

    @functools.wraps(orig)
    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig(*args, **kwargs)

    load._anchor_weights_only_shim = True
    torch.load = load


_PATCHERS = {"numpy": _patch_numpy, "torch": _patch_torch}


class _AnchorShimFinder:
    """Meta-path hook: run the matching patcher right after first import."""

    def find_spec(self, name, path=None, target=None):
        if name not in _PATCHERS:
            return None
        # Delegate to the real finders with ourselves removed, then wrap the
        # loader so the patcher fires after the module body executes.
        try:
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        patcher = _PATCHERS[name]
        orig_exec = spec.loader.exec_module

        def exec_module(module, _orig=orig_exec, _patch=patcher):
            _orig(module)
            try:
                _patch(module)
            except Exception as exc:  # loud, but never break the import itself
                print(f"[recbole-anchor sitecustomize] shim for "
                      f"{module.__name__} FAILED: {exc!r}", file=sys.stderr)

        try:
            spec.loader.exec_module = exec_module
        except AttributeError:
            # Exotic loader refusing attribute assignment — skip the lazy
            # wrap; the eager fallback below only helps if already imported.
            pass
        return spec


for _mod, _patch in _PATCHERS.items():
    if _mod in sys.modules:  # already imported before sitecustomize (unusual)
        _patch(sys.modules[_mod])
sys.meta_path.insert(0, _AnchorShimFinder())
