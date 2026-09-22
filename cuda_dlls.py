"""Make the pip-installed CUDA/cuDNN DLLs findable on Windows.

The `nvidia-*-cu12` wheels drop their DLLs under
`site-packages/nvidia/<lib>/bin`, which is nowhere near the default loader
search path. onnxruntime loads `onnxruntime_providers_cuda.dll` with the plain
loader flags, so `os.add_dll_directory` alone is not enough - the dependency
chain (cublasLt64_12.dll, cudnn64_9.dll, ...) is only resolved via PATH.

Import this module BEFORE `import onnxruntime`.
"""

import os
import sys
import site


def _candidate_roots():
    roots = []
    try:
        roots.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        roots.append(site.getusersitepackages())
    except Exception:
        pass
    roots.append(os.path.join(sys.prefix, "Lib", "site-packages"))
    seen, out = set(), []
    for r in roots:
        if r and r not in seen and os.path.isdir(r):
            seen.add(r)
            out.append(r)
    return out


def register(verbose=False):
    """Add every nvidia wheel bin directory to PATH and the DLL search path."""
    if os.name != "nt":
        return []
    added = []
    for root in _candidate_roots():
        nvidia = os.path.join(root, "nvidia")
        if not os.path.isdir(nvidia):
            continue
        for dirpath, _dirs, files in os.walk(nvidia):
            if not any(f.lower().endswith(".dll") for f in files):
                continue
            added.append(dirpath)
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(dirpath)
                except Exception:
                    pass
    if added:
        os.environ["PATH"] = os.pathsep.join(added) + os.pathsep + os.environ.get("PATH", "")
    if verbose:
        for d in added:
            print("cuda dll dir:", d)
    return added


registered = register()
