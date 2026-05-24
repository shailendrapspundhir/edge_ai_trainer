"""GGUF export via llama.cpp's convert + quantize tools.

Looks for llama.cpp at `${LLAMA_CPP_DIR}` (or `./vendor/llama.cpp`). If not
found, raises a clear actionable error — we do not auto-clone, that's a
decision for the operator (and a 200+ MB checkout).

Steps:
    1. ensure `merged/` exists (calls safetensors_merge.export)
    2. python3 convert_hf_to_gguf.py merged/ --outfile artifacts/<run_id>/model.f16.gguf
    3. llama-quantize model.f16.gguf model.<quant>.gguf <quant>
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from eat.logging_setup import get_logger
from eat.paths import repo_root, run_artifacts_dir

log = get_logger("export.gguf")


def _llama_cpp_dir() -> Path:
    env = os.environ.get("LLAMA_CPP_DIR")
    if env:
        return Path(env).expanduser()
    return repo_root() / "vendor" / "llama.cpp"


def _convert_script(llcpp_dir: Path) -> Path:
    for name in ("convert_hf_to_gguf.py", "convert-hf-to-gguf.py", "convert.py"):
        p = llcpp_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"convert script not found in {llcpp_dir}. Expected one of "
        "convert_hf_to_gguf.py / convert-hf-to-gguf.py / convert.py. "
        "Clone llama.cpp: `git clone --depth=1 https://github.com/ggerganov/llama.cpp vendor/llama.cpp`"
    )


def _quantize_bin(llcpp_dir: Path) -> str:
    # Prefer locally built; fall back to PATH.
    for cand in (llcpp_dir / "build" / "bin" / "llama-quantize",
                 llcpp_dir / "llama-quantize",
                 llcpp_dir / "quantize"):
        if cand.exists():
            return str(cand)
    onpath = shutil.which("llama-quantize") or shutil.which("quantize")
    if onpath:
        return onpath
    raise FileNotFoundError(
        "llama-quantize binary not found. Build llama.cpp: "
        "`cmake -B build -DGGML_CUDA=ON && cmake --build build -j --target llama-quantize`"
    )


def export(run_id: str, quant: str = "Q4_K_M") -> Path:
    from eat.export import safetensors_merge

    merged = safetensors_merge.export(run_id)
    llcpp = _llama_cpp_dir()
    if not llcpp.exists():
        raise FileNotFoundError(
            f"llama.cpp not found at {llcpp}. Set LLAMA_CPP_DIR or clone to vendor/llama.cpp"
        )
    out_dir = run_artifacts_dir(run_id)
    f16 = out_dir / "model.f16.gguf"
    q_out = out_dir / f"model.{quant}.gguf"
    if not f16.exists():
        log.info("gguf_convert", merged=str(merged), out=str(f16))
        subprocess.run(
            ["python3", str(_convert_script(llcpp)), str(merged),
             "--outfile", str(f16), "--outtype", "f16"],
            check=True,
        )
    if not q_out.exists():
        log.info("gguf_quantize", quant=quant, out=str(q_out))
        subprocess.run([_quantize_bin(llcpp), str(f16), str(q_out), quant], check=True)
    return q_out
