#!/usr/bin/env python
"""music_describe.py — describe a music file locally with TinyMU.

Self-contained runner. Designed to be portable: it locates the TinyMU source
checkout relative to itself (or via TINYMU_ROOT), downloads weights on first
run, fixes a known config path bug, checks torch/torchaudio version match,
and samples the track into evenly-spaced windows.

Usage:
    python music_describe.py --audio path/to/song.ogg
    python music_describe.py --audio song.mp3 --num-windows 8 --window-sec 10
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate the TinyMU source checkout
# ---------------------------------------------------------------------------
def find_tinymu_root():
    env = os.environ.get("TINYMU_ROOT")
    candidates = [
        Path(env) if env else None,
        Path(__file__).parent.parent / "TinyMU",          # sibling of this folder
        Path.home() / "TinyMU",                            # common default
    ]
    for c in candidates:
        if c and (c / "src" / "train_accelerate.py").exists():
            return c
    raise SystemExit(
        "Could not find the TinyMU source checkout.\n"
        "Clone it with:  git clone --depth 1 https://github.com/xiquan-li/TinyMU\n"
        "or set TINYMU_ROOT=/path/to/TinyMU and re-run."
    )


def preflight(root):
    """Cheap checks before we load heavy things."""
    import torch
    import torchaudio
    import torchvision
    print(f"[torch] {torch.__version__}  [torchaudio] {torchaudio.__version__} "
          f"[torchvision] {torchvision.__version__}")
    # The classic crash: torch and torchaudio on different releases.
    # (torchvision intentionally ships on a DIFFERENT version numbering scheme
    #  — 0.19.x pairs with torch 2.4.x — so only torch vs torchaudio matters.)
    def ver(x):
        m = re.match(r"(\d+\.\d+)", x)
        return m.group(1) if m else None
    vt, va = ver(torch.__version__), ver(torchaudio.__version__)
    if vt != va:
        raise SystemExit(
            f"VERSION MISMATCH: torch {torch.__version__} vs torchaudio "
            f"{torchaudio.__version__} vs torchvision {torchvision.__version__}.\n"
            "Reinstall a matched set, e.g.:\n"
            "  pip install torch==2.4.1+cu121 torchaudio==2.4.1+cu121 "
            "torchvision==0.19.1+cu121 --index-url https://download.pytorch.org/whl/cu121"
        )
    cuda = torch.cuda.is_available()
    print(f"[cuda] available={cuda}")
    return cuda


def ensure_weights(root):
    ckpt = root / "ckpt"
    need_model = not (ckpt / "tinymu.pt").exists()
    need_encoder = not (ckpt / "matpac_plus_as_48_1_map_enconly.pt").exists()
    if not need_model and not need_encoder:
        print(f"[weights] present at {ckpt}")
        return
    print(f"[download] fetching TinyMU weights -> {ckpt} "
          f"(~2.3 GB, first run only)")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="AndreasXi/TinyMU", local_dir=str(ckpt))
    # Known upstream config bug: tinymu.yaml points the MATPAC encoder at
    # ./weights/ but the snapshot lands in ./ckpt/. Rewrite the relative path.
    cfg = ckpt / "tinymu.yaml"
    if cfg.exists():
        txt = cfg.read_text()
        fixed = txt.replace("./weights/", "./ckpt/")
        if fixed != txt:
            cfg.write_text(fixed)
            print("[fix] repointed matpac_ckpt_path ./weights/ -> ./ckpt/ in tinymu.yaml")


# ---------------------------------------------------------------------------
# Windowed inference
# ---------------------------------------------------------------------------
def evenly_spaced_offsets(total_samples, sr, n, window_sec):
    total_sec = total_samples / sr
    win = window_sec * sr
    if total_sec <= window_sec:
        return [0]
    if n <= 1:
        return [0]
    span = total_samples - win
    return [int(i * span / (n - 1)) for i in range(n)]


def main():
    ap = argparse.ArgumentParser(description="Describe a music file with TinyMU.")
    ap.add_argument("--audio", required=True, help="path to audio file")
    ap.add_argument("--num-windows", type=int, default=8)
    ap.add_argument("--window-sec", type=int, default=10)
    ap.add_argument("--prompt", default=(
        "Describe the music you hear, including its style, tempo, energy, "
        "instruments, and overall mood. If it seems to reference a song, game, "
        "character, or other media, say so."))
    ap.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    ap.add_argument("--strategy", default="greedy", choices=["greedy", "top-p"])
    ap.add_argument("--out", default=None, help="output JSON path")
    a = ap.parse_args()

    root = find_tinymu_root()
    print(f"[root] {root}")
    sys.path.append(str(root))
    sys.path.append(str(root / "src"))

    cuda = preflight(root)
    device = a.device or ("cuda" if cuda else "cpu")

    ensure_weights(root)

    import torch
    import torchaudio
    os.chdir(root)  # the model loads relative ./ckpt/ paths
    from src.train_accelerate import get_model_and_tokenizer

    print("[model] loading TinyMU (MATPAC encoder + SmolLM2-135M)")
    model, tokenizer, config = get_model_and_tokenizer(
        config=str(root / "ckpt" / "tinymu.yaml"),
        model_ckpt_path=str(root / "ckpt" / "tinymu.pt"),
    )

    wav, sr = torchaudio.load(a.audio)
    total = wav.shape[-1]
    offsets = evenly_spaced_offsets(total, sr, a.num_windows, a.window_sec)
    print(f"[audio] {total / sr:.1f}s, sampling {len(offsets)} windows "
          f"of ~{a.window_sec}s each, device={device}")

    results = []
    for i, off in enumerate(offsets):
        seg = wav[:, off:off + a.window_sec * sr]
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        torchaudio.save(tmp, seg, sr)
        resp = model.generate(
            samples=[(tmp, a.prompt)], max_len=300, top_p=0.8,
            temperature=1.0, tokenizer=tokenizer, strategy=a.strategy,
        )
        txt = resp[0] if isinstance(resp, list) else str(resp)
        start_s = off / sr
        print(f"\n=== window {i + 1}/{len(offsets)}  (start {start_s:.1f}s) ===")
        print(txt.strip(), flush=True)
        results.append({"window": i + 1, "start_s": round(start_s, 1), "text": txt.strip()})
        os.unlink(tmp)

    out = a.out or os.path.splitext(a.audio)[0] + ".music.json"
    with open(out, "w") as f:
        json.dump({"audio": a.audio, "device": device,
                   "num_windows": len(offsets), "results": results}, f, indent=2)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
