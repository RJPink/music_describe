# music-describe

Describe a music file with your own machine. Uses **TinyMU**, a compact 229M-parameter
music-understanding model (MATPAC audio encoder + SmolLM2-135M text decoder) that runs
locally — no cloud, no API keys.

## What it does

`music_describe.py` takes an audio file (mp3/ogg/wav/flac — anything torchaudio reads),
splits it into evenly-spaced ~10-second windows (the model's comfortable input length),
and generates a text description for each: style, tempo, energy, instruments, mood, and
sometimes media references ("game soundtrack", "retro game title screen").

## Caveats (read these)

- **TinyMU is 229M parameters. It genre-tags; it doesn't deeply understand music.**
  Descriptions are often stock-tag-ish ("hip-hop track with prominent bassline"), and on
  unfamiliar material (remixes, medleys) it can confidently *hallucinate* details that
  aren't in the audio (instruments, vocals, etc.). Treat output as a starting point to
  check against your own ears, not as ground truth.
- **Windows only describe what's in that 10-second slice.** A full-track read is the
  combination of the windows; the script reports them separately.
- One window = one generation pass (~10-30s on a modern GPU, 1-3 min on CPU for a
  short clip).

## Requirements

- Python 3.11 (3.10 may work; untested)
- ~2.5 GB disk for the TinyMU weights (downloaded from Hugging Face on first run)
- GPU optional but recommended: CUDA build of PyTorch. The script falls back to CPU
  automatically (slower).
- Working internet connection on first run (weight download).

## Setup

```bash
# 1. Clone TinyMU (model code + config)
git clone --depth 1 https://github.com/xiquan-li/TinyMU
cd TinyMU

# 2. Create an isolated environment
python -m venv tinymu-env
# Windows:  tinymu-env\Scripts\activate
# Linux:    source tinymu-env/bin/activate

# 3. Install the model's inference dependencies
pip install -r ../music-describe/requirements.txt
```

That's it. No ffmpeg needed (torchaudio reads ogg/mp3/wav directly).

## Use

```bash
python ../music-describe/music_describe.py --audio path/to/your/song.ogg
```

Useful flags:

| flag | default | meaning |
|---|---|---|
| `--num-windows` | 8 | how many evenly-spaced slices to describe |
| `--window-sec` | 10 | slice length (keep <= ~15 for best results) |
| `--prompt` | (see script) | description instruction you want |
| `--device` | auto | `cuda` or `cpu` |
| `--out` | `<audio>.music.json` | save results as JSON |

JSON output is one entry per window with start time and description text.

## How it works (for the curious)

The model's audio encoder was trained around a specific input length, so this tool
deliberately feeds it short, evenly-spaced windows instead of one long clip. It also
wraps the upstream `demo.py` behavior (single 10-second pass) with sampling, so a
3-minute track actually gets sampled across instead of just its intro.

## License / sharing note

The TinyMU model weights are published on Hugging Face by the authors (AndreasXi/TinyMU).
At the time of packaging, the upstream repo does not carry an explicit LICENSE file.
For personal, non-commercial use between friends that's generally fine in practice,
but if you intend to redistribute this package publicly or use it commercially, check
the upstream repo / contact the authors first.

---
Packaged by Sable for Shade, 2026-10-21.
