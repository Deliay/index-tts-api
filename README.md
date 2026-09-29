# IndexTTS API

HTTP API service wrapping [IndexTTS-2.5](https://github.com/index-tts/index-tts)
zero-shot text-to-speech inference.

## Requirements

- [uv](https://docs.astral.sh/uv/) (manages the Python toolchain and dependencies)
- An NVIDIA GPU with ~8 GB of free VRAM for comfortable inference
- The IndexTTS-2.5 checkpoints (see below)

Python is pinned to 3.11 because the upstream `indextts` package requires
`>=3.10,<3.12`.

## Getting started

```bash
uv sync
```

If PyPI is slow, re-resolve through a mirror. The lockfile pins the exact
download URLs, so the mirror has to be chosen at lock time:

```bash
uv lock --default-index "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
uv sync
```

This rewrites `uv.lock` with the mirror's URLs; omit the flag on a fast network.
Mirrors can lag behind PyPI — switch mirrors if resolution reports a missing
version.

Download the IndexTTS-2.5 checkpoints into `checkpoints/`:

```bash
uv tool install huggingface-hub
hf download IndexTeam/IndexTTS-2.5 --local-dir checkpoints
```

or, via ModelScope:

```bash
uv tool install modelscope
modelscope download --model IndexTeam/IndexTTS-2.5 --local_dir checkpoints
```

Then start the API:

```bash
uv run index-tts-api
```

The server listens on `http://0.0.0.0:8000` by default. Interactive API docs are
served at `/docs` and `/openapi.json`. The model is loaded lazily on the first
synthesis request unless `INDEX_TTS_EAGER_LOAD=true` is set.

## Endpoints

| Method | Path          | Description                          |
| ------ | ------------- | ------------------------------------ |
| GET    | `/health`     | Liveness check plus model state      |
| POST   | `/synthesize` | Clone a voice and speak text         |

### `POST /synthesize`

`multipart/form-data`. Returns a mono 16-bit PCM WAV at 22.05 kHz.

| Field                         | Type   | Required | Default | Description                                                                     |
| ----------------------------- | ------ | -------- | ------- | ------------------------------------------------------------------------------- |
| `text`                        | string | yes      |         | Text to speak.                                                                   |
| `lang`                        | string | yes      |         | One of `ZH`, `EN`, `JA`, `ES`, `AR`.                                             |
| `speaker_prompt`              | file   | yes      |         | Reference voice clip to clone.                                                   |
| `emo_prompt`                  | file   | no       |         | Separate emotional reference clip.                                               |
| `emo_alpha`                   | float  | no       | `1.0`   | Emotion strength when `emo_prompt` is given (`0.0`–`1.0`).                       |
| `emo_vector`                  | string | no       |         | Eight comma-separated intensities: `happy,angry,sad,afraid,disgusted,melancholic,surprised,calm`. |
| `emo_text`                    | string | no       |         | Emotion description; used with `use_emo_text`.                                   |
| `use_emo_text`                | bool   | no       | `false` | Infer emotion from the text (requires `INDEX_TTS_USE_QWEN_EMO=true`).             |
| `use_random`                  | bool   | no       | `false` | Sample emotion vectors randomly.                                                 |
| `duration_factor`             | float  | no       | `1.0`   | `>1.0` slows speech down, `<1.0` speeds it up (`0.5`–`2.0`).                      |
| `text_normalization`          | bool   | no       | `true`  | Normalize numbers/dates before synthesis.                                        |
| `max_text_tokens_per_segment` | int    | no       | `120`   | Text chunk size.                                                                 |
| `interval_silence`            | int    | no       | `200`   | Silence between text segments, in milliseconds.                                  |

```bash
curl -X POST http://localhost:8000/synthesize \
  -F "text=Hello world, this is a cloned voice." \
  -F "lang=EN" \
  -F "speaker_prompt=@voice.wav" \
  -o speech.wav
```

With an explicit emotion vector and slower delivery:

```bash
curl -X POST http://localhost:8000/synthesize \
  -F "text=对不起嘛！我的记性真的不太好。" \
  -F "lang=ZH" \
  -F "speaker_prompt=@voice.wav" \
  -F "emo_vector=0,0,0.8,0,0,0,0,0" \
  -F "duration_factor=1.2" \
  -o speech.wav
```

### `GET /health`

```json
{
  "status": "ok",
  "service": "IndexTTS API",
  "version": "0.1.0",
  "model": { "name": "IndexTTS-2.5", "state": "not_loaded", "version": null, "error": null }
}
```

`model.state` is one of `not_loaded`, `loading`, `ready`, `error` — use it as a
readiness signal. `status` only reflects process liveness.

## Configuration

Settings are read from the environment. All variables are optional.

| Variable                          | Default               | Description                                        |
| --------------------------------- | --------------------- | -------------------------------------------------- |
| `INDEX_TTS_APP_NAME`              | `IndexTTS API`        | Service name reported by `/health`.                |
| `INDEX_TTS_HOST`                  | `0.0.0.0`             | Bind address.                                       |
| `INDEX_TTS_PORT`                  | `8000`                | Bind port.                                          |
| `INDEX_TTS_RELOAD`                | `false`               | Enable uvicorn auto-reload.                         |
| `INDEX_TTS_MODEL_DIR`             | `checkpoints`         | IndexTTS-2.5 checkpoint directory.                  |
| `INDEX_TTS_CFG_PATH`              | `<model_dir>/config.yaml` | Model config path.                              |
| `INDEX_TTS_DEVICE`                | auto                  | `cuda:0`, `cpu`, …                                  |
| `INDEX_TTS_USE_BF16`              | `true`                | BF16 inference (IndexTTS-2.5).                      |
| `INDEX_TTS_USE_CUDA_KERNEL`       | `false`               | BigVGAN fused CUDA kernel.                          |
| `INDEX_TTS_USE_QWEN_EMO`          | `false`               | Load the Qwen emotion model (needed for `use_emo_text`). |
| `INDEX_TTS_USE_TORCH_COMPILE`     | `false`               | `torch.compile` for the s2mel stage.                |
| `INDEX_TTS_VERBOSE`               | `false`               | Verbose inference logging.                          |
| `INDEX_TTS_EAGER_LOAD`            | `false`               | Load the model at startup.                          |
| `INDEX_TTS_WORK_DIR`              | `<tmp>/index-tts-api` | Scratch space for reference-audio caching.          |

## Docker

The image carries the runtime only — the checkpoints are multi-gigabyte and
change independently of the code, so they are mounted at `/app/checkpoints`.

```bash
docker build -t index-tts-api .

docker run --rm --gpus all -p 8000:8000 \
  -v "$PWD/checkpoints:/app/checkpoints" \
  index-tts-api
```

On a slow network, point the build at mirrors. The lockfile pins absolute download
URLs, so a PyPI mirror has to be applied by re-locking, and `uv` also needs a
mirror for the Python interpreter it downloads from GitHub:

```bash
podman build -t index-tts-api \
  --build-arg UV_DEFAULT_INDEX=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
  --build-arg UV_PYTHON_INSTALL_MIRROR=https://ghfast.top/https://github.com/astral-sh/python-build-standalone/releases/download \
  .
```

Both arguments are empty by default, which is what CI uses.

`--gpus all` needs the NVIDIA Container Toolkit. The mount point must contain the
same layout as the local `checkpoints/` directory:

```
checkpoints/
├── config.yaml                                  # declares version: 2.5
├── gpt.pth  codec.pth  s2mel.pth  feat1.pt  feat2.pt
├── wav2vec2bert_stats.pt
├── multilingual_zh_ja_yue_char_del.tiktoken
├── qwen0.6bemo4-merge/                          # only for INDEX_TTS_USE_QWEN_EMO=true
└── hf_cache/
    ├── w2v-bert-2.0/
    ├── campplus_cn_common.bin
    └── bigvgan/
```

The container runs as the non-root `app` user (uid 1000), so the mounted
directory must be readable by it. If your checkpoints are owned by a different
uid, either `chown` them or run with `--user "$(id -u):$(id -g)"`.

Configuration is unchanged (`INDEX_TTS_*` variables, see above), and `/health` is
wired up as the container `HEALTHCHECK`.

The build fetches from three hosts: PyPI, GitHub (the pinned `indextts`
revision), and `download.pytorch.org` (the CUDA wheel index upstream pins
`torch` to).

### Continuous integration

`.github/workflows/build.yml` builds the image and pushes it to
`<DOCKER_ENDPOINT>/zero-tools/index-tts-api` on every push to `main` (and on
manual dispatch), tagged with both `latest` and the commit SHA. It reads the
repository secrets `DOCKER_ENDPOINT`, `DOCKER_USER` and `DOCKER_PASSWORD`.

## Development

```bash
uv run pytest                 # endpoint and unit tests (no GPU needed)
uv run pytest -m model        # exercises the real model (needs checkpoints + GPU)
```
