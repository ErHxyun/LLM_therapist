This repository reproduces the core CaiTI pipeline from:

**LLM-based Conversational AI Therapist for Daily Functioning Screening and Psychotherapeutic Intervention via Everyday Smart Devices**  
<https://doi.org/10.1145/3712299>

```bibtex
@article{10.1145/3712299,
author = {Nie, Jingping and Shao, Hanya (Vera) and Fan, Yuang and Shao, Qijia and You, Haoxuan and Preindl, Matthias and Jiang, Xiaofan},
title = {LLM-based Conversational AI Therapist for Daily Functioning Screening and Psychotherapeutic Intervention via Everyday Smart Devices},
year = {2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3712299},
doi = {10.1145/3712299},
journal = {ACM Trans. Comput. Healthcare},
month = jan,
keywords = {Large Language Models, AI therapist, Psychotherapy, Everyday Smart Devices, Cognitive Behavioral Therapy, Motivational Interviewing}
}
```

## Overview

This branch is set up for a local CaiTI deployment, especially Jetson Orin Nano 8GB validation. It does not require OpenAI or any cloud API. The LLM path uses the Hugging Face bundle `xxue752/caiti_best_model`, with one local base model and six task-specific LoRA adapters.

The implementation is intentionally constrained to match the CaiTI paper:

- Screening question selection uses epsilon-greedy Q-learning.
- The conversation flow is fixed as screening -> Response Analyzer -> R-V process -> CBT.
- The 37 screening dimensions are fixed in order and naming.
- Scores are limited to `0`, `1`, and `2`.
- Task-specific LLM roles remain separate; Reasoner, Guide, and Validator are not merged into one prompt.
- LLM modules receive text only. Audio, STT/TTS metadata, and emotion labels must not enter LLM prompts.
- Session report columns are fixed as `Score`, `Responses`, `Analysis`.

See [docs/jetson_deployment.md](docs/jetson_deployment.md) for the Jetson deployment runbook and [docs/jetson_codex_handoff_report.md](docs/jetson_codex_handoff_report.md) for detailed handoff notes.

## Project Structure

```text
.
|-- LLM_therapist_Application.py          # Text/console entrypoint
|-- LLM_therapist_Application_server.py   # Flask text API entrypoint
|-- LLM_therapist_Voice_Application.py    # Local STT/TTS shell around text pipeline
|-- config.yaml                           # Paths, RL settings, local model, voice backend config
|-- assets/audio/                         # Optional local waiting-music assets, ignored except .gitkeep
|-- data/
|   |-- libs/question_lib_v4.json         # Active 37-dimension question library
|   `-- q_tables/                         # Subject Q-tables
|-- docs/
|   |-- jetson_deployment.md             # Jetson setup, validation, running, troubleshooting
|   `-- jetson_codex_handoff_report.md
|-- scripts/
|   |-- benchmark_jetson.py               # Local LLM latency/memory benchmark
|   |-- faster_whisper_stt_command.py     # Local microphone recording + Faster-Whisper STT command
|   |-- long_session_reliability.py       # Mocked full 37-dimension + CBT reliability run
|   |-- piper_tts_command.py              # Local Piper TTS command
|   |-- run_voice_app.sh                  # One-command voice app launcher
|   |-- smoke_test_voice.py               # Local voice hardware/config smoke test
|   |-- trace_mock_e2e.py                 # Mock end-to-end control-flow trace
|   |-- trace_voice_io.py                 # Mock voice bridge trace
|   `-- smoke_test_adapters.py            # Real local adapter smoke test
|-- src/
|   |-- handler_rl.py                     # Top-level Q-learning screening loop
|   |-- questioner.py                     # Question asking, scoring, R-V trigger
|   |-- response_analyzer.py              # Task1/task2 response analysis prompts
|   |-- reflection_validation.py          # R-V Reasoner, Guide, Validator
|   |-- CBT.py                            # 3-stage CBT flow
|   |-- local_llm/                        # Local model runtime and adapter routing
|   |-- voice/                            # Console/command STT, TTS, and waiting-music backends
|   `-- utils/                            # Config, logging, reports, contracts, Q-tables
`-- tests/                                # Unit and mock-flow tests
```

## Local Model Setup

Default model settings live in `config.yaml`:

```yaml
local_llm:
  model_id: "xxue752/caiti_best_model"
  base_subdir: "compressed_model_int4"
  tokenizer_id: "xxue752/caiti_best_model"
  tokenizer_subdir: "compressed_model_int4"
  device_map: "cuda:0"
```

For an offline Jetson deployment, download the model bundle locally and override the paths:

```bash
export CAITI_MODEL_ID=/path/to/caiti_best_model
export CAITI_TOKENIZER_ID=/path/to/caiti_best_model
export CAITI_BASE_SUBDIR=compressed_model_int4
export CAITI_TOKENIZER_SUBDIR=compressed_model_int4
```

Expected model layout:

```text
caiti_best_model/
  compressed_model_int4/
  adapters/
    task1_response_analyzer/
    task2_general_response/
    task3_rv_reasoner/
    task4_cbt_stage1/
    task4_cbt_stage2/
    task4_cbt_stage3/
```

## Environment

Use Python 3.10 or 3.11. On Jetson, install PyTorch, Transformers, PEFT, Accelerate, and bitsandbytes according to the JetPack/CUDA version on the device.

```bash
conda env create -f environment_upgradable.yml
conda activate LLM_therapist
```

On JetPack 6 / CUDA 12.6 aarch64, the standard PyPI bitsandbytes wheel may import but fail at generation time. Use the Jetson AI Lab package index for Jetson-specific PyTorch and bitsandbytes builds:

```bash
python -m pip install --user --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126 torch
python -m pip install --user transformers==5.5.3 peft accelerate pandas numpy flask flask-cors coloredlogs word2number 'Pillow>=10'
python -m pip install --user --no-cache-dir --force-reinstall --no-deps \
  --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126 bitsandbytes
python -m bitsandbytes
```

The CaiTI model bundle was exported with Transformers 5.5.3. Older 4.x releases may require tokenizer fallback and can produce unstable adapter output formats.

Core dependencies:

- `torch`
- `transformers`
- `Pillow`
- `peft`
- `accelerate`
- `bitsandbytes`
- `pandas`
- `numpy`
- `pyyaml`
- `flask`
- `flask-cors`
- `coloredlogs`
- `word2number`

Optional local voice dependencies, following the Jetson smart-speaker pattern:

- `faster-whisper>=1.0.0,<2.0.0` for local STT
- `webrtcvad` for WebRTC voice-activity detection during microphone capture
- `piper-tts` plus a Piper `.onnx` voice model for local TTS

Do not install `openai-whisper` or the `whisper` alias for Jetson voice mode; they pull a much heavier PyTorch stack and can crowd the 8GB memory budget.

The default voice config expects a Piper voice at:

```text
models/piper/en_US-amy-medium.onnx
models/piper/en_US-amy-medium.onnx.json
```

The `models/piper/*.onnx` files are intentionally ignored by git because they are local device assets.

Optional waiting music can be stored at:

```text
assets/audio/music.wav
```

Audio files under `assets/audio/` are ignored by git, except `.gitkeep`, because they are local runtime assets. The stage-one music backend uses the system player command in `config.yaml` and does not add a new Python audio dependency.

## Validation Order

Run these in order on the target machine. If one step fails, stop and debug that step before moving on.

```bash
python -m unittest discover -s tests -q
python -m compileall src scripts tests LLM_therapist_Application.py LLM_therapist_Voice_Application.py LLM_therapist_Application_server.py
python scripts/trace_mock_e2e.py
python scripts/trace_voice_io.py
python scripts/smoke_test_voice.py --dry-run
CAITI_DEVICE_MAP=cuda:0 python scripts/smoke_test_adapters.py
CAITI_DEVICE_MAP=cuda:0 python scripts/benchmark_jetson.py --iterations 3 --warmup 1
python scripts/long_session_reliability.py
```

Purpose:

- Unit tests check contracts, paper consistency, R-V, CBT, logging, reports, and voice shell behavior.
- Compile check catches syntax/import-time issues.
- Mock E2E verifies control flow without loading the model.
- Voice I/O trace verifies the `record.csv` bridge without using real audio hardware.
- Voice smoke dry-run verifies command configuration, Piper model path, and local tools without recording or playback.
- Adapter smoke test loads the real local model and checks all six task adapters.
- Jetson benchmark records model load time, per-task latency, adapter switch latency, memory snapshots, and contract failures.
- Long-session reliability runs a deterministic mocked full 37-dimension session plus CBT and checks record locks, report rows, SQLite events, and memory snapshots.

On the target Jetson with a microphone and speaker attached, run the real voice smoke test:

```bash
python scripts/smoke_test_voice.py
```

Benchmark reports are written to `data/results/jetson_benchmark_*.json`:

```bash
CAITI_DEVICE_MAP=cuda:0 python scripts/benchmark_jetson.py --iterations 3 --warmup 1
```

Long-session reliability artifacts are written to `data/results/long_session_reliability_*`:

```bash
python scripts/long_session_reliability.py
```

## Running

Text session:

```bash
python LLM_therapist_Application.py
```

Flask text API:

```bash
python LLM_therapist_Application_server.py
curl -s http://127.0.0.1:8080/health
curl -sX POST 'http://127.0.0.1:8080/gpt' \
  -H 'Content-Type: application/json' \
  -d '{"user_input":"start","subject_ID":"8080"}'
```

Voice shell with persistent local Faster-Whisper STT and Piper TTS:

```bash
./scripts/run_voice_app.sh
```

The launcher uses `config.yaml` voice defaults:

```yaml
voice:
  stt_backend: "faster_whisper"
  tts_backend: "command"
  stt_whisper_model: "small.en"
  stt_record_seconds: 30
  stt_audio_device: "plughw:0,0"
  stt_device: "cpu"
  stt_compute_type: "int8"
  stt_beam_size: 5
  stt_best_of: 5
  stt_vad_filter: true
  stt_auto_stop: true
  stt_vad_detector: "auto"
  stt_vad_aggressiveness: 3
  stt_vad_chunk_ms: 30
  stt_silence_timeout_sec: 1.2
  stt_trailing_pad_sec: 0.4
  stt_no_speech_timeout_sec: 5.0
  tts_command: "python scripts/piper_tts_command.py --model models/piper/en_US-amy-medium.onnx --player aplay --sentence-silence 0.4"
  music_backend: "command"
  music_path: "assets/audio/music.wav"
  music_command: "aplay -q {path}"
```

The `faster_whisper` backend loads Whisper once and reuses it across turns. It
records until the user starts speaking and then stops after sustained silence
plus a short trailing pad, with `stt_record_seconds` acting as the maximum
capture window. `stt_vad_detector: "auto"` uses WebRTC VAD when `webrtcvad` is
installed and falls back to the older dBFS detector otherwise.

If you switch back to the `command` STT backend, STT commands must write
transcripts to stdout. TTS commands must read text from stdin and play audio
locally.

Waiting music is optional and plays while the model is loading and while CaiTI is thinking after a user response. It stops before TTS speaks and before STT listens, so music never enters the transcript or LLM prompts. Disable it with:

```bash
CAITI_MUSIC_BACKEND=off ./scripts/run_voice_app.sh
```

To temporarily fall back to console input/output:

```bash
CAITI_STT_BACKEND=console CAITI_TTS_BACKEND=console python LLM_therapist_Voice_Application.py
```

To override the tested Jetson USB microphone device:

```bash
CAITI_STT_AUDIO_DEVICE=plughw:1,0 ./scripts/run_voice_app.sh
```

To debug STT quality without running the full CaiTI session:

```bash
python scripts/faster_whisper_stt_command.py \
  --model small.en \
  --record-seconds 30 \
  --audio-device plughw:0,0 \
  --stt-device cpu \
  --compute-type int8 \
  --beam-size 5 \
  --best-of 5 \
  --vad-filter \
  --auto-stop \
  --debug-audio \
  --save-wav data/results/stt_quality_latest.wav
```

## Runtime Outputs

- `data/record.csv`: local question/response exchange file used by console, server, and voice shells.
- `data/logs/session_events.sqlite3`: structured event log for LLM calls and key flow events.
- `data/results/Report_${subject_id}.csv`: final report with `Score`, `Responses`, `Analysis`.
- `data/results/Notes_${subject_id}.csv`: auxiliary notes output.
- `data/q_tables/item_qtable_${subject_id}.csv`: learned item Q-table.

## Important Constraints

Do not change these without an explicit paper-consistency decision:

- Do not replace Q-learning with LLM ranking or embeddings.
- Do not add, remove, rename, or reorder the 37 dimensions.
- Do not change score granularity beyond `0/1/2`.
- Do not merge the task adapters or role-specific modules.
- Do not feed audio or emotion labels into LLM prompts.
- Do not add other therapy frameworks to the conversation flow.
- Do not redesign the session report schema.

## License

This code is provided for research and educational use without warranty. Check with the repository owner before commercial use.
