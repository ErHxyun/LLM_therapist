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
|   |-- benchmark_stt.py                  # Fixed-audio Faster-Whisper STT benchmark
|   |-- faster_whisper_stt_command.py     # Local microphone recording + Faster-Whisper STT command
|   |-- long_session_reliability.py       # Mocked full 37-dimension + CBT reliability run
|   |-- piper_tts_command.py              # Local Piper TTS command
|   |-- run_llm_server.py                 # Persistent local LLM server
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

Optional background music can be stored at:

```text
assets/audio/music.wav
assets/audio/fireplace.wav
assets/audio/seawaves.wav
```

Audio files under `assets/audio/` are ignored by git, except `.gitkeep`, because they are local runtime assets. The background music backend uses `mpv` IPC so music can keep playing while CaiTI lowers it during TTS and STT.

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
- STT benchmark compares Faster-Whisper latency and transcripts on fixed WAV files without running the CaiTI session pipeline.
- Long-session reliability runs a deterministic mocked full 37-dimension session plus CBT and checks record locks, report rows, SQLite events, and memory snapshots.

On the target Jetson with a microphone and speaker attached, run the real voice smoke test:

```bash
python scripts/smoke_test_voice.py
```

Benchmark reports are written to `data/results/jetson_benchmark_*.json`:

```bash
CAITI_DEVICE_MAP=cuda:0 python scripts/benchmark_jetson.py --iterations 3 --warmup 1
```

STT benchmark audio should be fixed WAV files, usually saved under
`data/benchmark/stt/`. For example, record a standard answer once:

```bash
python scripts/faster_whisper_stt_command.py \
  --model small.en \
  --record-seconds 30 \
  --audio-device plughw:0,0 \
  --stt-device cpu \
  --compute-type int8 \
  --beam-size 1 \
  --best-of 1 \
  --vad-filter \
  --auto-stop \
  --save-wav data/benchmark/stt/sleep_answer.wav
```

Then compare STT settings on the same audio without touching the main session:

```bash
python scripts/benchmark_stt.py \
  --audio-dir data/benchmark/stt \
  --model small.en \
  --model base.en \
  --vad-filter-mode both \
  --iterations 3
```

Reports are written to `data/results/stt_benchmark_*.jsonl` and `.csv`.

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

### Persistent LLM Server

By default, each Python process loads the CaiTI model into memory. To keep the
model warm across voice-app restarts, run the local LLM in one terminal:

```bash
python scripts/run_llm_server.py
```

Wait until it logs that the server is ready, then start the voice app from a
second terminal:

```bash
CAITI_LLM_SERVER_URL=http://127.0.0.1:8890 ./scripts/run_voice_app.sh
```

Health check:

```bash
curl http://127.0.0.1:8890/health
```

The server is only an inference cache/process boundary. It does not change the
RL, response-analysis, R-V, CBT, STT, TTS, button, or LED logic. Stop the server
with `Ctrl+C` when you want to free GPU memory.

The launcher uses `config.yaml` voice defaults:

```yaml
voice:
  stt_backend: "faster_whisper"
  tts_backend: "persistent_piper"
  stt_whisper_model: "small.en"
  stt_record_seconds: 30
  stt_audio_device: "plughw:0,0"
  stt_device: "cpu"
  stt_compute_type: "int8"
  stt_beam_size: 1
  stt_best_of: 1
  stt_vad_filter: true
  stt_auto_stop: true
  stt_vad_detector: "auto"
  stt_vad_aggressiveness: 3
  stt_vad_chunk_ms: 30
  stt_silence_timeout_sec: 1.2
  stt_trailing_pad_sec: 0.4
  stt_no_speech_timeout_sec: 5.0
  tts_command: "python scripts/piper_tts_command.py --model models/piper/en_US-amy-medium.onnx --player aplay --sentence-silence 0.25"
  music_backend: "mpv"
  music_path: "assets/audio/music.wav"
  music_fireplace_path: "assets/audio/fireplace.wav"
  music_seawaves_path: "assets/audio/seawaves.wav"
  music_command: "aplay -q {path}"
  music_volume_percent: 80
  music_duck_volume_percent: 40
  music_ipc_path: "/tmp/caiti_mpv_music.sock"

intermission:
  enabled: true
  tts_backend: "persistent_piper"
  tts_command: "python scripts/piper_tts_command.py --model models/piper/en_US-lessac-medium.onnx --player aplay --length-scale 1.1 --sentence-silence 0.6"
  fallback_to_primary_tts: true
  screening_enabled: true
  breathing_enabled: true
  mindfulness_enabled: true
  max_seconds: 45
  max_screening_items_per_turn: 1
  trigger_min_user_speech_sec: 10.0
  trigger_min_interval_turns: 1
  trigger_probability: 1.0
  cooldown_turns: 0
  persist_results: true
  db_path: ""
  results_json_path: "data/phq_gad_results.json"

emotion:
  enabled: true
  service_url: "http://127.0.0.1:8000/analyze"
  user_id: "${subject_id}"
  language: "en"
  timeout_sec: 60
  results_jsonl_path: "data/emotion/results.jsonl"
  audio_dir: "data/emotion/audio"
  keep_audio: false
  assist_followup_enabled: true
  assist_wait_timeout_sec: 1.5
  assist_late_followup_window_sec: 20.0
  assist_min_confidence: 50
  assist_risk_threshold: 55
  assist_light_risk_threshold: 45
```

The `faster_whisper` backend loads Whisper once and reuses it across turns. It
records until the user starts speaking and then stops after sustained silence
plus a short trailing pad, with `stt_record_seconds` acting as the maximum
capture window. `stt_vad_detector: "auto"` uses WebRTC VAD when `webrtcvad` is
installed and falls back to the older dBFS detector otherwise.
STT code is split under `src/voice/stt/`: command/console adapters live in
`command.py`, the persistent Faster-Whisper backend lives in
`faster_whisper.py`, and recorder helpers live in `recorder.py`.

If you switch back to the `command` STT backend, STT commands must write
transcripts to stdout. TTS commands must read text from stdin and play audio
locally. The bundled Piper command validates both the `.onnx` voice and its
`.onnx.json` config before playback, then falls back to `espeak-ng` when Piper
cannot synthesize audio.
When launched through the command-line script, Piper TTS keeps synthesized WAVs
under `data/cache/tts`, keyed by text, voice model, and speaking parameters, so
repeated prompts can replay without resynthesizing.
The `persistent_piper` backend loads the Python Piper voice model once per role
and reuses it across turns, while still honoring the configured Piper command's
model, player, length scale, sentence silence, and cache settings.
Voice role selection is centralized in `src/voice/tts/router.py`: the main and
CBT roles use the primary TTS, while intermission prefers its configured voice
and falls back to the primary TTS when allowed.

Background music is optional. With `music_backend: "mpv"`, music loops through the whole session, lowers during CaiTI TTS and user STT, pauses on the button pause, and resumes from the same playback process when the session continues. The optional music-mode button cycles `music.wav` -> `fireplace.wav` -> `seawaves.wav` -> off without changing the main session or intermission state. Install `mpv` on the Jetson before using this backend:

```bash
sudo apt install mpv
```

Disable it with:

```bash
CAITI_MUSIC_BACKEND=off ./scripts/run_voice_app.sh
```

The intermission layer runs only while CaiTI is thinking after a user answer.
It is isolated from the paper pipeline: it does not write to `record.csv`, does
not call the LLM, and does not update RL/CBT scores or reports. PHQ-2/GAD-4
check-ins are framed as private, optional mini-tasks and are stored in the
structured SQLite intermission tables plus `data/phq_gad_results.json`,
separate from the main CaiTI question/response record. By default, mini-tasks
only start after a main-session user answer with at least 10 seconds of
captured speech. The default `trigger_min_interval_turns: 1` means the first
eligible long-answer turn is counted, and the second eligible long-answer
thinking gap can use the waiting time. The
intermission voice is configured separately;
until the male Piper model is present, `fallback_to_primary_tts: true` keeps
the app audible with CaiTI's primary voice.

The emotion module is a side-channel for the external
`xxue752-nz/emo_module` FastAPI service. With `emotion.enabled: true`, each
non-empty STT turn is copied to `data/emotion/audio`, sent asynchronously with
the transcript to `/analyze`, and appended to `data/emotion/results.jsonl`.
The external service must be able to read the local `audio_file_path` sent in
the request. Emotion outputs never enter LLM prompts, `record.csv`, CBT
scoring, or final reports. When `assist_followup_enabled` is on, the main
session waits up to `assist_wait_timeout_sec` for a reliable emotion result. If
it arrives in time, a mismatch or strong strained vocal cue can add one gentle
follow-up during the current question flow, but it never changes the text-based
`0/1/2` score. For high-content scores, a moderate emotion risk can also add a
short meaning-check follow-up when the tone may be sarcastic, exaggerated, or
otherwise worth clarifying. If the result arrives after that wait but within
`assist_late_followup_window_sec`, the follow-up is asked as its own turn
before the next screening question, followed by a short transition message.
Reflection-validation acceptance also uses a short standalone transition
message instead of prepending validation text to the next question. If the
service is down, too slow, low-confidence, or reports poor audio quality, the
main session continues without
emotion-assisted follow-up and the side-channel records the row. Completed
emotion calls also log a compact score summary with latency, credibility risk,
confidence, audio/context emotion labels, consistency flags, and key audio/text
scores.

To temporarily fall back to console input/output:

```bash
CAITI_STT_BACKEND=console CAITI_TTS_BACKEND=console python LLM_therapist_Voice_Application.py
```

To override the tested Jetson USB microphone device:

```bash
CAITI_STT_AUDIO_DEVICE=plughw:1,0 ./scripts/run_voice_app.sh
```

### Node.js Status Dashboard

The Python voice app exposes the source runtime state at:

```text
http://127.0.0.1:8765/status
```

For a separate Node.js dashboard/proxy, start the voice app first, then run:

```bash
npm run monitor
```

Open:

```text
http://127.0.0.1:8787
```

The Node process only reads the Python monitor. It does not control GPIO or
change the therapy workflow. Useful endpoints:

```bash
curl http://127.0.0.1:8787/health
curl http://127.0.0.1:8787/api/status
```

To change ports:

```bash
CAITI_NODE_MONITOR_PORT=8790 CAITI_MONITOR_URL=http://127.0.0.1:8765 npm run monitor
```

### Remote Monitor on a Real Domain

To publish the research monitor on a domain such as
`https://monitor.example.com`, use the Cloudflare Tunnel deployment under:

- [deploy/cloudflared/README.md](deploy/cloudflared/README.md)

That path is recommended over exposing port `8765` directly because the monitor
can display participant IDs, answers, scores, emotion summaries, and session
history. The deployment keeps the origin on `127.0.0.1:8765`, publishes it
through `cloudflared`, and protects access with Cloudflare Access login
policies.

To debug STT quality without running the full CaiTI session:

```bash
python scripts/faster_whisper_stt_command.py \
  --model small.en \
  --record-seconds 30 \
  --audio-device plughw:0,0 \
  --stt-device cpu \
  --compute-type int8 \
  --beam-size 1 \
  --best-of 1 \
  --vad-filter \
  --auto-stop \
  --debug-audio \
  --save-wav data/results/stt_quality_latest.wav
```

## Runtime Outputs

- `data/record.csv`: local question/response exchange file used by console, server, and voice shells.
- `data/logs/session_events.sqlite3`: structured event log for LLM calls, key flow events, and isolated intermission PHQ/GAD item records.
- `data/phq_gad_results.json`: JSON mirror of private intermission PHQ-2/GAD-4 item records and per-scale totals.
- `data/emotion/results.jsonl`: optional emotion side-channel responses from the external emotion service.
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
