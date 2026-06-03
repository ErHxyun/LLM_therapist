# CaiTI Jetson Deployment Guide

This guide describes the validated local deployment path for CaiTI on Jetson
Orin Nano. It is a runbook for getting from a fresh project checkout to a
working text and voice session without cloud APIs.

## Validated Device

Current validated target:

```text
Device: Jetson Orin Nano / aarch64
Jetson Linux: L4T R36.4.3
Kernel: 5.15.148-tegra
Python: 3.10.12
PyTorch CUDA: 12.6
GPU name from PyTorch: Orin
```

Validated Python package versions:

```text
torch==2.8.0
transformers==5.5.3
peft==0.19.1
accelerate==1.13.0
bitsandbytes==0.48.0.dev0+ff389db
faster-whisper==1.2.1
piper-tts==1.4.2
soundfile==0.13.1
ctranslate2==4.7.2
onnxruntime==1.23.2
numpy==1.26.1
pandas==1.3.5
```

## Deployment Constraints

- Do not use OpenAI or other cloud APIs for this local deployment.
- Do not install `openai-whisper` or the `whisper` package alias on Jetson.
- Do not feed audio metadata, emotion labels, or STT confidence into LLM prompts.
- Keep the CaiTI paper flow unchanged: screening -> Response Analyzer -> R-V -> CBT.
- Use `CAITI_DEVICE_MAP=cuda:0` on the validated Jetson runtime.

## Project Checkout

Run from the project root:

```bash
cd /home/xiyun/Desktop/Projects/LLM_therapist
```

If deploying elsewhere, keep all commands rooted at the repository directory.

## Python Environment

The easiest project-level install path is:

```bash
conda env create -f environment_upgradable.yml
conda activate LLM_therapist
```

On JetPack 6 / CUDA 12.6 aarch64, install Jetson-compatible PyTorch and
bitsandbytes from the Jetson AI Lab index:

```bash
python -m pip install --user --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126 torch
python -m pip install --user transformers==5.5.3 peft accelerate pandas numpy flask flask-cors coloredlogs word2number 'Pillow>=10'
python -m pip install --user --no-cache-dir --force-reinstall --no-deps \
  --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126 bitsandbytes
python -m pip install --user 'faster-whisper>=1.0.0,<2.0.0' webrtcvad piper-tts soundfile
```

Check the critical runtime packages:

```bash
python - <<'PY'
import importlib.metadata as m
for pkg in ["torch", "transformers", "peft", "accelerate", "bitsandbytes", "faster-whisper", "webrtcvad", "piper-tts"]:
    print(f"{pkg}=={m.version(pkg)}")
PY
python -m bitsandbytes
```

## Model Setup

Default `config.yaml` expects the Hugging Face bundle:

```yaml
local_llm:
  model_id: "xxue752/caiti_best_model"
  base_subdir: "compressed_model_int4"
  tokenizer_id: "xxue752/caiti_best_model"
  tokenizer_subdir: "compressed_model_int4"
  device_map: "cuda:0"
```

Expected bundle layout when deploying offline:

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

For an offline local model directory:

```bash
export CAITI_MODEL_ID=/path/to/caiti_best_model
export CAITI_TOKENIZER_ID=/path/to/caiti_best_model
export CAITI_BASE_SUBDIR=compressed_model_int4
export CAITI_TOKENIZER_SUBDIR=compressed_model_int4
export CAITI_DEVICE_MAP=cuda:0
```

## Voice Assets

The validated Piper voice files are local device assets and are ignored by git:

```text
models/piper/en_US-amy-medium.onnx
models/piper/en_US-amy-medium.onnx.json
```

The validated file size for the `.onnx` model is about 61 MB.

Optional background music lives under `assets/audio/`:

```text
assets/audio/music.wav
assets/audio/fireplace.wav
assets/audio/seawaves.wav
```

Audio files in this directory are ignored by git because they are local runtime
assets. The default background backend uses `mpv` IPC so the music can stay
alive, lower volume during voice activity, and recover after pause/resume.

## Audio Devices

Validated capture device:

```text
card 0: USB PnP Sound Device
device 0: USB Audio
CAITI STT device argument: plughw:0,0
```

Validated playback route:

```text
aplay available at /usr/bin/aplay
piper available at /home/xiyun/.local/bin/piper
```

This Jetson also sees a USB playback device:

```text
card 3: UACDemoV1.0
device 0: USB Audio
```

If audio hardware changes, inspect devices again:

```bash
arecord -l
aplay -l
```

Then override only the STT audio device:

```bash
CAITI_STT_AUDIO_DEVICE=plughw:1,0 ./scripts/run_voice_app.sh
```

## Voice Configuration

The validated defaults live in `config.yaml`:

```yaml
voice:
  stt_backend: "faster_whisper"
  tts_backend: "command"
  stt_command: "python scripts/faster_whisper_stt_command.py --model small.en --record-seconds 30 --audio-device plughw:0,0 --stt-device cpu --compute-type int8 --beam-size 5 --best-of 5 --vad-filter --auto-stop --vad-detector auto --vad-aggressiveness 3 --trailing-pad-sec 0.4 --no-speech-timeout-sec 5"
  stt_whisper_model: "small.en"
  stt_record_seconds: 30
  stt_sample_rate: 16000
  stt_channels: 1
  stt_audio_device: "plughw:0,0"
  stt_device: "cpu"
  stt_compute_type: "int8"
  stt_beam_size: 5
  stt_best_of: 5
  stt_language: "en"
  stt_initial_prompt: "The speaker is answering CaiTI daily functioning screening questions in English. Common words include doctor, therapist, case manager, medication, mood, sleep, appetite, hygiene, chores, work, school, family, alcohol, drugs, safety, and daily life."
  stt_vad_filter: true
  stt_auto_stop: true
  stt_vad_detector: "auto"
  stt_vad_aggressiveness: 3
  stt_vad_chunk_ms: 30
  stt_silence_threshold_dbfs: -45
  stt_silence_timeout_sec: 1.2
  stt_trailing_pad_sec: 0.4
  stt_min_speech_seconds: 0.25
  stt_min_record_seconds: 1.0
  stt_no_speech_timeout_sec: 5.0
  tts_command: "python scripts/piper_tts_command.py --model models/piper/en_US-amy-medium.onnx --player aplay --sentence-silence 0.25"
  stt_timeout_sec: 120
  tts_timeout_sec: 60
  empty_transcript_retries: 2
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
  tts_backend: "command"
  tts_command: "python scripts/piper_tts_command.py --model models/piper/en_US-lessac-medium.onnx --player aplay --length-scale 1.1 --sentence-silence 0.4"
  fallback_to_primary_tts: true
  screening_enabled: true
  breathing_enabled: true
  mindfulness_enabled: true
  max_seconds: 45
  max_screening_items_per_turn: 1
  trigger_min_user_speech_sec: 10.0
  trigger_min_interval_turns: 2
  trigger_probability: 0.5
  cooldown_turns: 1
  persist_results: true
  db_path: ""
  results_json_path: "data/phq_gad_results.json"
```

The `faster_whisper` backend is persistent: the Whisper model is loaded once
and reused across turns. The app also starts an STT warm-up thread while the
CaiTI LLM is loading, so the first microphone turn does not pay the full STT
model load cost after the first spoken question.

STT auto-stop behavior:

- `stt_record_seconds` is the maximum capture window, currently 30 seconds.
- Recording stops early after speech starts and then stays below
  `stt_silence_threshold_dbfs` for `stt_silence_timeout_sec` seconds.
- `stt_no_speech_timeout_sec` is the time CaiTI waits for the user to begin
  speaking after TTS finishes, currently 5 seconds.
- This keeps short answers fast while allowing longer answers without the old
  fixed 8-second cutoff.

With `music_backend: "mpv"`, background music starts during model loading,
keeps looping through the session, lowers volume during CaiTI TTS and user STT,
and resumes after the button pause. The optional music-mode button cycles
`music.wav` -> `fireplace.wav` -> `seawaves.wav` -> off without changing main
session or intermission state. Install `mpv` first:

```bash
sudo apt install mpv
```

Set `music_backend: "command"` to use the older `aplay` waiting-music behavior
that stops music before TTS/STT and restarts it while CaiTI is thinking.

The intermission layer runs only while CaiTI is thinking after a user answer.
It is deliberately separate from the paper pipeline: it does not write to
`record.csv`, does not call the LLM, and does not update RL/CBT scores or
reports. PHQ-2/GAD-4 check-ins are framed as private, optional mini-tasks and
are stored in the structured SQLite intermission tables plus
`data/phq_gad_results.json`, separate from the main CaiTI question/response
record. By default, mini-tasks only start after a
main-session user answer with at least 10 seconds of captured speech. Eligible
long-answer turns increment a counter; the counter must be greater than 2, then
the final trigger uses `trigger_probability` randomness. If an activity runs,
it cools down for one turn. The configured intermission voice comes from
`intermission.tts_command`; if that model is not installed and
`fallback_to_primary_tts` is true, the app falls back to the primary CaiTI voice.

Final/system TTS messages use a no-response record marker internally. The voice
loop speaks those messages without collecting another STT turn, and the voice
app drains the final TTS before stopping the music and exiting. This prevents
the final closing sentence from being clipped.

Disable it for debugging:

```bash
CAITI_MUSIC_BACKEND=off ./scripts/run_voice_app.sh
```

## Validation Order

Run these in order. Stop at the first failure and debug that layer before
continuing.

```bash
python -m unittest discover -s tests -q
python -m compileall src scripts tests LLM_therapist_Application.py LLM_therapist_Voice_Application.py LLM_therapist_Application_server.py
python scripts/trace_mock_e2e.py
python scripts/trace_voice_io.py
python scripts/smoke_test_voice.py --dry-run
python scripts/smoke_test_voice.py
CAITI_DEVICE_MAP=cuda:0 python scripts/smoke_test_adapters.py
CAITI_DEVICE_MAP=cuda:0 python scripts/benchmark_jetson.py --iterations 3 --warmup 1
python scripts/long_session_reliability.py
```

Validated recent results:

```text
unit tests: 94 tests OK
compileall: application, src, scripts, and tests OK
voice smoke: dry-run config, Piper playback, and Faster-Whisper auto-stop microphone STT OK
adapter smoke: all six adapter contracts OK
benchmark one-iteration load time: about 55s
long-session reliability: 37/37 dimensions scored, CBT success, report 37 rows
```

## Running CaiTI

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

Voice session:

```bash
./scripts/run_voice_app.sh
```

Persistent local LLM server mode:

```bash
# Terminal 1: keep the model loaded
python scripts/run_llm_server.py

# Terminal 2: run the voice app against the warm model process
CAITI_LLM_SERVER_URL=http://127.0.0.1:8890 ./scripts/run_voice_app.sh
```

Check server readiness:

```bash
curl http://127.0.0.1:8890/health
```

This avoids reloading the base model and adapters every time the voice app is
restarted. It is still local-only; the server does not alter the paper logic.

Node.js hardware/status dashboard:

```bash
npm run monitor
```

Open `http://127.0.0.1:8787`. This reads the Python monitor at
`http://127.0.0.1:8765` and does not write GPIO state.

Console fallback for debugging:

```bash
CAITI_STT_BACKEND=console CAITI_TTS_BACKEND=console python LLM_therapist_Voice_Application.py
```

## Runtime Artifacts

Primary runtime outputs:

```text
data/record.csv
data/logs/session_events.sqlite3
data/results/Report_${subject_id}.csv
data/results/Notes_${subject_id}.csv
data/q_tables/item_qtable_${subject_id}.csv
```

Validation outputs:

```text
data/results/jetson_benchmark_*.json
data/results/long_session_reliability_*/
```

## Troubleshooting

### No TTS Sound

Check that Piper and `aplay` exist:

```bash
which piper
which aplay
python scripts/smoke_test_voice.py --skip-stt
```

If HDMI or USB audio routing changed, inspect:

```bash
aplay -l
```

### Waiting Music Does Not Play

Check the configured asset and command:

```bash
ls -lh assets/audio/music.wav
python scripts/smoke_test_voice.py --dry-run
```

Music starts only while CaiTI is thinking between turns, not during the first
question, TTS playback, or STT recording. To test the raw player command:

```bash
aplay -q assets/audio/music.wav
```

### Microphone Records Silence

List capture devices:

```bash
arecord -l
```

Try a direct capture:

```bash
arecord -q -f S16_LE -r 16000 -c 1 -d 6 -D plughw:0,0 /tmp/caiti_mic_test.wav
```

If the USB mic card changed, update `--audio-device`.

### STT Stops Too Early or Waits Too Long

The voice app records until the user starts speaking and then falls below the
configured silence threshold long enough to count as done.

If CaiTI cuts off the end of an answer, make the stop rule more patient:

```bash
CAITI_STT_SILENCE_TIMEOUT_SEC=1.6 ./scripts/run_voice_app.sh
```

If CaiTI waits too long after the user finishes speaking, make the stop rule
more responsive:

```bash
CAITI_STT_SILENCE_TIMEOUT_SEC=0.9 ./scripts/run_voice_app.sh
```

If CaiTI waits too long before deciding nobody started speaking, use the
v1.1-style five-second open-mic window:

```bash
CAITI_STT_NO_SPEECH_TIMEOUT_SEC=5 ./scripts/run_voice_app.sh
```

For very quiet microphones, lower the silence threshold, for example `-50`.
For noisy rooms, raise it toward `-40`. Keep `stt_record_seconds` as the maximum
capture window rather than the normal turn length. `CAITI_STT_VAD_DETECTOR=auto`
uses WebRTC VAD when `webrtcvad` is installed, otherwise it falls back to the
dBFS detector.

### Faster-Whisper Works Slowly

The validated STT config uses CPU int8 with a stronger local model and beam search:

```bash
--stt-device cpu --compute-type int8 --model small.en --beam-size 5 --best-of 5 --vad-filter --auto-stop --vad-detector auto
```

The main voice app uses a persistent `faster_whisper` backend, so the model is
loaded once and reused. The standalone command is still useful for debugging,
but it reloads Whisper each time it is executed.

If quality is still poor and latency is acceptable, try `--model medium.en`. If
latency becomes too high, fall back to `--model base.en` or reduce
`CAITI_WHISPER_BEAM_SIZE` and `CAITI_WHISPER_BEST_OF` for a faster run.

For STT quality debugging, run a single recording with audio metrics and keep
the WAV artifact:

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
  --vad-detector auto \
  --debug-audio \
  --save-wav data/results/stt_quality_latest.wav
```

Healthy speech capture should usually have a peak comfortably above silence
without clipping. If `peak` is near `0.0dBFS`, lower input gain; if `rms` is
very low, move closer to the mic or increase gain.

### bitsandbytes or CUDA Fails

Check the installed wheel and CUDA visibility:

```bash
python -m bitsandbytes
python - <<'PY'
import torch
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

On Jetson, prefer the Jetson AI Lab package index for `torch` and
`bitsandbytes`.

### Model Download or Rate Limit Problems

If Hugging Face rate limits or network access is unreliable, download the model
bundle once and run with local paths:

```bash
export CAITI_MODEL_ID=/path/to/caiti_best_model
export CAITI_TOKENIZER_ID=/path/to/caiti_best_model
```

### `record.csv` Appears Stuck

First run the mocked bridge and long-session checks:

```bash
python scripts/trace_voice_io.py
python scripts/long_session_reliability.py
```

For a fresh local run, reinitialize through the normal entrypoint. The
application calls `init_record()` at startup.

### Memory Pressure

Use the benchmark and system tools:

```bash
free -h
tegrastats
CAITI_DEVICE_MAP=cuda:0 python scripts/benchmark_jetson.py --iterations 1 --warmup 0
```

If memory is tight, close browsers and other GPU workloads before loading the
model.
