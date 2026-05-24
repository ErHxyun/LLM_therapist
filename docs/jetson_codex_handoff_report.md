# CaiTI Jetson Codex Handoff Report

本文档给 Jetson Orin Nano 本地 VSCode/Codex 使用，目的是让新的本地 Codex thread 快速接上当前工作，不要重新发散设计。

## 1. 当前代码状态

- Repository: `https://github.com/ErHxyun/LLM_therapist.git`
- Branch: `paper-baseline-local-adapters`
- Latest commit: `a601fec Integrate local CaiTI adapters and voice IO`
- Base commit: `3b664eb08b397dc2a97b7caac56e9581b54c47e1`
- Working target: Jetson Orin Nano 8GB unified memory，本地部署 CaiTI，不使用 OpenAI 或云 API。

建议在 Jetson 上直接拉这个 branch：

```bash
git clone https://github.com/ErHxyun/LLM_therapist.git
cd LLM_therapist
git checkout paper-baseline-local-adapters
```

## 2. 论文一致性约束

所有后续修改必须严格跟随 Nie et al. 2024 CaiTI paper (`arXiv:2403.10779v1`)。

关键约束：

- Section 4 / 4.2: CaiTI conversation flow 固定为 Q-learning question selection -> Response Analyzer -> R-V process -> CBT。
- Section 4.2: Questioner 必须使用 Epsilon-Greedy Q-learning；不能换成 LLM ranking。
- Section 5: 必须保留 task-specific LLM modules，不能把 Reasoner / Guide / Validator 合并成一个大 prompt。
- Appendix A: 37 dimensions 固定，不能新增、删除、重命名、重排。
- Score 只能是 `0/1/2`。
- R-V 流程固定：ReflectiveSummarizer -> follow-up -> R-V Reasoner -> invalid 时 R-V Guide 回到主题并循环 -> valid 时 R-V Validator 做 empathic validation。
- CBT 固定为 3 stages，每个 stage 最多 3 attempts；失败后结束 CBT 并建议寻求专业帮助。
- LLM 只做文本 semantic analysis；audio/emotion output 不能进入 LLM prompt。
- Session report 格式固定为 `Score, Responses, Analysis`。

## 3. 已完成的主要改动

### 3.1 Local LLM runtime

新增/修改：

- `src/local_llm/runtime.py`
- `src/local_llm/routing.py`
- `src/local_llm/types.py`
- `src/utils/llm_client.py`

功能：

- 使用 `xxue752/caiti_best_model`。
- base model: `compressed_model_int4/`
- base 只加载一次。
- 6 个 LoRA adapters 预加载后通过 `set_adapter` 切换。
- base model generation 使用 `tokenizer.apply_chat_template`，适配 Llama 3.2 chat format。
- adapter calls 使用 adapter repo 中对应的 plain prompt continuation 风格。

### 3.2 Adapter wiring

已接入 6 个 adapters：

- `task1_response_analyzer`
  - paper role: Response Analyzer
  - 输入格式：`{"in":"...", "res":`
  - 输出标准化为 `(Dimension, Score)`，如 `('weight', 2)`。

- `task2_general_response`
  - paper role: General Response classifier
  - 用于 Yes/No/Maybe/Question/Stop。
  - 输入格式：`Response: ...`
  - 数字 fallback: `1 -> Yes`, `2 -> No`, `3 -> Maybe`, `4 -> Question`, `5 -> Stop`。

- `task3_rv_reasoner`
  - paper role: R-V Reasoner
  - 输入格式：
    `{"Topic": "...", "Original Response": "...", "Follow Up Response": "..."}`
  - 输出标准化为 `DECISION: 0/1`。

- `task4_cbt_stage1`
  - paper role: CBT Stage 1 Reasoner
  - 输入格式：`STATEMENT: ...; UNHELPFUL_THOUGHTS: ...;`

- `task4_cbt_stage2`
  - paper role: CBT Stage 2 Reasoner
  - 输入格式：`STATEMENT: ...; UNHELPFUL_THOUGHTS: ...; CHALLENGE: ...;`

- `task4_cbt_stage3`
  - paper role: CBT Stage 3 Reasoner
  - 输入格式：`STATEMENT: ...; UNHELPFUL_THOUGHTS: ...; CHALLENGE: ...; REFRAME: ...;`

Task4 输出也统一标准化为 `DECISION: 0/1`。

### 3.3 Output contracts

新增：

- `src/utils/llm_output_contracts.py`

功能：

- 强制校验 score 只允许 `{0,1,2}`。
- 强制校验 Reasoner output 只允许 `DECISION: 0/1`。
- 强制校验 task2 只允许 `Yes/No/Maybe/Question/Stop`。
- 处理 adapter label 到本地 label 的映射，例如：
  - `31_motivation -> work_motivation`
  - `19_family_relationship -> family`

### 3.4 R-V process

主要文件：

- `src/questioner.py`
- `src/reflection_validation.py`

当前实现：

- 当 Response Analyzer 任一 segment 得到 score `2`，触发 R-V。
- ReflectiveSummarizer 生成 third-person/simple reflection + follow-up。
- task3 R-V Reasoner 判断 follow-up 是否 valid。
- invalid 时调用 R-V Guide，把用户拉回 original response / original dimension。
- valid 时调用 R-V Validator，生成 affective reflection, affirmation, empathic validation, support。

重要设计收敛：

- R-V Guide 是 deterministic redirect-only，不给临床建议。
- R-V Guide 遇到严重安全风险时，只建议寻求专业帮助/紧急帮助/988 等，不做治疗建议。
- R-V Validator 使用 bounded generation，不是死模板。
- Validator 不应发问，不应给 medical/diet/exercise/medication/treatment advice，不应编造用户没有表达的情绪。

### 3.5 CBT process

主要文件：

- `src/CBT.py`

当前实现：

- Stage 1: Recognize Negative Thoughts
- Stage 2: Challenge Negative Thoughts
- Stage 3: Reframe Thoughts
- 每个 stage 最多 3 attempts。
- 任何 stage 失败 3 次后，结束 CBT 并建议寻求专业帮助。
- task4 stage adapters 只做 validity/reasoning decision。
- Stage Guides 使用 base model + bounded prompt，不固定模板，但限制不越界。

### 3.6 Structured logging

新增：

- `src/utils/session_event_logger.py`

功能：

每个关键 turn 记录：

- timestamp
- session/task
- adapter used
- dimension
- score
- segment text
- question/follow-up
- raw LLM output
- normalized output
- metadata

默认 SQLite，用于 therapist review。

### 3.7 Session report

主要文件：

- `src/utils/io_question_lib.py`

报告格式已收敛到论文要求：

- `Score`
- `Responses`
- `Analysis`

不要 redesign report，不要增加 dashboard。

### 3.8 Paper consistency / dimension fix

已修复：

- `config.yaml`
  - `item_n_states: 39`
  - `epsilon: 0.9`
  - `alpha: 0.1`
  - `gamma: 0.9`
  - 37 dimensions + START + END。

- `data/libs/question_lib_v4.json`
  - dimension 31 label 从 `motivation` 修为 `work_motivation`。

- `src/handler_rl.py`
  - 明确 START/END state。
  - 旧的 38x38 q-table 会被忽略并重新初始化。

- `src/utils/rl_qtables.py`
  - mask 不再污染 learned q-table。
  - START/END 不会被当作 screening dimension。

### 3.9 STT/TTS voice shell

新增：

- `LLM_therapist_Voice_Application.py`
- `src/voice/backends.py`
- `src/voice/io_loop.py`
- `src/voice/music.py`
- `src/voice/sentence_stream.py`

设计：

- STT/TTS 是 I/O shell，不进入 LLM reasoning。
- STT: microphone/audio -> text。
- LLM pipeline 仍然只接收 text。
- TTS: generated text -> speech。
- 支持按句子边界 streaming TTS。
- 支持第一阶段 waiting music：model loading 时播放，也会在 user transcript 写入 `record.csv` 后、CaiTI 生成下一句前播放；每次 TTS/STT 前都会停止，避免进入 transcript 或 LLM prompt。
- 当前 Jetson 默认 backend 是 local command；调试时可用环境变量切回 `console`。
- `scripts/trace_voice_io.py` 可在不使用真实音频硬件、不加载 LLM 的情况下验证 `record.csv -> TTS -> STT -> record.csv` 协议。
- 参考 `NIcE-X-Lab/conversational_ai_therapist_smart_speaker` 的 `v1.1` 分支时，只采用运行方式：STT 用 Faster-Whisper，本地 TTS 用 Piper；不要照搬其较重的 service/audio 架构。
- 当前 repo 新增两个轻量 command adapter：
  - `scripts/faster_whisper_stt_command.py`: `arecord` 录制本地 WAV，然后 Faster-Whisper 转写，只把 transcript 输出到 stdout。
  - `scripts/piper_tts_command.py`: stdin 读文本，Piper 生成临时 WAV，然后用 `aplay`/`paplay` 播放。

`config.yaml` 中：

```yaml
voice:
  stt_backend: "command" # console | command
  tts_backend: "command" # console | command
  stt_command: "python scripts/faster_whisper_stt_command.py --model base.en --record-seconds 8 --audio-device plughw:0,0 --stt-device cpu --compute-type int8 --beam-size 5 --best-of 5 --vad-filter"
  tts_command: "python scripts/piper_tts_command.py --model models/piper/en_US-amy-medium.onnx --player aplay --sentence-silence 0.4"
  stt_timeout_sec: 120
  tts_timeout_sec: 60
  empty_transcript_retries: 2
  music_backend: "command" # off | command
  music_path: "assets/audio/music.wav"
  music_command: "aplay -q {path}"
```

也可以用环境变量覆盖：

```bash
export CAITI_STT_BACKEND=command
export CAITI_STT_COMMAND="your-local-stt-command"
export CAITI_TTS_BACKEND=command
export CAITI_TTS_COMMAND="your-local-tts-command"
```

Faster-Whisper + Piper 示例：

```bash
python -m pip install --user 'faster-whisper>=1.0.0,<2.0.0' piper-tts soundfile

export CAITI_STT_BACKEND=command
export CAITI_STT_COMMAND="python scripts/faster_whisper_stt_command.py --model base.en --stt-device cpu --compute-type int8 --record-seconds 8 --audio-device plughw:0,0 --beam-size 5 --best-of 5 --vad-filter"
export CAITI_TTS_BACKEND=command
export CAITI_TTS_COMMAND="python scripts/piper_tts_command.py --model models/piper/en_US-amy-medium.onnx --player aplay --sentence-silence 0.4"
export CAITI_MUSIC_BACKEND=command
export CAITI_MUSIC_PATH=assets/audio/music.wav
```

不要安装 `openai-whisper` 或 `whisper` alias；Jetson 8GB 上它们会拉入更重的 PyTorch 栈并挤占本地 LLM 内存。

要求：

- STT command 必须把 transcript 输出到 stdout。
- TTS command 必须从 stdin 读取文本并播放。
- STT/TTS 必须本地运行，不能调用云 API。
- Waiting music 是 voice I/O 层的环境音，不能参与 clinical flow、scoring、R-V、CBT，也不能进入 LLM prompt。

## 4. Jetson 本地适配步骤

### 4.1 创建环境

建议先使用 Python 3.10 或 3.11。Jetson 上 PyTorch/Transformers/bitsandbytes 组合可能需要按 JetPack 版本调整。

基础依赖：

```bash
pip install -U pip
pip install -r requirements.txt  # 如果不存在 requirements.txt，则参考 environment_upgradable.yml
```

当前 repo 主要依赖：

- `torch`
- `transformers==5.5.3`
- `Pillow>=10`
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

注意：Jetson/aarch64 上最大风险是 `bitsandbytes` 和 NF4 INT4 quantized loading。不要先做架构改造，先收集真实报错和内存数据。

### 4.2 模型路径

默认从 Hugging Face 读取：

```yaml
local_llm:
  model_id: "xxue752/caiti_best_model"
  base_subdir: "compressed_model_int4"
  tokenizer_id: "xxue752/caiti_best_model"
  tokenizer_subdir: "compressed_model_int4"
```

如果 Jetson 离线部署，建议先在 Jetson 或外部机器下载模型到本地目录，然后设置：

```bash
export CAITI_MODEL_ID=/path/to/caiti_best_model
export CAITI_TOKENIZER_ID=/path/to/caiti_best_model
export CAITI_BASE_SUBDIR=compressed_model_int4
export CAITI_TOKENIZER_SUBDIR=compressed_model_int4
```

目录应包含：

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

## 5. Jetson 首次验证顺序

请按顺序跑，遇到失败先停，不要跳到 full session。

### 5.1 Unit tests

```bash
python -m unittest discover -s tests -q
```

预期：80 tests OK。

### 5.2 Compile check

```bash
python -m compileall src scripts tests LLM_therapist_Voice_Application.py
```

预期：无 compile error。

### 5.3 Mock E2E trace

```bash
python scripts/trace_mock_e2e.py
```

用途：

- 不加载真实模型。
- 验证 CaiTI control flow：screening -> task1 -> R-V -> task3 -> guide/validator -> CBT task4。

### 5.4 Voice I/O dry-run

```bash
python scripts/trace_voice_io.py
```

用途：

- 不加载真实模型。
- 不使用麦克风或扬声器。
- 验证 CaiTI question -> TTS chunks -> STT transcript -> `record.csv` response bridge。

### 5.5 Real adapter smoke test

```bash
python scripts/smoke_test_adapters.py
```

用途：

- 加载 base + adapters。
- 分别验证 task1/task2/task3/task4 outputs。

这是 Jetson 最关键的一步。若失败，优先检查：

- `bitsandbytes`
- CUDA/JetPack/PyTorch ABI
- model path
- unified memory pressure
- adapter path
- `peft` version

当前 Jetson Orin Nano / JetPack 6 / CUDA 12.6 实测：

- 标准 PyPI `bitsandbytes==0.49.2` 会在真实生成时失败，错误包含 `named symbol not found`。
- Jetson AI Lab `bitsandbytes==0.48.0.dev0+ff389db` 通过 `python -m bitsandbytes` 诊断，并能完成真实 adapter 生成。
- `device_map: "auto"` 会触发 CPU/disk offload 报错；当前默认应使用 `device_map: "cuda:0"`。
- `transformers==4.53.3` 需要 tokenizer fallback，且真实 adapter 输出格式不稳定；模型包标记为 `transformers_version: 5.5.3`。
- 升级到 `transformers==5.5.3` 后需要 `Pillow>=10`，否则 `peft` 导入会因为 `PIL.Image.Resampling` 缺失失败。
- 在 `torch==2.8.0`、`transformers==5.5.3`、`peft==0.19.1`、`accelerate==1.13.0`、`bitsandbytes==0.48.0.dev0+ff389db` 下，真实 smoke test 已完成 6 个 adapter 调用，全部通过输出契约。
- 当前实测 load / latency：runtime load 约 54.7s；task1 首次调用含加载约 69.7s；后续 adapter 调用约 2.3s-3.9s；base 16-token generation 约 4.2s。

### 5.6 Text mode session

```bash
python LLM_therapist_Application.py
```

用途：

- 不接语音，先验证完整 text pipeline。

### 5.7 Voice shell session

默认 console STT/TTS：

```bash
python LLM_therapist_Voice_Application.py
```

接本地命令后：

```bash
export CAITI_STT_BACKEND=command
export CAITI_STT_COMMAND="your-local-stt-command"
export CAITI_TTS_BACKEND=command
export CAITI_TTS_COMMAND="your-local-tts-command"
python LLM_therapist_Voice_Application.py
```

当前 Jetson 镜像实测有 `spd-say`、`arecord`、`aplay`、PulseAudio `parec/pacat`。USB 麦克风可通过 `plughw:0,0` 录制为 16kHz mono WAV。可用本地 TTS 示例：

```bash
export CAITI_TTS_BACKEND=command
export CAITI_TTS_COMMAND="spd-say --wait --pipe-mode"
```

当前未检测到已安装的 local STT 包（如 faster-whisper/vosk）。接入 STT 时必须使用本地/offline command，并且只把最终 transcript 写到 stdout。

## 6. 当前未完成 TODO

### High priority

1. Jetson real adapter smoke test [done]
   - 记录 load time。
   - 记录 peak memory。
   - 记录 task1/task2/task3/task4 latency。
   - 记录 base generation latency。

2. 解决 Jetson `bitsandbytes` / NF4 INT4 loading 风险 [done]
   - 只有在实际失败并有错误信息后再改。
   - 不要提前换模型或换推理框架。

3. STT/TTS 本地后端选择与验证 [done]
   - STT command 必须 stdout 输出文本。
   - TTS command 必须 stdin 输入文本并播放。
   - 不得调用云 API。
   - 当前 TTS 使用 `scripts/piper_tts_command.py` 接 Piper，voice model 位于 `models/piper/en_US-amy-medium.onnx`。
   - STT 使用 `scripts/faster_whisper_stt_command.py` 接 Faster-Whisper；当前 USB mic 参数为 `--audio-device plughw:0,0`。
   - 可用 `python scripts/smoke_test_voice.py --dry-run` 检查配置；可用 `python scripts/smoke_test_voice.py` 做真实麦克风/扬声器 smoke test。

4. Voice E2E test [done]
   - CaiTI question -> TTS 播放。
   - User speech -> STT transcript。
   - transcript -> existing LLM pipeline。
   - R-V Validator / Guides / CBT 输出按句子 TTS。
   - 第一阶段 waiting music 已接入：`src/voice/music.py` + `config.yaml` 的 `music_backend/music_path/music_command`，在 model loading 和 CaiTI 思考时播放。
   - 已完成 dry-run `scripts/trace_voice_io.py`。
   - 已完成真实链路：Piper TTS 播放、USB mic 录音、Faster-Whisper 转写、本地 LLM adapter 分类、下一题/追问输出。

### Medium priority

5. Jetson benchmark script [done: `scripts/benchmark_jetson.py`]
   - model load time
   - adapter switch latency
   - per-task P50/P95
   - memory before/after generation
   - failure logs
   - 默认输出到 `data/results/jetson_benchmark_*.json`：
     ```bash
     CAITI_DEVICE_MAP=cuda:0 python scripts/benchmark_jetson.py --iterations 3 --warmup 1
     ```

6. Emotion module logging only
   - local path from Windows: `C:\Users\胡溪筼\Desktop\2026-7spring\CaiTi\emo_module`
   - Jetson 上如果接入，只能作为 metadata logging 或 RL reward metadata。
   - emotion label 不允许进入 LLM prompt。

7. Jetson deployment docs [done: `docs/jetson_deployment.md`]
   - JetPack version
   - PyTorch install command
   - CUDA/cuDNN versions
   - exact model cache path
   - exact STT/TTS backend command

### Low priority

8. Flask/server voice endpoint
   - 当前 Flask server 仍是 text API。
   - 若要 browser/mobile UI，可在不改 LLM logic 的前提下增加 audio endpoint。

9. Long-session reliability test [done: `scripts/long_session_reliability.py`]
   - 连续跑完整 37 dimensions + CBT。
   - 检查 record lock、SQLite logs、report output。
   - 当前实现使用真实 `HandlerRL().run()`，但用 deterministic LLM mocks 和 fake record.csv user，不加载真实模型。
   - 最近一次通过结果：37/37 dimensions scored，CBT success，`Question_Lock=0`，report 37 rows，task1 structured events 37 条，运行约 20 秒。
   - 默认输出到 `data/results/long_session_reliability_*`：
     ```bash
     python scripts/long_session_reliability.py
     ```

## 7. 不要做的事情

后续 Codex 不能做：

- 不要调用 OpenAI/cloud API。
- 不要把 37 dimensions 改成 embedding/clustering。
- 不要换掉 Q-learning。
- 不要合并 Reasoner/Guide/Validator。
- 不要让 emotion/audio label 进入 LLM prompt。
- 不要新增 DBT/ACT 或其他 therapy framework。
- 不要改 score granularity。
- 不要 redesign session report。
- 不要为了速度跳过 logging。
- 不要先做 speculative optimization；先测量 bottleneck。

## 8. 如果 Jetson 上失败，应收集的信息

请把以下内容贴回给 Codex：

```bash
python --version
pip freeze | grep -E "torch|transformers|peft|accelerate|bitsandbytes"
python -m unittest discover -s tests -q
python scripts/smoke_test_adapters.py
```

同时收集：

```bash
uname -a
free -h
df -h
tegrastats
```

如果是 CUDA/PyTorch 问题，还需要：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

## 9. 给 Jetson 本地 Codex 的第一条建议任务

建议先让 Jetson 本地 Codex 执行：

```text
请阅读 docs/jetson_codex_handoff_report.md。不要修改 CaiTI 论文定义的 conversation logic。先在 Jetson 上运行 unit tests、compileall、trace_mock_e2e.py、smoke_test_adapters.py，并汇总真实报错、内存、延迟。不要接入 emotion 到 LLM prompt，不要换 Q-learning，不要合并 adapters/roles。
```

第一轮目标不是完整产品化，而是确认：

- local model 能否加载；
- adapters 能否切换；
- 输出格式是否稳定；
- Jetson 8GB unified memory 是否够；
- text pipeline 能否跑通。
