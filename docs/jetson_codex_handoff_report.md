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
- `src/voice/sentence_stream.py`

设计：

- STT/TTS 是 I/O shell，不进入 LLM reasoning。
- STT: microphone/audio -> text。
- LLM pipeline 仍然只接收 text。
- TTS: generated text -> speech。
- 支持按句子边界 streaming TTS。
- 默认 backend 是 `console`，Jetson 上可以切换为 local command backend。

`config.yaml` 中：

```yaml
voice:
  stt_backend: "console" # console | command
  tts_backend: "console" # console | command
  stt_command: ""
  tts_command: ""
  stt_timeout_sec: 120
  tts_timeout_sec: 60
  empty_transcript_retries: 2
```

也可以用环境变量覆盖：

```bash
export CAITI_STT_BACKEND=command
export CAITI_STT_COMMAND="your-local-stt-command"
export CAITI_TTS_BACKEND=command
export CAITI_TTS_COMMAND="your-local-tts-command"
```

要求：

- STT command 必须把 transcript 输出到 stdout。
- TTS command 必须从 stdin 读取文本并播放。
- STT/TTS 必须本地运行，不能调用云 API。

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
- `transformers`
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

预期：52 tests OK。

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

### 5.4 Real adapter smoke test

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

### 5.5 Text mode session

```bash
python LLM_therapist_Application.py
```

用途：

- 不接语音，先验证完整 text pipeline。

### 5.6 Voice shell session

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

## 6. 当前未完成 TODO

### High priority

1. Jetson real adapter smoke test
   - 记录 load time。
   - 记录 peak memory。
   - 记录 task1/task2/task3/task4 latency。
   - 记录 base generation latency。

2. 解决 Jetson `bitsandbytes` / NF4 INT4 loading 风险
   - 只有在实际失败并有错误信息后再改。
   - 不要提前换模型或换推理框架。

3. STT/TTS 本地后端选择与验证
   - STT command 必须 stdout 输出文本。
   - TTS command 必须 stdin 输入文本并播放。
   - 不得调用云 API。

4. Voice E2E test
   - CaiTI question -> TTS 播放。
   - User speech -> STT transcript。
   - transcript -> existing LLM pipeline。
   - R-V Validator / Guides / CBT 输出按句子 TTS。

### Medium priority

5. Jetson benchmark script
   - model load time
   - adapter switch latency
   - per-task P50/P95
   - memory before/after generation
   - failure logs

6. Emotion module logging only
   - local path from Windows: `C:\Users\胡溪筼\Desktop\2026-7spring\CaiTi\emo_module`
   - Jetson 上如果接入，只能作为 metadata logging 或 RL reward metadata。
   - emotion label 不允许进入 LLM prompt。

7. Jetson deployment docs
   - JetPack version
   - PyTorch install command
   - CUDA/cuDNN versions
   - exact model cache path
   - exact STT/TTS backend command

### Low priority

8. Flask/server voice endpoint
   - 当前 Flask server 仍是 text API。
   - 若要 browser/mobile UI，可在不改 LLM logic 的前提下增加 audio endpoint。

9. Long-session reliability test
   - 连续跑完整 37 dimensions + CBT。
   - 检查 record lock、SQLite logs、report output。

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

