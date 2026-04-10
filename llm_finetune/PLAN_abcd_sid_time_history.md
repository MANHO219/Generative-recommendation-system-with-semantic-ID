# LLM 微调 A/B/C/D 统一改造计划（SID+时间历史主路径）

## 0. 结论先行（已确认决策）

本计划用于统一 `main/llm_finetune` 训练与推理链路，默认采用 **SID+时间历史** 提示范式，并将 `sid_format_mode` 固定为 **`angle_bracket`**。

已确认决策如下：

- 默认主路径：`sid+时间历史`（V1 风格）
- `sid_format_mode`：`angle_bracket`
- 缓存策略：**A（直接清空旧缓存并重建）**
- 其余策略：采用推荐方案
  - SID 校验：`regex + 解析器` 双重校验
  - 迁移节奏：先双模式灰度，再切换主路径

---

## 1. 目标与范围

### 1.1 改动目标

在不破坏现有训练框架（QLoRA + SFTTrainer + Trie 约束解码）的前提下，完成 A/B/C/D 四项改造：

- **A（配置统一）**：配置层明确默认模式与 SID 格式策略
- **B（数据统一）**：兼容 GNPR `instruction/input/output` 样式并标准化为 `messages`
- **C（训练评测对齐）**：训练构造、评测、约束解码使用同源 SID 规则
- **D（推理入口对齐）**：推理 prompt 与训练 prompt 同构，减少分布偏移

### 1.2 范围边界

- 涉及目录：`main/llm_finetune/`、`main/inference/`
- 不改内容：基础模型结构、LoRA 核心超参、RQ-VAE 主体
- 目标是“提示链路一致性与数据表达一致性”，不是“重做语义ID生成算法”

---

## 2. 实施步骤（按依赖顺序）

### Step 1：A 配置统一（先做）

**文件**：`main/llm_finetune/config.py`

**动作**：

1. 新增/固化配置项（命名可微调，但语义必须一致）：
   - `default_prompt_mode = "sid_time_history"`
   - `sid_format_mode = "angle_bracket"`
   - `enable_legacy_prompt_mode = True`（灰度期保留）
2. 将 `PROMPT_TEMPLATE` 拆分为：
   - `sid_time_history`（主模板）
   - `legacy_profile_context`（兼容模板）
3. 在配置中显式写明输出格式：`<a_x><b_y><c_z>`（可选 `<d_n>`）

**验收（DoD）**：

- `config.py` 中可明确看见默认主路径与 `angle_bracket` 设定
- 下游代码可通过同一配置键读取模式，而非硬编码

---

### Step 2：B 数据标准化（GNPR 与现有数据并存）

**文件**：`main/llm_finetune/dataset.py`

**动作**：

1. 在数据构建层增加输入源判断与分流：
   - Yelp 原生样本（当前结构）
   - GNPR 样本（`instruction/input/output`）
2. 增加标准化函数（统一输出）：
   - 标准输出固定为：`{"messages": [{role, content}, ...]}`
3. 对 GNPR 样本：
   - 保留其核心表达（历史 SID + 时间 + 目标时间）
   - 仅做最小重排，不引入额外冗余字段
4. 对 Yelp 样本：
   - 构建 V1 风格的 SID+时间历史文本
   - 避免引入大量 profile/context 冗余信息

**验收（DoD）**：

- 两类输入都能产出 `messages`
- 缺字段时抛出可读错误，不 silent fail
- `target_sid` 统一为 `angle_bracket` 格式

---

### Step 3：D 主路径模板落地（SID+时间历史）

**文件**：`main/llm_finetune/dataset.py`（`format_prompt` / `format_instruction`）

**动作**：

1. 主模板对齐 `GNPR-SID/V1` 风格：
   - 历史：`time visited <a_...><b_...><c_...>`
   - 查询：`When {target_time} user_{id} is likely to visit:`
2. 输出严格为目标 SID（assistant 内容只含 SID）
3. 减少非必要自然语言说明，突出时序和 SID 转移模式

**验收（DoD）**：

- 随机抽样 20 条 `messages`，`assistant` 字段均为 SID 串
- `user` 文本均含“历史时间+SID”与“目标时间查询”结构

---

### Step 4：C 训练与评测同源对齐

**文件**：`main/llm_finetune/trainer.py`

**动作**：

1. `build_hf_dataset()` 与评测 prompt 构造统一使用 `tokenizer.apply_chat_template`
2. 训练与评测读取同一 `messages` 结构，避免手写 ChatML 与模板漂移
3. Trie 约束解码 SID 集合来源唯一：
   - 与训练标签同源（同一 `semantic_ids_path` / 同一 SID 格式）
4. 在评测中增加 SID 格式合法性检查（配合双重校验）

**验收（DoD）**：

- 训练、评测、生成三处 prompt 构造路径一致
- `TokenTrie` 构建 SID 与训练标签 SID 格式一致（均为 `angle_bracket`）

---

### Step 5：推理入口同构（训练即推理）

**文件**：`main/inference/run_inference.py`

**动作**：

1. 推理端 prompt 复用训练主模板（`sid_time_history`）
2. 支持 GNPR 风格输入样本直通/轻转换
3. 推理端 Trie 构建逻辑与训练评测一致

**验收（DoD）**：

- 相同样本在训练评测脚本与推理脚本中，prompt 主体一致
- 约束生成输出 SID 均通过格式校验

---

### Step 6：文档与操作流程收口

**文件**：`main/llm_finetune/README.md`

**动作**：

1. 更新默认主路径说明：`sid+时间历史`
2. 更新 SID 格式说明：`angle_bracket`
3. 明确缓存策略 A：
   - 每次切换模板/schema/SID 字典后，清空缓存重建
4. 写入回滚步骤（灰度失败时可切回 legacy）

**验收（DoD）**：

- 新同学仅看 README 即可按当前策略启动训练
- 缓存与回滚说明清晰、可执行

---

## 3. 缓存策略（已选 A）

### 3.1 执行规则

采用 **A：直接清空旧缓存**。

触发条件（任一满足即清空）：

- Prompt 模板改动
- `sid_format_mode` 改动
- `semantic_ids_path` 改动
- 数据标准化 schema 改动

### 3.2 操作命令

```bash
rm -rf /mnt/data/liuwei/yewenhao/main/output/dataset_cache/
```

---

## 4. 推荐策略（除缓存外）

### 4.1 SID 双重校验（推荐 B）

对训练标签、评测输出、推理输出执行：

1. Regex 校验（快速过滤）
2. 解析器校验（结构化验证）

目标：尽早发现非法 SID 或混杂格式。

### 4.2 迁移节奏（推荐 B）

采用“先灰度、后切主”两阶段：

1. **灰度期**：保留 legacy 模式，默认仍可切换
2. **切主期**：将 `sid_time_history + angle_bracket` 固化为唯一默认

---

## 5. 风险与回滚

### 风险 1：旧缓存污染新模板

- 现象：训练表现异常、标签分布不一致
- 处理：执行缓存策略 A（直接清空）

### 风险 2：训练/推理模板不一致

- 现象：离线评测与在线推理差异大
- 处理：统一 `apply_chat_template` 构造路径

### 风险 3：SID 混合格式（`angle_bracket` 与 `hyphen/grid`）

- 现象：Trie 命中异常、valid_sid_rate 虚高/虚低
- 处理：启用双重校验并在入口拒绝不合规样本

### 快速回滚

1. 切回 `legacy_profile_context`
2. 保留 `sid_format_mode=angle_bracket` 不变（避免多重变量）
3. 清空缓存并重建

---

## 6. 完成定义（整体 DoD）

满足以下全部条件即视为完成：

1. 默认配置为 `sid_time_history + angle_bracket`
2. 训练/评测/推理 prompt 生成路径同源
3. GNPR 与现有样本均可标准化为 `messages`
4. 清缓存后可稳定重建数据并启动训练
5. SID 输出通过双重校验
6. README 与计划文档同步完成

---

## 7. 实施后建议的最小检查清单

1. 抽样打印 20 条训练样本，确认 `assistant` 仅含 SID
2. 抽样打印 20 条推理 prompt，确认与训练主模板一致
3. 运行一次小规模 eval（几十到几百样本）观察：
   - `valid_sid_rate`
   - `exact_match`
   - 输出 SID 合法率

---

## 8. 执行顺序摘要（TL;DR）

- **先 A**：定配置（默认主路径 + angle_bracket）
- **再 B/D**：建标准化并落地 SID+时间历史模板
- **后 C**：统一训练评测与约束解码
- **最后文档**：README + 本计划落地
- **全程用缓存 A**：每次 schema/模板变化都清空重建
