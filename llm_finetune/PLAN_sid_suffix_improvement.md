# Semantic ID 冲突后缀语义化改进（可执行版清单）

## 0. 目标与范围（先锁定）

### 0.1 改动目标
- 将冲突消歧后缀从无语义的 `<d_N>` 改为有空间语义的 `[GG]`（`GG = plus_code_neighborhood` 最后 2 位）。
- 新目标格式：`level0-level1-level2[GG]`，例如 `12-34-56[GQ]`。

### 0.2 范围边界
- 仅改 SID 生成、SID 使用提示模板、SID 解析工具。
- 不改 RQ-VAE 主体网络结构，不改损失函数。

### 0.3 关键决策（必须统一）
- 决策 A：冲突后缀统一使用 `[]` 风格，不再混用 `<d_N>`。
- 决策 B：`[GG]` 仍冲突时，fallback 也使用 `[]`，例如 `[GQ_1]`、`[GQ_2]`，避免训练标签双格式。

---

## 1. 实施清单（按顺序执行）

### Step 1：改 `resolve_sid_collisions()`（核心）
文件：
- `main/semantic_id/model.py`

动作：
- 扩展函数签名：
```python
def resolve_sid_collisions(
    semantic_ids: Dict[str, str],
    pluscode_neighborhoods: Optional[Dict[str, str]] = None,
    suffix_mode: str = "grid"  # "grid" | "index"
) -> Dict[str, str]:
```
- `suffix_mode="grid"` 时：
  - 从 `pluscode_neighborhoods[bid]` 取 neighborhood。
  - `grid = neighborhood[-2:]`（不足 2 位则用 `UNK`）。
  - 无冲突：保持原 SID。
  - 有冲突：第一条用 `[GG]`，后续同组用 `[GG_1]`、`[GG_2]`...（保证唯一）。
- `suffix_mode="index"` 时保留旧逻辑（向后兼容），但仅用于兼容，不用于新实验默认。

验收（DoD）：
- 输出 `semantic_ids.json` 不出现 `<d_`。
- 每个 `business_id` 的 SID 全局唯一。

---

### Step 2：把 `plus_code_neighborhood` 接入生成链路
文件：
- `main/semantic_id/trainer.py`
- （如需要）`main/semantic_id/dataset.py`

动作（推荐最小改动路径）：
- 在 `generate_semantic_ids()` 中基于 `business_id` 构建：
  - `pluscode_neighborhoods[bid] = poi_dataset.businesses[bid].get("plus_code_neighborhood", "")`
- 调用：
```python
resolve_sid_collisions(
    semantic_ids,
    pluscode_neighborhoods=pluscode_neighborhoods,
    suffix_mode="grid"
)
```
- 若当前函数拿不到 `poi_dataset`，则补参数透传；不要依赖 batch 已含该字段。

验收（DoD）：
- 不改 batch 结构也能正常跑通 SID 导出。
- `plus_code_neighborhood` 缺失样本可稳定 fallback，不报错。

---

### Step 3：更新 LLM prompt 模板
文件：
- `main/llm_finetune/config.py`

动作：
- `system` 中格式说明改为 `level0-level1-level2[GRID]`。
- `instruction` 改为 `Output only the Semantic ID in format: XX-XX-XX[GG]`。

验收（DoD）：
- 运行配置检查脚本可打印新格式文案。

---

### Step 4：同步 SID 解析工具（避免统计失真）
文件：
- `main/tool/semantic_id_analysis.py`
- `main/tool/sid_statistics.py`
- `main/tool/collision_tuple_analysis.py`（若复用同一 parser）

动作：
- `parse_sid()` 同时支持：
  - 旧格式：`12-34-56<d_0>`
  - 新格式：`12-34-56[GG]`、`12-34-56[GG_2]`
- 解析规则：
  - `base = 12-34-56`
  - `disambig = <...> 或 [...] 或 None`

验收（DoD）：
- 对旧实验 `semantic_ids.json` 统计结果不回归。
- 对新实验可正确识别 disambig，不把 `p3` 解析成 `56[GG]`。

---

### Step 5：文档与缓存流程同步
文件：
- `main/llm_finetune/README.md`
- （可选）`main/semantic_id/README.md`

动作：
- 更新 SID 格式说明为可带 `[]` 后缀。
- 明确写入：更换 `semantic_ids.json` 后必须删除 `output/dataset_cache` 并重建。

验收（DoD）：
- 新同学按 README 可复现实验，不踩缓存旧标签坑。

---

## 2. 验证清单（命令级）

### 2.1 语法检查
```bash
python -c "import ast; ast.parse(open('main/semantic_id/model.py', encoding='utf-8').read()); print('model.py OK')"
python -c "import ast; ast.parse(open('main/tool/semantic_id_analysis.py', encoding='utf-8').read()); print('semantic_id_analysis.py OK')"
python -c "import ast; ast.parse(open('main/tool/sid_statistics.py', encoding='utf-8').read()); print('sid_statistics.py OK')"
```

### 2.2 端到端生成
```bash
cd main
python semantic_id/train.py --data_dir ./dataset/yelp/processed --device cpu
```
检查：
- `semantic_ids.json` 存在。
- 随机抽样包含 `XX-XX-XX[GG]` 或 `XX-XX-XX[GG_n]`。
- 不应出现 `<d_`.

### 2.3 Prompt 检查
```bash
python -c "from llm_finetune.config import PROMPT_TEMPLATE; print(PROMPT_TEMPLATE['system']); print(PROMPT_TEMPLATE['instruction'])"
```
检查：
- 文案包含 `[GRID]` / `XX-XX-XX[GG]`。

### 2.4 解析兼容性检查
```bash
python main/tool/sid_statistics.py --experiment <old_exp>
python main/tool/sid_statistics.py --experiment <new_exp>
```
检查：
- 两者都能正常跑完，无解析异常。

---

## 3. 风险与回滚

### 风险 1：`GG` 重复率高
- 现象：同一 SID 组内 `GG` 不足以唯一。
- 处理：组内追加 `_n`（`[GG_1]`）保证唯一，不回退 `<d_N>`。

### 风险 2：部分样本缺少 `plus_code_neighborhood`
- 处理：使用 `[UNK]`，并在组内继续 `_n` 去重。

### 风险 3：历史实验工具失配
- 处理：解析函数双栈兼容 `<...>` 和 `[...]`。

### 快速回滚
- 将 `suffix_mode` 默认改回 `"index"`，重新导出 `semantic_ids.json`。
- 恢复旧 prompt 文案并清理 `output/dataset_cache` 后重建。

---

## 4. 完成定义（整体）
- 生成链路默认输出 `XX-XX-XX[...]`，无 `<d_N>`。
- SID 全局唯一且工具链可解析。
- LLM 微调模板、README、缓存说明全部同步。
- 旧实验与新实验分析脚本均可运行。
