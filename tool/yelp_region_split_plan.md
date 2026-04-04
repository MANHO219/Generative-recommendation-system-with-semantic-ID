# Yelp 按州预拆分数据集方案

## 背景
目标：为不同州训练区域专用模型（包括 Semantic ID 训练和 LLM 微调），需要将数据按州预拆分到独立目录。

## 现状
- `processed/` 下有 `business_poi.json`、`review_poi.json`、`user_poi_interactions.json` 等
- 这些文件包含所有 24 个州的数据
- LLM 微调需要完整的用户-商铺交互数据，不能仅靠加载时过滤

## 方案

### 1. 脚本 `tool/split_by_state.py`
按州预拆分数据到 `{processed_dir}/{state}/` 目录：

```
dataset/yelp/processed/
├── PA/
│   ├── business_poi.json      # PA 州的商铺
│   ├── review_poi.json         # PA 商铺的评论
│   └── user_poi_interactions.json  # PA 用户的交互
├── FL/
│   ├── business_poi.json
│   ├── review_poi.json
│   └── user_poi_interactions.json
└── ...
```

**处理逻辑**：
1. 读取 `business_poi.json`，按 `state` 分组得到各州 business_id 集合
2. 读取 `review_poi.json`，筛选 business_id 在目标州集合内的记录
3. 读取 `user_poi_interactions.json`，筛选涉及目标州商铺的交互
4. 对每州写入独立文件

### 2. 关于用户数据
采用**截断 + 过滤短序列**策略：

| 场景 | 处理 |
|---|---|
| 用户只在目标州有交互 | 直接保留 |
| 用户跨州但目标州 ≥ N 条 | 截断，只保留目标州记录 |
| 截断后目标州 < N 条（建议 N=3） | 丢弃该用户 |

理由：跨州消费是低频行为，截断后序列在时间上仍有序；复制全量会引入其他州 business_id 造成 OOV，并稀释区域信号。

不在州内再按城市拆分（数据量适中）。

## 执行结果

### 保留州（15 个）

| 州 | POI 数 | 保留用户 | 丢弃用户 | 备注 |
|----|--------|---------|---------|------|
| PA | 26,938 | 66,110 | 83,690 | 最大州，以费城/匹兹堡为主 |
| FL | 20,396 | 48,967 | 68,856 | 第二大，以坦帕为主 |
| TN | 9,573 | 31,994 | 62,423 | |
| MO | 8,710 | 21,507 | 33,969 | |
| IN | 9,038 | 18,376 | 26,007 | |
| LA | 8,126 | 48,300 | 91,490 | |
| AZ | 7,215 | 17,752 | 28,994 | |
| NJ | 6,864 | 11,590 | 24,146 | |
| NV | 5,295 | 17,888 | 38,622 | |
| CA | 3,821 | 19,202 | 65,528 | 非 LA/SF，为周边小城市 |
| AB | 4,828 | 3,260 | 2,515 | **加拿大省**（非美国州），含在内供参考 |
| ID | 3,333 | 7,434 | 14,250 | |
| DE | 1,848 | 3,656 | 9,979 | |
| IL | 1,791 | 2,415 | 6,035 | |
| XMS | 1 | 0 | — | 已排除 |

### 排除州（10 个，垃圾州）

| 州 | POI 数 | 保留用户 | 原因 |
|----|--------|---------|------|
| CO | 2 | 0 | 极少量数据（错误录入） |
| HI | 1 | 0 | 同上 |
| MT | 1 | 0 | 同上 |
| NC | 1 | 0 | 同上 |
| SD | 1 | 0 | 同上 |
| TX | 2 | 0 | 同上 |
| UT | 1 | 0 | 同上 |
| VI | 1 | 0 | 同上 |
| WA | 1 | 0 | 同上 |
| XMS | 1 | 0 | 无效代码，已丢弃 |

> 注：AB (Alberta) 为加拿大省，数据量尚可但非美国州。如研究限定美国本土，建议排除。

### 各州跨州用户丢弃率参考

| 州 | 保留率 | 丢弃率 |
|----|--------|--------|
| PA | 44.1% | 55.9% |
| FL | 41.5% | 58.5% |
| TN | 33.9% | 66.1% |
| LA | 34.6% | 65.4% |
| AZ | 38.0% | 62.0% |
| DE | 26.8% | 73.2% |
| AB | 56.4% | 43.6% |

丢弃率高是因为 Yelp 数据大量用户跨州低频消费，min_interactions=3 过滤后仅保留有效序列用户。

## 关键文件
- `semantic_id/dataset.py`: 现有数据加载逻辑，无需改动，直接通过 `--data_dir` 指向州子目录
- `tool/split_by_state.py`: 预拆分脚本

## 验证方式
```bash
# 1. 运行拆分脚本（已执行完毕）
python tool/split_by_state.py --data_dir ./dataset/yelp/processed

# 2. 按 PA 州训练（直接传子目录，无需其他改动）
python semantic_id/train.py --data_dir ./dataset/yelp/processed/PA

# 3. 按 FL 州训练
python semantic_id/train.py --data_dir ./dataset/yelp/processed/FL
```

## 注意
- 建议使用 min_interactions=3 以保证训练序列有效性
- AB 省数据如需美国本土研究可手动删除
- 大州（PA/FL）数据量充足可直接训练；小州（DE/IL）可用于小规模场景测试
