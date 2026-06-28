# WavLM 干净跨说话人评估（v2）

- 协议：WavLM clean speaker-independent evaluation v2
- 数据集：CREMA-D + EmoDB（六个共同情绪）
- 初始划分：61 名训练、20 名验证、20 名测试说话人，集合完全互斥
- 最终训练：冻结配置后合并训练+验证，共 81 名说话人；测试 20 名说话人
- 样本：最终训练 6333 条，测试 1563 条
- 测试状态：evaluated-once（配置冻结后统一评估一次）
- TESS 排除原因：只有两名说话人，无法同时构成互斥的训练、验证、测试三部分
- 本轮未修改或重新打包 EmotionRecognition.exe

## 模型对照

| 模型 | 特征 | 分类器 | 验证 Macro-F1 | 测试 Accuracy | 测试 Macro-Recall | 测试 Macro-F1 |
|---|---|---|---:|---:|---:|---:|
| MFCC + RBF-SVM | MFCC handcrafted full 177 | RBF-SVM balanced | 49.19% | 50.16% | 50.08% | 49.32% |
| WavLM mean pooling | WavLM mean | RBF-SVM | 74.66% | 72.81% | 72.90% | 72.45% |
| WavLM mean+std pooling | WavLM mean+std | RBF-SVM | 74.96% | 73.58% | 73.65% | 73.21% |
| WavLM mean+std + balanced SVM | WavLM mean+std | RBF-SVM balanced | 75.16% | 73.90% | 74.00% | 73.54% |
## 最佳冻结配置

- 模型：WavLM mean+std + balanced SVM
- WavLM 层：第 7 层
- 池化：mean_std
- SVM：C=5.0，gamma=0.25/特征维数，class_weight=balanced
- 验证 Macro-F1：75.16%
- 测试 Accuracy：73.90%
- 测试 Macro-Recall：74.00%
- 测试 Macro-F1：73.54%

## 最佳模型逐类测试召回率

| 情绪 | Recall | F1 | Support |
|---|---:|---:|---:|
| 愤怒 | 93.43% | 83.52% | 274 |
| 厌恶 | 63.57% | 67.08% | 258 |
| 恐惧 | 60.84% | 68.23% | 263 |
| 高兴 | 73.06% | 74.72% | 271 |
| 悲伤 | 65.53% | 65.41% | 264 |
| 中性 | 87.55% | 82.26% | 233 |

## 证据文件

- `models/speech/wavlm_clean_split_v2.json`：不可重叠说话人划分及测试状态
- `models/speech/wavlm_clean_v2_frozen_evaluation_plan.json`：测试开封前冻结的四模型计划
- `models/speech/wavlm_clean_v2_final_results.json`：一次性测试完整指标、分类报告、混淆矩阵及模型哈希
- `models/speech/clean_v2_models/`：冻结模型文件
