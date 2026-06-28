# 图像模块 v4 强视觉模型实验总结

## 1. 实验目标

本轮实验目标是提升项目图像情感识别模块性能，将原有基于 RAF-DB 训练的 SE-ResNet18 图像模型升级为更强的视觉主干网络，并通过表情预训练、RAF-DB 微调和多模型 ensemble 提高 RAF-DB Basic 官方测试集上的表现。

原图像模块参考结果：

| 版本 | RAF-DB test Accuracy | RAF-DB test Macro-F1 | 说明 |
| --- | ---: | ---: | --- |
| 已部署 SE-ResNet18 | 78.16% | 约 69% | YuNet + SE-ResNet18 + Flip TTA |
| image-v3 FER 预训练最佳 | 81.71% | 73.80% | FER2013 预训练 + RAF-DB 微调 |

本轮 v4 的核心改动：

- 使用 `timm` 加载 ImageNet / 大规模视觉预训练模型。
- 保留 FER2013 表情预训练阶段，再进行 RAF-DB Basic 微调。
- 固定 RAF-DB 官方 test set，仅用 RAF train 内部 validation 做模型选择。
- 保存每个模型的 validation/test logits，用 soft-voting ensemble 搜索组合。
- 最优模型包已回传本地，后续可集成到本地应用调用。

## 2. 数据与训练流程

训练流程由 `scripts/image/train_rafdb_image_v4_strong.py` 执行，主要阶段如下：

| 阶段 | 内容 |
| --- | --- |
| 强视觉 backbone 初始化 | 使用 `timm.create_model` 加载 ConvNeXt、EfficientNetV2、MaxViT 等预训练模型 |
| 表情预训练 | 默认使用项目中的 FER2013/fear2013 目录，也支持 `--expression-pretrain-root` 接入 AffectNet、FERPlus、ExpW 等 ImageFolder 数据 |
| RAF-DB 划分 | RAF 官方 train 中按 seed 划出 10% validation，官方 test 只用于最终评估 |
| RAF-DB 微调 | 按 validation Macro-F1 保存 `best_raf_finetune.pth` |
| TTA 评估 | validation/test 均使用水平翻转 TTA |
| logits 保存 | 输出 `raf_val_logits.npz`、`raf_test_logits.npz`，供 ensemble 使用 |

ensemble 由 `scripts/image/ensemble_rafdb_image_v4.py` 执行。该脚本会校验不同 run 的 validation/test 样本顺序一致，再枚举 1 到 `--max-members` 个模型组合，按 validation Macro-F1 排序。

## 3. 单模型结果

以下为本轮主要单模型在 RAF-DB official test 上的结果。

| 模型 run | Test Accuracy | Test Macro-F1 | Fear F1 | Disgust F1 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| `efficientnetv2_m_224_seed42` | 87.39% | 80.84% | 64.79% | 66.47% | 强单模，进入最终 ensemble |
| `maxvit_base_224_seed42` | 87.09% | 79.47% | 60.00% | 63.80% | Accuracy 高，有 ensemble 互补性 |
| `convnext_base_224_seed42` | 86.60% | 80.95% | 68.00% | 66.86% | 最早的强基线，单模 Macro-F1 高 |
| `convnext_base_224_seed45` | 86.08% | 79.96% | 68.92% | 65.69% | 不同 split，未纳入严格 ensemble |
| `efficientnetv2_m_224_seed43` | 85.53% | 78.93% | 62.86% | 67.71% | 不同 split，只做过 test-only 参考 |
| `convnext_large_224_seed42` | 85.50% | 77.42% | 57.35% | 63.41% | 单模一般，但与其他模型互补 |
| `convnextv2_base_224_seed42` | 78.13% | 69.01% | 50.60% | 46.63% | 不适合本任务，丢弃 |
| `vit_base_224_seed44` | 62.42% | 54.60% | 40.18% | 34.78% | 训练效果差，丢弃 |
| `swinv2_base_256_seed42` | 49.05% | 44.23% | 45.03% | 14.26% | 崩溃，丢弃 |

观察：

- ConvNeXt Base、EfficientNetV2-M、MaxViT Base 是本轮最有效的三类主干。
- ConvNeXt Large 单模不突出，但在 ensemble 中提供了有效互补。
- SwinV2、ViT、ConvNeXtV2 在本配置下表现不稳定或明显落后，不纳入最终模型。

## 4. Ensemble 结果

### 4.1 严格 validation 选择的主结果

最终用于正式报告的主结果建议采用 validation 排名第一的组合：

| 组合 | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| EfficientNetV2-M + ConvNeXt-Large + MaxViT-Base | 89.98% | 85.29% | **89.41%** | **83.17%** |

该组合成员：

- `efficientnetv2_m_224_seed42`
- `convnext_large_224_seed42`
- `maxvit_base_224_seed42`

各类别 Test F1：

| 类别 | F1 |
| --- | ---: |
| surprise | 89.02% |
| fear | 66.19% |
| disgust | 71.21% |
| joy | 95.33% |
| sadness | 88.73% |
| anger | 83.89% |
| neutral | 87.84% |

相对 image-v3 最佳结果提升：

| 指标 | image-v3 | image-v4 validation-selected ensemble | 提升 |
| --- | ---: | ---: | ---: |
| Accuracy | 81.71% | **89.41%** | +7.70 个百分点 |
| Macro-F1 | 73.80% | **83.17%** | +9.37 个百分点 |

### 4.2 探索性 test-best 组合

在 `ensemble_split42_with_maxvit/ensemble_summary.csv` 中，事后查看 test 指标时，有两个包含 ConvNeXt Base 的组合 test Accuracy 达到 89.60%。其中四模型组合的 test Macro-F1 最高：

| 组合 | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| ConvNeXt-Base + EfficientNetV2-M + ConvNeXt-Large + MaxViT-Base | 89.33% | 83.66% | 89.60% | **84.23%** |
| ConvNeXt-Base + EfficientNetV2-M + MaxViT-Base | 89.41% | 83.60% | 89.60% | 84.07% |

注意：这两个组合不是按 validation Macro-F1 排名第一，因此更适合作为探索性观察，而不是主报告中的严格选型结果。若报告强调严格验证集选型，应使用 89.41% / 83.17% 的三模型组合；若报告允许补充说明 test-set 观察，可注明四模型组合达到 89.60% Accuracy / 84.23% Macro-F1。

## 5. 最终文件位置

本地已解压结果位于：

```text
C:\Users\Lenovo\Desktop\del\server-results\image-v4\extracted
```

### 5.1 可部署最优模型包

```text
C:\Users\Lenovo\Desktop\del\server-results\image-v4\extracted\image_v4_deploy_best
```

关键内容：

| 路径 | 用途 |
| --- | --- |
| `runs/efficientnetv2_m_224_seed42/best_raf_finetune.pth` | 最优 ensemble 成员 1 |
| `runs/convnext_large_224_seed42/best_raf_finetune.pth` | 最优 ensemble 成员 2 |
| `runs/maxvit_base_224_seed42/best_raf_finetune.pth` | 最优 ensemble 成员 3 |
| `runs/*/config.json` | 模型架构、输入尺寸、训练参数 |
| `runs/*/summary.json` | 单模型训练和测试指标 |
| `ensemble_split42_with_maxvit/ensemble_results.json` | 最终 ensemble 搜索完整结果 |
| `ensemble_split42_with_maxvit/ensemble_summary.csv` | ensemble 排名表 |
| `train_rafdb_image_v4_strong.py` | 训练/模型定义脚本 |
| `ensemble_rafdb_image_v4.py` | ensemble 评估脚本 |

这三个 `.pth` 是后续集成本地调用的核心权重文件。

### 5.2 报告主要数据包

```text
C:\Users\Lenovo\Desktop\del\server-results\image-v4\extracted\image_v4_report_main
```

关键内容：

| 路径 | 用途 |
| --- | --- |
| `summary_*.csv` | 每个单模型 run 的主要指标 |
| `ensembles/ensemble_split42_with_maxvit/ensemble_summary.csv` | 最终 ensemble 排名 |
| `ensembles/ensemble_split42_with_maxvit/ensemble_results.json` | 最终 ensemble 详细指标、混淆矩阵、各类 precision/recall/F1 |
| `ensembles/ensemble_split42_three_models/` | 早期三模型 ensemble 结果 |
| `ensembles/ensemble_convnext_effnet_split42/` | 早期二模型 ensemble 结果 |

## 6. 可部署集成建议

当前本地应用的图像识别器仍是 OpenCV DNN + ONNX 的单模型推理路径。v4 最优结果是 PyTorch/timm 三模型 ensemble，后续集成有两条路线：

| 路线 | 优点 | 风险 |
| --- | --- | --- |
| PyTorch/timm 本地加载 `.pth` ensemble | 最接近服务器结果，改动少 | 打包体积大，需要 PyTorch/timm 运行时 |
| 导出各模型 ONNX 后用 ONNX Runtime ensemble | 更适合桌面部署 | ConvNeXt/MaxViT/EfficientNet 导出和预处理一致性需要单独验证 |

建议先做 PyTorch 版本地验证推理，确认单图、人脸对齐、归一化、Flip TTA 和服务器一致；再考虑 ONNX 导出和桌面打包。

## 7. 结论

本轮图像 v4 实验完成了从轻量 SE-ResNet18 到强视觉模型 ensemble 的升级。按 validation 选型的正式主结果为：

```text
EfficientNetV2-M + ConvNeXt-Large + MaxViT-Base
RAF-DB official test Accuracy = 89.41%
RAF-DB official test Macro-F1 = 83.17%
```

相对 image-v3 最佳结果，Accuracy 提升 7.70 个百分点，Macro-F1 提升 9.37 个百分点。弱类 fear/disgust 仍是后续主要瓶颈，但 disgust F1 已提升到 71.21%，整体图像模块性能已经接近 90% Accuracy，可以作为当前图像板块报告结果提交。
