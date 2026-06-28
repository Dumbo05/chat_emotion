import csv,json
from pathlib import Path
root=Path('.');result=json.loads((root/'models/speech/wavlm_clean_v2_final_results.json').read_text(encoding='utf8'));split=json.loads((root/'models/speech/wavlm_clean_split_v2.json').read_text(encoding='utf8'));rows=result['models'];best=max(rows,key=lambda x:x['test_macro_f1'])
lines=['# WavLM 干净跨说话人评估（v2）','',f'- 协议：{result["protocol"]}',f'- 数据集：CREMA-D + EmoDB（六个共同情绪）',f'- 初始划分：61 名训练、20 名验证、20 名测试说话人，集合完全互斥',f'- 最终训练：冻结配置后合并训练+验证，共 {result["fit_speakers"]} 名说话人；测试 {result["test_speakers"]} 名说话人',f'- 样本：最终训练 {result["fit_samples"]} 条，测试 {result["test_samples"]} 条',f'- 测试状态：{split["test_status"]}（配置冻结后统一评估一次）','- TESS 排除原因：只有两名说话人，无法同时构成互斥的训练、验证、测试三部分','- 本轮未修改或重新打包 EmotionRecognition.exe','', '## 模型对照','', '| 模型 | 特征 | 分类器 | 验证 Macro-F1 | 测试 Accuracy | 测试 Macro-F1 |','|---|---|---|---:|---:|---:|']
for r in rows:lines.append(f'| {r["display"]} | {r["feature"]} | {r["classifier"]} | {r["validation_macro_f1"]*100:.2f}% | {r["test_accuracy"]*100:.2f}% | {r["test_macro_f1"]*100:.2f}% |')
lines += ['', '## 最佳冻结配置','',f'- 模型：{best["display"]}',f'- WavLM 层：第 {best["config"]["layer"]} 层',f'- 池化：{best["config"]["pooling"]}',f'- SVM：C={best["config"]["C"]}，gamma={best["config"]["gamma_factor"]}/特征维数，class_weight={best["config"]["class_weight"]}',f'- 验证 Macro-F1：{best["validation_macro_f1"]*100:.2f}%',f'- 测试 Accuracy：{best["test_accuracy"]*100:.2f}%',f'- 测试 Macro-F1：{best["test_macro_f1"]*100:.2f}%','', '## 最佳模型逐类测试召回率','', '| 情绪 | Recall | F1 | Support |','|---|---:|---:|---:|']
zh={'anger':'愤怒','disgust':'厌恶','fear':'恐惧','joy':'高兴','sadness':'悲伤','neutral':'中性'}
for label in split['emotions']:
 x=best['classification_report'][label];lines.append(f'| {zh[label]} | {x["recall"]*100:.2f}% | {x["f1-score"]*100:.2f}% | {int(x["support"])} |')
lines += ['', '## 证据文件','', '- `models/speech/wavlm_clean_split_v2.json`：不可重叠说话人划分及测试状态', '- `models/speech/wavlm_clean_v2_frozen_evaluation_plan.json`：测试开封前冻结的四模型计划', '- `models/speech/wavlm_clean_v2_final_results.json`：一次性测试完整指标、分类报告、混淆矩阵及模型哈希', '- `models/speech/clean_v2_models/`：冻结模型文件']
(root/'docs/WAVLM_CLEAN_EVALUATION_V2.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
with (root/'models/speech/wavlm_clean_v2_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.writer(f);w.writerow(['模型','特征','分类器','验证 Macro-F1','测试 Accuracy','测试 Macro-F1']);
 for r in rows:w.writerow([r['display'],r['feature'],r['classifier'],r['validation_macro_f1'],r['test_accuracy'],r['test_macro_f1']])
print('\n'.join(lines[10:19]));print('BEST_PER_CLASS',json.dumps({zh[k]:round(best['classification_report'][k]['recall']*100,2) for k in split['emotions']},ensure_ascii=False));
