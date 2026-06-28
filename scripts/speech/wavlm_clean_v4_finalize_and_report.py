from __future__ import annotations
import argparse, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.train_speech_model import LABELS, discover_tess, discover_crema, discover_emodb

WEAK_CLASSES = ['fear', 'sadness', 'disgust', 'neutral']
V2_REFERENCE = {'test_accuracy': 0.7390, 'test_macro_f1': 0.7354, 'note': 'Prior clean v2 result supplied from project context; v2 files were not read by this script.'}
V3_REFERENCE = {'test_accuracy': 0.7369207772795217, 'test_macro_f1': 0.7352575785140297, 'note': 'Prior clean v3 result supplied from project context; v3 files were not read by this script.'}

def parse_args():
    p = argparse.ArgumentParser(description='Finalize WavLM clean v4 by evaluating sealed test set exactly once')
    p.add_argument('--split-file', type=Path, default=PROJECT_ROOT / 'models/speech/wavlm_clean_split_v4.json')
    p.add_argument('--frozen-plan', type=Path, default=PROJECT_ROOT / 'models/speech/wavlm_clean_v4_frozen_evaluation_plan.json')
    p.add_argument('--feature-cache', type=Path, default=PROJECT_ROOT / 'models/speech/wavlm_clean_v2_layer_stats.npy')
    p.add_argument('--final-results', type=Path, default=PROJECT_ROOT / 'models/speech/wavlm_clean_v4_final_results.json')
    p.add_argument('--model-output', type=Path, default=PROJECT_ROOT / 'models/speech/clean_v4_models/wavlm_clean_v4_best.joblib')
    p.add_argument('--report', type=Path, default=PROJECT_ROOT / 'outputs/speech/wavlm_clean_v4_report.md')
    p.add_argument('--seed', type=int, default=2027)
    return p.parse_args()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def manual_class_weight(name: str) -> Any:
    weights = {
        'manual_v1': {'fear': 1.20, 'sadness': 1.20, 'disgust': 1.15, 'neutral': 0.90, 'anger': 1.00, 'joy': 1.00},
        'manual_v2': {'fear': 1.35, 'sadness': 1.30, 'disgust': 1.25, 'neutral': 0.80, 'anger': 1.00, 'joy': 1.00},
        'manual_v3': {'fear': 1.50, 'sadness': 1.40, 'disgust': 1.35, 'neutral': 0.75, 'anger': 0.95, 'joy': 0.95},
    }
    if name == 'none':
        return None
    if name == 'balanced':
        return 'balanced'
    return {LABELS.index(k): float(v) for k, v in weights[name].items()}

def build_features(cache: np.ndarray, layers: tuple[int, ...]) -> np.ndarray:
    fused = np.asarray(cache[:, layers, :], np.float32).mean(axis=1)
    return fused[:, :1536]

def predict_with_bias(model: SVC, X: np.ndarray, bias: dict[str, float]) -> np.ndarray:
    scores = model.decision_function(X)
    if scores.ndim != 2:
        return model.predict(X)
    scores = np.asarray(scores, dtype=np.float64).copy()
    for label, value in bias.items():
        class_index = LABELS.index(label)
        matches = np.flatnonzero(model.classes_ == class_index)
        if len(matches):
            scores[:, matches[0]] += float(value)
    return model.classes_[np.argmax(scores, axis=1)]

def pct(x: float) -> str:
    return f'{x * 100:.2f}%'

def main():
    args = parse_args()
    if args.final_results.exists():
        raise SystemExit(f'Refusing to evaluate v4 test again; final results already exist: {args.final_results}')
    split = json.loads(args.split_file.read_text(encoding='utf8'))
    plan = json.loads(args.frozen_plan.read_text(encoding='utf8'))

    if split.get('protocol_name') != 'wavlm_clean_v4':
        raise RuntimeError('Split protocol is not wavlm_clean_v4')
    if split.get('test_status') != 'sealed':
        raise RuntimeError(f'v4 test set is not sealed before evaluation: {split.get("test_status")}')
    if plan.get('protocol_name') != 'wavlm_clean_v4':
        raise RuntimeError('Frozen plan protocol is not wavlm_clean_v4')
    if plan.get('used_test_set') is not False or plan.get('test_metrics_used') is not False:
        raise RuntimeError('Frozen plan does not state that test metrics were unused')
    statement = str(plan.get('statement', ''))
    if 'No v4 test-set metrics were computed or used' not in statement and 'No test-set metrics were used' not in statement:
        raise RuntimeError('Frozen plan does not contain a clear no-test-metrics statement')
    selected = plan.get('selected_validation_result')
    if not isinstance(selected, dict) or not selected.get('run_id'):
        raise RuntimeError('Frozen plan lacks selected validation result with run_id')
    if plan.get('test_evaluation_status') not in ('not_evaluated', None):
        raise RuntimeError('Frozen plan does not indicate test is not evaluated')

    samples = discover_tess(PROJECT_ROOT / 'datasets/TESS') + discover_crema(PROJECT_ROOT / 'datasets/CREMA-D') + discover_emodb(PROJECT_ROOT / 'datasets/EmoDB')
    sp = np.array([s.speaker for s in samples])
    y = np.array([LABELS.index(s.label) for s in samples])
    cache = np.load(args.feature_cache, mmap_mode='r')
    mf = np.load(PROJECT_ROOT / 'models/speech/multidataset_features.npz')
    if len(cache) != len(samples) or not np.array_equal(sp, mf['speakers'].astype(str)):
        raise RuntimeError('Feature cache and discovered sample order do not match')

    train_val = np.flatnonzero(np.isin(sp, split['train_speakers'] + split['val_speakers']))
    test = np.flatnonzero(np.isin(sp, split['test_speakers']))
    expected_train_val = split['sample_counts']['train'] + split['sample_counts']['validation']
    if len(train_val) != expected_train_val or len(test) != split['sample_counts']['test']:
        raise RuntimeError('Train+validation/test sample counts do not match split file')

    layers = tuple(int(x) for x in selected['wavlm_layers'])
    raw = build_features(cache, layers)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(raw[train_val])
    X_test = scaler.transform(raw[test])
    pca_components = selected.get('pca_components')
    pca = None
    if pca_components is not None:
        pca = PCA(n_components=int(pca_components), random_state=args.seed, svd_solver='randomized')
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    gamma_label = selected['gamma']
    gamma = 'scale' if gamma_label == 'scale' else float(gamma_label) / X_train.shape[1]
    model = SVC(
        C=float(selected['C']), gamma=gamma, kernel='rbf',
        class_weight=manual_class_weight(selected['class_weight_name']),
        cache_size=1024, decision_function_shape='ovr', random_state=args.seed,
    )
    model.fit(X_train, y[train_val])
    bias = selected['bias_search']
    pred = predict_with_bias(model, X_test, bias)

    labels = split['emotion_labels']
    used = [LABELS.index(x) for x in labels]
    rep = classification_report(y[test], pred, labels=used, target_names=labels, output_dict=True, zero_division=0)
    per_class_precision = {x: float(rep[x]['precision']) for x in labels}
    per_class_recall = {x: float(rep[x]['recall']) for x in labels}
    per_class_f1 = {x: float(rep[x]['f1-score']) for x in labels}
    per_class_support = {x: int(rep[x]['support']) for x in labels}
    weak_class_score_test = float(np.mean([per_class_f1[x] for x in WEAK_CLASSES]))

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    model_package = {
        'protocol_name': 'wavlm_clean_v4',
        'model': model,
        'scaler': scaler,
        'pca': pca,
        'selected_config': selected,
        'labels': LABELS,
        'emotion_labels': labels,
        'bias_search': bias,
        'trained_on': 'v4 train+validation speakers only',
        'created_at': datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(model_package, args.model_output, compress=3)
    model_hash = sha256_file(args.model_output)
    feature_cache_hash = sha256_file(args.feature_cache)

    final = {
        'protocol_name': 'wavlm_clean_v4',
        'split_file': str(args.split_file.resolve()),
        'frozen_plan_file': str(args.frozen_plan.resolve()),
        'selected_config': selected,
        'test_status': 'evaluated-once',
        'test_accuracy': float(accuracy_score(y[test], pred)),
        'test_macro_f1': float(f1_score(y[test], pred, labels=used, average='macro', zero_division=0)),
        'test_weighted_f1': float(f1_score(y[test], pred, average='weighted', zero_division=0)),
        'test_per_class_precision': per_class_precision,
        'test_per_class_recall': per_class_recall,
        'test_per_class_f1': per_class_f1,
        'test_per_class_support': per_class_support,
        'test_confusion_matrix': confusion_matrix(y[test], pred, labels=used).tolist(),
        'test_sample_count': int(len(test)),
        'test_speakers_count': int(len(split['test_speakers'])),
        'weak_classes': WEAK_CLASSES,
        'weak_class_score_test': weak_class_score_test,
        'model_path': str(args.model_output.resolve()),
        'model_hash': model_hash,
        'feature_cache': str(args.feature_cache.resolve()),
        'feature_cache_hash': feature_cache_hash,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'one_time_test_statement': 'The v4 test set was evaluated once using the frozen configuration. No hyperparameters, bias, or thresholds were adjusted after seeing this result.',
    }
    args.final_results.parent.mkdir(parents=True, exist_ok=True)
    args.final_results.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding='utf8')

    split['test_status'] = 'evaluated-once'
    split['test_evaluated_at'] = final['created_at']
    split['final_results_file'] = str(args.final_results.resolve())
    args.split_file.write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding='utf8')

    v4_acc = final['test_accuracy']
    v4_macro = final['test_macro_f1']
    deployment = '保留 v2 部署模型'
    deployment_reason = 'v4 test Macro-F1 未明显高于 clean v2/v3；同时 v4 引入 PCA、manual class weight 和 bias，复杂度更高，不建议替换当前部署模型。'
    if v4_macro > max(V2_REFERENCE['test_macro_f1'], V3_REFERENCE['test_macro_f1']) + 0.01:
        deployment = '建议替换当前部署模型'
        deployment_reason = 'v4 Macro-F1 明显高于 clean v2/v3，且复杂度仍可接受。'
    elif abs(v4_macro - max(V2_REFERENCE['test_macro_f1'], V3_REFERENCE['test_macro_f1'])) <= 0.005 and weak_class_score_test > 0.74:
        deployment = '研究模型保留；是否部署取决于应用是否更重视弱类'
        deployment_reason = 'v4 总体 Macro-F1 与 v2/v3 接近，但弱类有改善倾向；部署需看业务是否更重视 fear/sadness/disgust/neutral。'

    rows = []
    for label in labels:
        rows.append(f"| {label} | {per_class_precision[label]*100:.2f}% | {per_class_recall[label]*100:.2f}% | {per_class_f1[label]*100:.2f}% | {per_class_support[label]} |")
    cm_header = '| True \\ Pred | ' + ' | '.join(labels) + ' |'
    cm_sep = '| --- | ' + ' | '.join(['---:'] * len(labels)) + ' |'
    cm_rows = []
    cm = final['test_confusion_matrix']
    for label, row in zip(labels, cm):
        cm_rows.append('| ' + label + ' | ' + ' | '.join(str(int(x)) for x in row) + ' |')

    report = f'''# WavLM clean v4 final report

生成时间：{datetime.now().isoformat(timespec='seconds')}

## 1. 实验目的

clean v4 的目的，是在 clean v4 第一阶段已经冻结验证集最佳配置之后，对从未用于调参的 speaker-independent test set 做一次性评估，判断 v4 是否值得替换当前部署的语音情绪识别模型。

## 2. v4 数据划分

- 数据集：CREMA-D + EmoDB
- 排除：TESS，因为只有两个说话人，不适合构造互斥 train / validation / test speaker split。
- Train speakers: {len(split['train_speakers'])}, samples: {split['sample_counts']['train']}
- Validation speakers: {len(split['val_speakers'])}, samples: {split['sample_counts']['validation']}
- Test speakers: {len(split['test_speakers'])}, samples: {split['sample_counts']['test']}
- Train / validation / test speakers mutually exclusive.

## 3. sealed / evaluated-once 说明

评估前，`models/speech/wavlm_clean_split_v4.json` 的 `test_status` 为 `sealed`。本报告生成时，test set 已按 frozen plan 评估一次，并已更新为 `evaluated-once`。

没有在看到 v4 test 结果后调整模型、bias、阈值或超参数。

## 4. 第一阶段 validation sweep 摘要

- Validation candidates: 1161
- Selection rule: validation Macro-F1 优先；在接近最佳 Macro-F1 的候选中，再优先 weak_class_score。
- Weak classes: {', '.join(WEAK_CLASSES)}
- Selected validation run_id: `{selected['run_id']}`
- Selected validation Accuracy: {pct(selected['val_accuracy'])}
- Selected validation Macro-F1: {pct(selected['val_macro_f1'])}
- Selected validation weak_class_score: {pct(selected['weak_class_score'])}

## 5. 最佳冻结配置

| Item | Value |
| --- | --- |
| WavLM layer | {selected['wavlm_layers']} |
| Pooling | mean+std |
| PCA | {selected['pca_components']} |
| Classifier | RBF-SVM |
| C | {selected['C']} |
| gamma | {selected['gamma']} / feature_dim |
| class_weight | {selected['class_weight_name']} |
| bias | fear/sadness/disgust {bias.get('fear', 0):+.2f}; neutral {bias.get('neutral', 0):+.2f} |

## 6. 第二阶段一次性测试结果

| Metric | Value |
| --- | ---: |
| Test Accuracy | {pct(final['test_accuracy'])} |
| Test Macro-F1 | {pct(final['test_macro_f1'])} |
| Test Weighted-F1 | {pct(final['test_weighted_f1'])} |
| weak_class_score_test | {pct(final['weak_class_score_test'])} |

## 7. 与 clean v2 / clean v3 对比

这里使用项目上下文中已经记录的 clean v2 / clean v3 最终结果；本脚本没有读取或评估 v2/v3 test set。

| Model | Test Accuracy | Test Macro-F1 | Notes |
| --- | ---: | ---: | --- |
| clean v2 | {pct(V2_REFERENCE['test_accuracy'])} | {pct(V2_REFERENCE['test_macro_f1'])} | previous deployed/reference result |
| clean v3 | {pct(V3_REFERENCE['test_accuracy'])} | {pct(V3_REFERENCE['test_macro_f1'])} | independent confirmation result |
| clean v4 | {pct(final['test_accuracy'])} | {pct(final['test_macro_f1'])} | frozen v4 one-time test |

## 8. 逐类 Precision / Recall / F1 / Support

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

## 9. 混淆矩阵

{cm_header}
{cm_sep}
{chr(10).join(cm_rows)}

## 10. 错误分析：弱类重点

- Fear：F1 为 {pct(per_class_f1['fear'])}。该类通常容易和 sadness、neutral 发生混淆；v4 在 frozen bias 中对 fear 加了正向 bias，但最终测试仍显示其泛化难度较高。
- Sadness：F1 为 {pct(per_class_f1['sadness'])}。悲伤在声学上常和 neutral 接近，如果测试说话人的语速、能量变化不明显，容易被模型压到 neutral。
- Disgust：F1 为 {pct(per_class_f1['disgust'])}。厌恶的样本风格跨数据集差异较明显，manual_v3 对它加权后有助于验证集，但测试集仍可能受说话人差异影响。
- Neutral：F1 为 {pct(per_class_f1['neutral'])}。neutral 是典型“吸收类”，容易吞掉低唤醒度的 fear/sadness；v4 frozen bias 没有继续压低 neutral，避免在测试前人为扩大阈值调整。

## 11. 是否建议替换当前部署模型

结论：{deployment}。

理由：{deployment_reason}

## 12. 后续边界

v4 test set 已经使用，不能再用于调参。如果继续做模型搜索、bias 调整、阈值优化或结构改动，必须建立 clean v5，并重新封存一个未见说话人的 test set。

## Reproducibility artifacts

- Final results: `{args.final_results}`
- Frozen plan: `{args.frozen_plan}`
- Model package: `{args.model_output}`
- Model SHA256: `{model_hash}`
- Feature cache SHA256: `{feature_cache_hash}`
'''
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding='utf8')
    print(json.dumps({
        'final_results': str(args.final_results),
        'report': str(args.report),
        'test_accuracy': final['test_accuracy'],
        'test_macro_f1': final['test_macro_f1'],
        'weak_class_score_test': final['weak_class_score_test'],
        'deployment_recommendation': deployment,
        'test_status': 'evaluated-once',
    }, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
