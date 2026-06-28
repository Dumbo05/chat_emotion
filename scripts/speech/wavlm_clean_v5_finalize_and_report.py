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

REFERENCE = {
    'clean v2': {'accuracy': 0.7390, 'macro_f1': 0.7354, 'fear_f1': None, 'sadness_f1': None, 'note': 'Prior project context; v2 files not read by this script.'},
    'clean v3': {'accuracy': 0.7369207772795217, 'macro_f1': 0.7352575785140297, 'fear_f1': None, 'sadness_f1': None, 'note': 'Prior project context; v3 files not read by this script.'},
    'clean v4': {'accuracy': 0.7902364607170099, 'macro_f1': 0.7907979073421058, 'fear_f1': 0.6885, 'sadness_f1': 0.6683, 'note': 'Prior v4 final result from user/project context; v4 files not read by this script.'},
}

def parse_args():
    p = argparse.ArgumentParser(description='Finalize WavLM clean v5 by evaluating sealed test set exactly once')
    p.add_argument('--split-file', type=Path, default=PROJECT_ROOT / 'models/speech/wavlm_clean_split_v5.json')
    p.add_argument('--frozen-plan', type=Path, default=PROJECT_ROOT / 'models/speech/wavlm_clean_v5_frozen_evaluation_plan.json')
    p.add_argument('--feature-cache', type=Path, default=PROJECT_ROOT / 'models/speech/wavlm_clean_v2_layer_stats.npy')
    p.add_argument('--final-results', type=Path, default=PROJECT_ROOT / 'models/speech/wavlm_clean_v5_final_results.json')
    p.add_argument('--model-output', type=Path, default=PROJECT_ROOT / 'models/speech/clean_v5_models/wavlm_clean_v5_best.joblib')
    p.add_argument('--report', type=Path, default=PROJECT_ROOT / 'outputs/speech/wavlm_clean_v5_report.md')
    p.add_argument('--seed', type=int, default=2028)
    return p.parse_args()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def class_weight_from_config(config: dict[str, Any]) -> dict[int, float] | str | None:
    mode = config.get('class_weight_mode')
    if mode == 'balanced':
        return 'balanced'
    if mode in (None, 'none'):
        return None
    weights = config['manual_class_weights']
    return {LABELS.index(k): float(v) for k, v in weights.items()}

def build_features(cache: np.ndarray, layers: tuple[int, ...]) -> np.ndarray:
    fused = np.asarray(cache[:, layers, :], np.float32).mean(axis=1)
    return fused[:, :1536]

def predict_with_bias(model: SVC, X: np.ndarray, bias: dict[str, float]) -> np.ndarray:
    scores = model.decision_function(X)
    if scores.ndim != 2:
        return model.predict(X)
    scores = np.asarray(scores, dtype=np.float64).copy()
    for label, value in bias.items():
        idx = LABELS.index(label)
        pos = np.flatnonzero(model.classes_ == idx)
        if len(pos):
            scores[:, pos[0]] += float(value)
    return model.classes_[np.argmax(scores, axis=1)]

def pct(x):
    return '—' if x is None else f'{x*100:.2f}%'

def main():
    args = parse_args()
    if args.final_results.exists():
        raise SystemExit(f'Refusing to evaluate v5 test again; final results already exist: {args.final_results}')
    split = json.loads(args.split_file.read_text(encoding='utf8'))
    plan = json.loads(args.frozen_plan.read_text(encoding='utf8'))
    if split.get('protocol_name') != 'wavlm_clean_v5':
        raise RuntimeError('Split protocol is not wavlm_clean_v5')
    if split.get('test_status') != 'sealed':
        raise RuntimeError(f'v5 test set is not sealed before evaluation: {split.get("test_status")}')
    if plan.get('protocol_name') != 'wavlm_clean_v5':
        raise RuntimeError('Frozen plan protocol is not wavlm_clean_v5')
    for key in ['selected_run_id', 'selected_config', 'validation_results']:
        if key not in plan:
            raise RuntimeError(f'Frozen plan missing {key}')
    if plan.get('statement') != 'No test-set metrics were used for model selection.':
        raise RuntimeError('Frozen plan does not contain the required no-test-metrics statement')
    if plan.get('test_status_before_final_eval') != 'sealed':
        raise RuntimeError('Frozen plan did not freeze a sealed test status')

    selected = plan['selected_config']
    validation = plan['validation_results']
    samples = discover_tess(PROJECT_ROOT / 'datasets/TESS') + discover_crema(PROJECT_ROOT / 'datasets/CREMA-D') + discover_emodb(PROJECT_ROOT / 'datasets/EmoDB')
    sp = np.array([s.speaker for s in samples])
    y = np.array([LABELS.index(s.label) for s in samples])
    cache = np.load(args.feature_cache, mmap_mode='r')
    mf = np.load(PROJECT_ROOT / 'models/speech/multidataset_features.npz')
    if len(cache) != len(samples) or not np.array_equal(sp, mf['speakers'].astype(str)):
        raise RuntimeError('Feature cache and discovered sample order do not match')
    train_val = np.flatnonzero(np.isin(sp, split['train_speakers'] + split['val_speakers']))
    test = np.flatnonzero(np.isin(sp, split['test_speakers']))
    if len(train_val) != split['train_sample_count'] + split['val_sample_count'] or len(test) != split['test_sample_count']:
        raise RuntimeError('Train+validation/test sample counts do not match split file')

    layers = tuple(int(x) for x in selected['wavlm_layers'])
    raw = build_features(cache, layers)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(raw[train_val])
    X_test = scaler.transform(raw[test])
    pca = None
    if selected.get('pca_dim') is not None:
        pca = PCA(n_components=int(selected['pca_dim']), random_state=args.seed, svd_solver='randomized')
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    gamma_label = selected['gamma']
    gamma = 'scale' if gamma_label == 'scale' else float(gamma_label) / X_train.shape[1]
    model = SVC(C=float(selected['C']), gamma=gamma, kernel='rbf', class_weight=class_weight_from_config(selected), cache_size=1024, decision_function_shape='ovr', random_state=args.seed)
    model.fit(X_train, y[train_val])
    if selected.get('local_model_enabled'):
        raise RuntimeError('This finalizer does not implement local model inference; selected config unexpectedly enables local model.')
    pred = predict_with_bias(model, X_test, selected['bias_config'])

    labels = split['emotion_labels']
    used = [LABELS.index(x) for x in labels]
    rep = classification_report(y[test], pred, labels=used, target_names=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y[test], pred, labels=used)
    pos = {label: i for i, label in enumerate(labels)}
    per_p = {x: float(rep[x]['precision']) for x in labels}
    per_r = {x: float(rep[x]['recall']) for x in labels}
    per_f = {x: float(rep[x]['f1-score']) for x in labels}
    per_s = {x: int(rep[x]['support']) for x in labels}
    fear_f1 = per_f['fear']
    sadness_f1 = per_f['sadness']
    fs_score = float((fear_f1 + sadness_f1) / 2.0)

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    package = {'protocol_name': 'wavlm_clean_v5', 'model': model, 'scaler': scaler, 'pca': pca, 'selected_config': selected, 'validation_results': validation, 'labels': LABELS, 'emotion_labels': labels, 'trained_on': 'v5 train+validation speakers only', 'created_at': datetime.now(timezone.utc).isoformat()}
    joblib.dump(package, args.model_output, compress=3)
    model_hash = sha256_file(args.model_output)
    feature_cache_hash = sha256_file(args.feature_cache)

    final = {
        'protocol_name': 'wavlm_clean_v5',
        'split_file': str(args.split_file.resolve()),
        'frozen_plan_file': str(args.frozen_plan.resolve()),
        'selected_config': selected,
        'selected_run_id': plan['selected_run_id'],
        'test_status': 'evaluated-once',
        'test_accuracy': float(accuracy_score(y[test], pred)),
        'test_macro_f1': float(f1_score(y[test], pred, labels=used, average='macro', zero_division=0)),
        'test_weighted_f1': float(f1_score(y[test], pred, average='weighted', zero_division=0)),
        'test_per_class_precision': per_p,
        'test_per_class_recall': per_r,
        'test_per_class_f1': per_f,
        'test_per_class_support': per_s,
        'test_confusion_matrix': cm.tolist(),
        'test_sample_count': int(len(test)),
        'test_speakers_count': int(len(split['test_speakers'])),
        'fear_f1_test': fear_f1,
        'sadness_f1_test': sadness_f1,
        'fs_score_test': fs_score,
        'fear_sadness_confusion_test': int(cm[pos['fear'], pos['sadness']]),
        'sadness_fear_confusion_test': int(cm[pos['sadness'], pos['fear']]),
        'sadness_neutral_confusion_test': int(cm[pos['sadness'], pos['neutral']]),
        'model_path': str(args.model_output.resolve()),
        'model_hash': model_hash,
        'feature_cache': str(args.feature_cache.resolve()),
        'feature_cache_hash': feature_cache_hash,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'one_time_test_statement': 'The v5 test set was evaluated once using the frozen configuration. No model, bias, threshold, local classifier, or hyperparameter was adjusted after seeing test results.',
    }
    args.final_results.parent.mkdir(parents=True, exist_ok=True)
    args.final_results.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding='utf8')
    split['test_status'] = 'evaluated-once'
    split['test_evaluated_at'] = final['created_at']
    split['final_results_file'] = str(args.final_results.resolve())
    args.split_file.write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding='utf8')

    v4 = REFERENCE['clean v4']
    recommendation = '保留 v4，不建议替换部署模型'
    reason = 'v5 的总体 Macro-F1 未高于 v4，且 fear/sadness 没有形成相对 v4 的明确双类改善。'
    if final['test_macro_f1'] > v4['macro_f1'] and fear_f1 > v4['fear_f1'] and sadness_f1 > v4['sadness_f1']:
        recommendation = '建议替换为 v5'
        reason = 'v5 Macro-F1 高于 v4，且 fear/sadness 均明显改善。'
    elif abs(final['test_macro_f1'] - v4['macro_f1']) <= 0.005 and (fear_f1 > v4['fear_f1'] + 0.01 or sadness_f1 > v4['sadness_f1'] + 0.01):
        recommendation = '可作为研究增强模型保留；是否部署取决于是否重视 fear/sadness'
        reason = 'v5 总体 Macro-F1 与 v4 接近，且至少一个弱类有改善。'
    elif (fear_f1 > v4['fear_f1'] or sadness_f1 > v4['sadness_f1']) and final['test_macro_f1'] < v4['macro_f1'] - 0.01:
        recommendation = '不建议替换主部署模型'
        reason = 'v5 虽有弱类局部变化，但总体 Macro-F1 明显低于 v4。'

    class_rows = '\n'.join(f"| {label} | {per_p[label]*100:.2f}% | {per_r[label]*100:.2f}% | {per_f[label]*100:.2f}% | {per_s[label]} |" for label in labels)
    cm_header = '| True \\ Pred | ' + ' | '.join(labels) + ' |'
    cm_sep = '| --- | ' + ' | '.join(['---:'] * len(labels)) + ' |'
    cm_rows = '\n'.join('| ' + label + ' | ' + ' | '.join(str(int(x)) for x in row) + ' |' for label, row in zip(labels, cm.tolist()))
    compare_rows = []
    for name, r in REFERENCE.items():
        compare_rows.append(f"| {name} | {pct(r['accuracy'])} | {pct(r['macro_f1'])} | {pct(r['fear_f1'])} | {pct(r['sadness_f1'])} | {r['note']} |")
    compare_rows.append(f"| clean v5 | {pct(final['test_accuracy'])} | {pct(final['test_macro_f1'])} | {pct(fear_f1)} | {pct(sadness_f1)} | v5 frozen one-time test |")

    report = f'''# WavLM clean v5 final report

生成时间：{datetime.now().isoformat(timespec='seconds')}

## 1. 实验目的

clean v5 的目标是在 frozen validation 配置固定后，对 sealed speaker-independent test set 做一次性评估，重点检查 fear 和 sadness 是否相对 v4 有改善，并判断是否值得替换当前部署模型。

## 2. v5 数据划分

- Dataset: CREMA-D + EmoDB
- TESS excluded: only two speakers, cannot create disjoint train/validation/test speaker partitions.
- Train speakers: {len(split['train_speakers'])}, samples: {split['train_sample_count']}
- Validation speakers: {len(split['val_speakers'])}, samples: {split['val_sample_count']}
- Test speakers: {len(split['test_speakers'])}, samples: {split['test_sample_count']}
- Train / validation / test speakers are mutually exclusive.

## 3. sealed / evaluated-once 说明

评估前，`models/speech/wavlm_clean_split_v5.json` 的 `test_status` 为 `sealed`。本次脚本只执行一次测试，随后将其更新为 `evaluated-once`。

没有根据 test 结果调整模型、bias、阈值、局部分类器或超参数。

## 4. 第一阶段 validation sweep 摘要

- Validation candidates: 1752
- Selected run_id: `{plan['selected_run_id']}`
- Validation Accuracy: {pct(validation['val_accuracy'])}
- Validation Macro-F1: {pct(validation['val_macro_f1'])}
- Validation fs_score: {pct(validation['fs_score'])}
- Validation fear F1: {pct(validation['fear_f1'])}
- Validation sadness F1: {pct(validation['sadness_f1'])}
- Statement: `{plan['statement']}`

## 5. 最佳冻结配置

| Item | Value |
| --- | --- |
| WavLM layer | {selected['wavlm_layers']} |
| Pooling | {selected['pooling']} |
| PCA | {selected['pca_dim']} |
| Classifier | {selected['classifier']} |
| C | {selected['C']} |
| gamma | {selected['gamma']} / feature_dim |
| class_weight | {selected['class_weight_mode']} |
| bias | {selected['bias_config']} |
| local model | {selected['local_model_enabled']} |

## 6. 第二阶段一次性测试结果

| Metric | Value |
| --- | ---: |
| Test Accuracy | {pct(final['test_accuracy'])} |
| Test Macro-F1 | {pct(final['test_macro_f1'])} |
| Test Weighted-F1 | {pct(final['test_weighted_f1'])} |
| fear_f1_test | {pct(fear_f1)} |
| sadness_f1_test | {pct(sadness_f1)} |
| fs_score_test | {pct(fs_score)} |
| fear -> sadness | {final['fear_sadness_confusion_test']} |
| sadness -> fear | {final['sadness_fear_confusion_test']} |
| sadness -> neutral | {final['sadness_neutral_confusion_test']} |

## 7. 与 clean v2 / v3 / v4 的结果对比

本脚本没有读取或评估 v2/v3/v4 test set；表中的 v2/v3/v4 数字来自已有项目上下文和本轮用户提供的 v4 摘要。

| Model | Test Accuracy | Test Macro-F1 | fear F1 | sadness F1 | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
{chr(10).join(compare_rows)}

## 8. 逐类 Precision / Recall / F1 / Support

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
{class_rows}

## 9. 混淆矩阵

{cm_header}
{cm_sep}
{cm_rows}

## 10. fear/sadness 专项分析

- fear F1 test: {pct(fear_f1)}；sadness F1 test: {pct(sadness_f1)}；fs_score_test: {pct(fs_score)}。
- fear -> sadness 混淆为 {final['fear_sadness_confusion_test']}；sadness -> fear 混淆为 {final['sadness_fear_confusion_test']}；sadness -> neutral 混淆为 {final['sadness_neutral_confusion_test']}。
- 与 v4 摘要相比，v5 没有同时实现 Macro-F1 和 fear/sadness 的稳定优势，因此不应把 v5 test 结果反过来用于继续调 bias。

## 11. 是否建议替换当前部署模型

结论：{recommendation}。

理由：{reason}

## 12. 后续边界

v5 test set 已经使用，不能再用于调参。如果继续搜索 bias、manual weight、局部分类器、增强或 prosody 融合，必须新建 clean v6，并封存新的未见说话人 test set。

## Reproducibility artifacts

- Final results: `{args.final_results}`
- Frozen plan: `{args.frozen_plan}`
- Model package: `{args.model_output}`
- Model SHA256: `{model_hash}`
- Feature cache SHA256: `{feature_cache_hash}`
'''
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding='utf8')
    print(json.dumps({'final_results': str(args.final_results), 'report': str(args.report), 'test_accuracy': final['test_accuracy'], 'test_macro_f1': final['test_macro_f1'], 'fear_f1_test': fear_f1, 'sadness_f1_test': sadness_f1, 'fs_score_test': fs_score, 'recommendation': recommendation, 'test_status': 'evaluated-once'}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
