from __future__ import annotations
import argparse, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.train_speech_model import LABELS, discover_tess, discover_crema, discover_emodb

EMOTION_LABELS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'neutral']
FEAR = LABELS.index('fear')
SADNESS = LABELS.index('sadness')
NEUTRAL = LABELS.index('neutral')

MANUAL_WEIGHTS = {
    'v4_manual_v3': {'anger': 1.0, 'disgust': 1.4, 'fear': 1.8, 'joy': 1.0, 'sadness': 1.5, 'neutral': 0.85},
    'fs_weight_v1': {'anger': 1.0, 'disgust': 1.2, 'fear': 2.0, 'joy': 1.0, 'sadness': 1.8, 'neutral': 0.80},
    'fs_weight_v2': {'anger': 1.0, 'disgust': 1.2, 'fear': 2.2, 'joy': 1.0, 'sadness': 2.0, 'neutral': 0.75},
    'fs_weight_v3': {'anger': 1.0, 'disgust': 1.1, 'fear': 2.4, 'joy': 1.0, 'sadness': 2.2, 'neutral': 0.70},
}

def parse_args():
    p = argparse.ArgumentParser(description='WavLM clean v5 validation-only sweep')
    p.add_argument('--split-file', type=Path, default=PROJECT_ROOT / 'models/speech/wavlm_clean_split_v5.json')
    p.add_argument('--feature-cache', type=Path, default=PROJECT_ROOT / 'models/speech/wavlm_clean_v2_layer_stats.npy')
    p.add_argument('--output-jsonl', type=Path, default=PROJECT_ROOT / 'outputs/speech/wavlm_clean_v5_validation_sweep.jsonl')
    p.add_argument('--summary-md', type=Path, default=PROJECT_ROOT / 'outputs/speech/wavlm_clean_v5_validation_sweep_summary.md')
    p.add_argument('--summary-json', type=Path, default=PROJECT_ROOT / 'models/speech/wavlm_clean_v5_validation_summary.json')
    p.add_argument('--frozen-plan', type=Path, default=PROJECT_ROOT / 'models/speech/wavlm_clean_v5_frozen_evaluation_plan.json')
    p.add_argument('--seed', type=int, default=2028)
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def manual_weight(name: str) -> dict[int, float]:
    return {LABELS.index(k): float(v) for k, v in MANUAL_WEIGHTS[name].items()}

def build_features(cache: np.ndarray, layers: tuple[int, ...]) -> np.ndarray:
    fused = np.asarray(cache[:, layers, :], np.float32).mean(axis=1)
    return fused[:, :1536]

def predict_with_bias(model, X, bias):
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

def top2_contains(model, X, wanted: set[int]) -> np.ndarray:
    scores = model.decision_function(X)
    if scores.ndim != 2:
        return np.zeros(X.shape[0], dtype=bool)
    order = np.argsort(scores, axis=1)[:, -2:]
    cls = model.classes_[order]
    return np.array([len(set(row.tolist()) & wanted) >= 2 for row in cls], dtype=bool)

def compute_metrics(y_true, pred, split, config, run_kind, notes):
    used = [LABELS.index(x) for x in split['emotion_labels']]
    rep = classification_report(y_true, pred, labels=used, target_names=split['emotion_labels'], output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, pred, labels=used)
    label_to_pos = {label: i for i, label in enumerate(split['emotion_labels'])}
    fear_f1 = float(rep['fear']['f1-score'])
    sadness_f1 = float(rep['sadness']['f1-score'])
    fs_score = float((fear_f1 + sadness_f1) / 2.0)
    config_doc = {**config, 'run_kind': run_kind}
    run_id = hashlib.sha256(json.dumps(config_doc, sort_keys=True).encode('utf8')).hexdigest()[:16]
    return {
        'run_id': run_id,
        'protocol_name': 'wavlm_clean_v5',
        'split_file': config['split_file'],
        'used_test_set': False,
        'dataset_names': split['dataset_names'],
        'emotion_labels': split['emotion_labels'],
        'random_seed': config['random_seed'],
        'train_speakers_count': len(split['train_speakers']),
        'val_speakers_count': len(split['val_speakers']),
        'test_speakers_count': len(split['test_speakers']),
        'train_sample_count': split['train_sample_count'],
        'val_sample_count': split['val_sample_count'],
        'test_sample_count': split['test_sample_count'],
        'wavlm_model_name': 'microsoft/wavlm-base-plus',
        'wavlm_layers': list(config['wavlm_layers']),
        'pooling': 'mean_std',
        'feature_dim': config['feature_dim'],
        'pca_dim': config['pca_dim'],
        'classifier': config['classifier'],
        'C': config.get('C'),
        'gamma': config.get('gamma'),
        'class_weight_mode': config['class_weight_mode'],
        'manual_class_weights': config['manual_class_weights'],
        'bias_config': config['bias_config'],
        'local_model_enabled': config.get('local_model_enabled', False),
        'local_model_type': config.get('local_model_type'),
        'local_trigger_rule': config.get('local_trigger_rule'),
        'local_trigger_count_val': config.get('local_trigger_count_val', 0),
        'changed_predictions_count_val': config.get('changed_predictions_count_val', 0),
        'augmentation_enabled': False,
        'augmentation_config': {'name': 'none', 'reason': 'Not run in this phase; WavLM feature extraction for augmented waveforms is expensive and lower priority than bias/weight/local validation.'},
        'prosody_enabled': False,
        'prosody_feature_names': [],
        'val_accuracy': float(accuracy_score(y_true, pred)),
        'val_macro_f1': float(f1_score(y_true, pred, labels=used, average='macro', zero_division=0)),
        'val_weighted_f1': float(f1_score(y_true, pred, average='weighted', zero_division=0)),
        'val_per_class_precision': {x: float(rep[x]['precision']) for x in split['emotion_labels']},
        'val_per_class_recall': {x: float(rep[x]['recall']) for x in split['emotion_labels']},
        'val_per_class_f1': {x: float(rep[x]['f1-score']) for x in split['emotion_labels']},
        'val_confusion_matrix': cm.tolist(),
        'fear_f1': fear_f1,
        'sadness_f1': sadness_f1,
        'fear_recall': float(rep['fear']['recall']),
        'sadness_recall': float(rep['sadness']['recall']),
        'fear_sadness_confusion': int(cm[label_to_pos['fear'], label_to_pos['sadness']]),
        'sadness_fear_confusion': int(cm[label_to_pos['sadness'], label_to_pos['fear']]),
        'sadness_neutral_confusion': int(cm[label_to_pos['sadness'], label_to_pos['neutral']]),
        'fs_score': fs_score,
        'balanced_target_score': float(0.5 * f1_score(y_true, pred, labels=used, average='macro', zero_division=0) + 0.5 * fs_score),
        'model_save_path': None,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'notes': notes,
        'run_kind': run_kind,
    }

def fit_global(Xt, ytr, C, gamma_label, class_weight_mode, seed):
    gamma = 'scale' if gamma_label == 'scale' else float(gamma_label) / Xt.shape[1]
    return SVC(C=C, gamma=gamma, kernel='rbf', class_weight=manual_weight(class_weight_mode), cache_size=1024, decision_function_shape='ovr', random_state=seed).fit(Xt, ytr)

def main():
    args = parse_args()
    for out in [args.output_jsonl, args.summary_md, args.summary_json, args.frozen_plan]:
        if out.exists() and not args.overwrite:
            raise SystemExit(f'Refusing to overwrite existing file; pass --overwrite: {out}')
    split = json.loads(args.split_file.read_text(encoding='utf8'))
    if split.get('protocol_name') != 'wavlm_clean_v5' or split.get('test_status') != 'sealed':
        raise RuntimeError('v5 split must be sealed before validation sweep')
    samples = discover_tess(PROJECT_ROOT / 'datasets/TESS') + discover_crema(PROJECT_ROOT / 'datasets/CREMA-D') + discover_emodb(PROJECT_ROOT / 'datasets/EmoDB')
    sp = np.array([s.speaker for s in samples])
    y = np.array([LABELS.index(s.label) for s in samples])
    cache = np.load(args.feature_cache, mmap_mode='r')
    mf = np.load(PROJECT_ROOT / 'models/speech/multidataset_features.npz')
    if len(cache) != len(samples) or not np.array_equal(sp, mf['speakers'].astype(str)):
        raise RuntimeError('Feature cache and discovered sample order do not match')
    tr = np.flatnonzero(np.isin(sp, split['train_speakers']))
    va = np.flatnonzero(np.isin(sp, split['val_speakers']))
    if len(tr) != split['train_sample_count'] or len(va) != split['val_sample_count']:
        raise RuntimeError('Train/validation counts do not match split file')
    # Deliberately no v5 test index is constructed in this validation-only script.

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.output_jsonl.exists():
        args.output_jsonl.unlink()
    records = []
    bases_for_local = []

    raw = build_features(cache, (7,))
    scaler = StandardScaler()
    Xt0 = scaler.fit_transform(raw[tr])
    Xv0 = scaler.transform(raw[va])
    pca = PCA(n_components=768, random_state=args.seed, svd_solver='randomized')
    Xt = pca.fit_transform(Xt0)
    Xv = pca.transform(Xv0)

    fear_biases = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
    sadness_biases = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
    disgust_biases = [0.00, 0.05, 0.10]
    neutral_biases = [0.00, -0.05, -0.10, -0.15]
    base_models = []
    for weight_name in ['v4_manual_v3', 'fs_weight_v1', 'fs_weight_v2', 'fs_weight_v3']:
        model = fit_global(Xt, y[tr], 3.0, '0.25', weight_name, args.seed)
        base_models.append((weight_name, model, Xt, Xv, (7,), 768))
    with args.output_jsonl.open('w', encoding='utf8') as fout:
        for weight_name, model, Xt_base, Xv_base, layers, pca_dim in base_models:
            for fb in fear_biases:
                for sb in sadness_biases:
                    for db in disgust_biases:
                        for nb in neutral_biases:
                            bias = {'fear': fb, 'sadness': sb, 'disgust': db, 'neutral': nb}
                            pred = predict_with_bias(model, Xv_base, bias)
                            cfg = {'split_file': str(args.split_file.resolve()), 'random_seed': args.seed, 'wavlm_layers': layers, 'feature_dim': 1536, 'pca_dim': pca_dim, 'classifier': 'RBF-SVM', 'C': 3.0, 'gamma': '0.25', 'class_weight_mode': weight_name, 'manual_class_weights': MANUAL_WEIGHTS[weight_name], 'bias_config': bias, 'local_model_enabled': False, 'local_model_type': None, 'local_trigger_rule': None}
                            kind = 'v5_baseline' if (weight_name == 'v4_manual_v3' and bias == {'fear': 0.10, 'sadness': 0.10, 'disgust': 0.10, 'neutral': 0.0}) else ('manual_weight_bias_search' if weight_name != 'v4_manual_v3' else 'fear_sadness_bias_search')
                            rec = compute_metrics(y[va], pred, split, cfg, kind, 'Validation-only global model with per-class bias search; v5 test set not accessed.')
                            fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
                            records.append(rec)
            base_pred = predict_with_bias(model, Xv_base, {'fear': 0.10, 'sadness': 0.10, 'disgust': 0.10, 'neutral': 0.0})
            bases_for_local.append((weight_name, model, Xt_base, Xv_base, layers, pca_dim, base_pred))
            print(json.dumps({'finished_weight': weight_name, 'records': len(records), 'best_fs': round(max(r['fs_score'] for r in records), 4), 'best_macro': round(max(r['val_macro_f1'] for r in records), 4)}, ensure_ascii=False), flush=True)

        # Local FS / FSN second-stage classifiers on top validation-leading bases.
        local_specs = []
        for item in bases_for_local:
            weight_name, model, Xt_base, Xv_base, layers, pca_dim, base_pred = item
            base_record = max([r for r in records if r['class_weight_mode'] == weight_name], key=lambda r: (r['balanced_target_score'], r['fs_score'], r['val_macro_f1']))
            local_specs.append((base_record, item))
        for base_record, item in local_specs:
            weight_name, global_model, Xt_base, Xv_base, layers, pca_dim, _ = item
            base_bias = base_record['bias_config']
            base_pred = predict_with_bias(global_model, Xv_base, base_bias)
            before = {k: base_record[k] for k in ('val_macro_f1', 'fear_f1', 'sadness_f1', 'fear_sadness_confusion', 'sadness_fear_confusion', 'sadness_neutral_confusion')}
            for local_type in ['FS', 'FSN']:
                if local_type == 'FS':
                    train_mask = np.isin(y[tr], [FEAR, SADNESS])
                    trigger = top2_contains(global_model, Xv_base, {FEAR, SADNESS})
                    local_y = y[tr][train_mask]
                    local_classes = [FEAR, SADNESS]
                    trigger_rule = 'global top-2 contains both fear and sadness'
                else:
                    train_mask = np.isin(y[tr], [FEAR, SADNESS, NEUTRAL])
                    top2_fsn = top2_contains(global_model, Xv_base, {FEAR, SADNESS, NEUTRAL})
                    trigger = np.isin(base_pred, [FEAR, SADNESS, NEUTRAL]) | top2_fsn
                    local_y = y[tr][train_mask]
                    local_classes = [FEAR, SADNESS, NEUTRAL]
                    trigger_rule = 'global prediction in fear/sadness/neutral or top-2 contains at least two FSN classes'
                candidates = [
                    ('local_RBF-SVM', SVC(C=1.0, gamma='scale', kernel='rbf', class_weight='balanced', random_state=args.seed)),
                    ('local_LogisticRegression', LogisticRegression(C=1.0, class_weight='balanced', max_iter=3000, random_state=args.seed)),
                    ('local_LinearSVC', LinearSVC(C=0.1, class_weight='balanced', max_iter=5000, dual='auto', random_state=args.seed)),
                ]
                for clf_name, clf in candidates:
                    clf.fit(Xt_base[train_mask], local_y)
                    pred = base_pred.copy()
                    if int(trigger.sum()) > 0:
                        local_pred = clf.predict(Xv_base[trigger])
                        pred[trigger] = local_pred
                    changed = int(np.sum(pred != base_pred))
                    cfg = {'split_file': str(args.split_file.resolve()), 'random_seed': args.seed, 'wavlm_layers': layers, 'feature_dim': 1536, 'pca_dim': pca_dim, 'classifier': 'RBF-SVM + ' + clf_name, 'C': 3.0, 'gamma': '0.25', 'class_weight_mode': weight_name, 'manual_class_weights': MANUAL_WEIGHTS[weight_name], 'bias_config': base_bias, 'local_model_enabled': True, 'local_model_type': local_type, 'local_trigger_rule': trigger_rule, 'local_trigger_count_val': int(trigger.sum()), 'changed_predictions_count_val': changed, 'local_classes': [LABELS[i] for i in local_classes], 'val_macro_f1_before': before['val_macro_f1'], 'fear_f1_before': before['fear_f1'], 'sadness_f1_before': before['sadness_f1'], 'fear_sadness_confusion_before': before['fear_sadness_confusion'], 'sadness_fear_confusion_before': before['sadness_fear_confusion'], 'sadness_neutral_confusion_before': before['sadness_neutral_confusion']}
                    rec = compute_metrics(y[va], pred, split, cfg, 'local_classifier', 'Validation-only local FS/FSN second-stage classifier; v5 test set not accessed.')
                    rec.update({'val_macro_f1_before': before['val_macro_f1'], 'val_macro_f1_after': rec['val_macro_f1'], 'fear_f1_before': before['fear_f1'], 'fear_f1_after': rec['fear_f1'], 'sadness_f1_before': before['sadness_f1'], 'sadness_f1_after': rec['sadness_f1'], 'neutral_f1_before': base_record['val_per_class_f1']['neutral'], 'neutral_f1_after': rec['val_per_class_f1']['neutral'], 'fear_sadness_confusion_before': before['fear_sadness_confusion'], 'fear_sadness_confusion_after': rec['fear_sadness_confusion'], 'sadness_neutral_confusion_before': before['sadness_neutral_confusion'], 'sadness_neutral_confusion_after': rec['sadness_neutral_confusion']})
                    fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    records.append(rec)

    baseline = next(r for r in records if r['run_kind'] == 'v5_baseline')
    min_macro = baseline['val_macro_f1'] - 0.005
    eligible = [r for r in records if r['val_macro_f1'] >= min_macro]
    selected = max(eligible, key=lambda r: (r['fs_score'], r['val_macro_f1'], -int(r['local_model_enabled'])))
    top = sorted(records, key=lambda r: (r['balanced_target_score'], r['fs_score'], r['val_macro_f1']), reverse=True)[:10]
    best_bias = max([r for r in records if r['run_kind'] == 'fear_sadness_bias_search'], key=lambda r: (r['balanced_target_score'], r['fs_score'], r['val_macro_f1']))
    best_weight = max([r for r in records if r['run_kind'] == 'manual_weight_bias_search'], key=lambda r: (r['balanced_target_score'], r['fs_score'], r['val_macro_f1']))
    local_records = [r for r in records if r['run_kind'] == 'local_classifier']
    best_local = max(local_records, key=lambda r: (r['balanced_target_score'], r['fs_score'], r['val_macro_f1'])) if local_records else None
    summary = {'protocol_name': 'wavlm_clean_v5', 'split_file': str(args.split_file.resolve()), 'used_test_set': False, 'test_status': split['test_status'], 'candidate_count': len(records), 'selection_rule': 'Filter candidates with validation Macro-F1 no more than 0.005 below v5 baseline; select highest fs_score, then higher validation Macro-F1, then simpler model.', 'baseline_validation_results': baseline, 'best_bias_search': best_bias, 'best_manual_weight': best_weight, 'best_local_classifier': best_local, 'selected': selected, 'top_10': top, 'validation_sweep_sha256': sha256_file(args.output_jsonl), 'feature_cache_hash': sha256_file(args.feature_cache), 'created_at': datetime.now(timezone.utc).isoformat(), 'augmentation_result': 'not_run', 'prosody_fusion_result': 'not_run'}
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
    frozen = {'protocol_name': 'wavlm_clean_v5', 'selected_run_id': selected['run_id'], 'selected_config': {k: selected[k] for k in ['wavlm_layers','pooling','pca_dim','classifier','C','gamma','class_weight_mode','manual_class_weights','bias_config','local_model_enabled','local_model_type','local_trigger_rule']}, 'selection_metric': 'fs_score among candidates within 0.005 validation Macro-F1 of baseline', 'selection_score': selected['fs_score'], 'validation_results': selected, 'baseline_validation_results': baseline, 'test_status_before_final_eval': split['test_status'], 'statement': 'No test-set metrics were used for model selection.', 'split_file': str(args.split_file.resolve()), 'validation_sweep_jsonl': str(args.output_jsonl.resolve()), 'validation_sweep_sha256': summary['validation_sweep_sha256'], 'created_at': datetime.now(timezone.utc).isoformat()}
    args.frozen_plan.parent.mkdir(parents=True, exist_ok=True)
    args.frozen_plan.write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding='utf8')

    def pct(x): return f'{x*100:.2f}%'
    def row(name, r):
        return f"| {name} | {r['run_kind']} | {r['class_weight_mode']} | {r['bias_config']} | {pct(r['val_accuracy'])} | {pct(r['val_macro_f1'])} | {pct(r['fs_score'])} | {pct(r['fear_f1'])} | {pct(r['sadness_f1'])} | {r['fear_sadness_confusion']} | {r['sadness_fear_confusion']} | {r['sadness_neutral_confusion']} | `{r['run_id']}` |"
    top_rows = '\n'.join(row(str(i+1), r) for i, r in enumerate(top))
    md = f'''# WavLM clean v5 validation sweep summary

生成时间：{datetime.now().isoformat(timespec='seconds')}

## 1. 实验目的

clean v5 的目标是在不使用任何 test metrics 的前提下，重点提升 fear 和 sadness 的验证集表现，降低 fear <-> sadness 以及 sadness -> neutral 混淆，同时尽量保持总体 Accuracy / Macro-F1。

## 2. v5 split

- Dataset: CREMA-D + EmoDB
- TESS excluded: only two speakers, cannot create disjoint train/val/test speaker partitions.
- Train speakers: {len(split['train_speakers'])}, samples: {split['train_sample_count']}
- Validation speakers: {len(split['val_speakers'])}, samples: {split['val_sample_count']}
- Test speakers: {len(split['test_speakers'])}, samples: {split['test_sample_count']}
- Test status: `{split['test_status']}`

## 3. Protocol guardrail

- Every JSONL candidate has `used_test_set=false`.
- This script never constructs a v5 test index.
- No final test was performed.
- v2/v3/v4 test sets were not read or evaluated.
- EmotionRecognition.exe was not repackaged.

## 4. v5 baseline

{row('baseline', baseline)}

## 5. fear/sadness bias search result

{row('best bias search', best_bias)}

## 6. manual weights result

{row('best manual weight', best_weight)}

## 7. local FS / FSN classifier result

{row('best local classifier', best_local) if best_local else 'No local classifier records generated.'}

## 8. augmentation result

Not run in this phase. Reason: train-only weak-class waveform augmentation requires extracting new WavLM features for augmented audio; priority was given to cheaper validation-only bias/weight/local classifier experiments. `augmentation_enabled=false` is recorded for all candidates.

## 9. prosody fusion result

Not run in this phase. Reason: reliable pitch/F0 extraction dependencies were not established in this workspace during this run. `prosody_enabled=false` is recorded for all candidates.

## 10. Top 10 validation candidates

| Rank | Kind | Weight | Bias | Val Acc | Val Macro-F1 | fs_score | fear F1 | sadness F1 | fear->sad | sad->fear | sad->neutral | run_id |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{top_rows}

## 11. Selected frozen validation config

{row('selected', selected)}

## 12. Why selected

Selection rule: candidates must keep validation Macro-F1 within 0.5 percentage points of the v5 baseline. Among those, the candidate with the highest `fs_score = mean(fear F1, sadness F1)` is selected; if tied, higher Macro-F1 and then simpler inference are preferred.

Compared with baseline:

- Fear F1: {pct(baseline['fear_f1'])} -> {pct(selected['fear_f1'])}
- Sadness F1: {pct(baseline['sadness_f1'])} -> {pct(selected['sadness_f1'])}
- fs_score: {pct(baseline['fs_score'])} -> {pct(selected['fs_score'])}
- Val Macro-F1: {pct(baseline['val_macro_f1'])} -> {pct(selected['val_macro_f1'])}

## 13. Final-test boundary

No final test was performed. v5 test remains sealed. If final testing is needed, run a separate second-stage command that loads `models/speech/wavlm_clean_v5_frozen_evaluation_plan.json` and evaluates the sealed v5 test set exactly once.
'''
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text(md, encoding='utf8')
    print(json.dumps({'candidate_count': len(records), 'baseline_macro_f1': baseline['val_macro_f1'], 'baseline_fs_score': baseline['fs_score'], 'selected_run_id': selected['run_id'], 'selected_macro_f1': selected['val_macro_f1'], 'selected_fs_score': selected['fs_score'], 'test_status': split['test_status'], 'frozen_plan': str(args.frozen_plan)}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
