from __future__ import annotations
import argparse, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
EMOTION_LABELS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'neutral']

MANUAL_CLASS_WEIGHTS = {
    'none': None,
    'balanced': 'balanced',
    'manual_v1': {'fear': 1.20, 'sadness': 1.20, 'disgust': 1.15, 'neutral': 0.90, 'anger': 1.00, 'joy': 1.00},
    'manual_v2': {'fear': 1.35, 'sadness': 1.30, 'disgust': 1.25, 'neutral': 0.80, 'anger': 1.00, 'joy': 1.00},
    'manual_v3': {'fear': 1.50, 'sadness': 1.40, 'disgust': 1.35, 'neutral': 0.75, 'anger': 0.95, 'joy': 0.95},
}

def parse_args():
    p = argparse.ArgumentParser(description='Validation-only WavLM clean v4 sweep; v4 test remains sealed')
    p.add_argument('--split-file', type=Path, default=PROJECT_ROOT / 'models/speech/wavlm_clean_split_v4.json')
    p.add_argument('--output-jsonl', type=Path, default=PROJECT_ROOT / 'outputs/speech/wavlm_clean_v4_validation_sweep.jsonl')
    p.add_argument('--output-summary-md', type=Path, default=PROJECT_ROOT / 'outputs/speech/wavlm_clean_v4_validation_sweep_summary.md')
    p.add_argument('--output-summary-json', type=Path, default=PROJECT_ROOT / 'models/speech/wavlm_clean_v4_validation_summary.json')
    p.add_argument('--frozen-plan', type=Path, default=PROJECT_ROOT / 'models/speech/wavlm_clean_v4_frozen_evaluation_plan.json')
    p.add_argument('--feature-cache', type=Path, default=PROJECT_ROOT / 'models/speech/wavlm_clean_v2_layer_stats.npy')
    p.add_argument('--seed', type=int, default=2027)
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def class_weight_value(name: str) -> Any:
    raw = MANUAL_CLASS_WEIGHTS[name]
    if raw is None or raw == 'balanced':
        return raw
    return {LABELS.index(label): float(weight) for label, weight in raw.items()}

def class_weight_for_record(name: str) -> Any:
    raw = MANUAL_CLASS_WEIGHTS[name]
    return raw

def make_biases(weak_bias: float, neutral_bias: float) -> dict[int, float]:
    biases = {LABELS.index('fear'): weak_bias, LABELS.index('sadness'): weak_bias, LABELS.index('disgust'): weak_bias, LABELS.index('neutral'): neutral_bias}
    return biases

def predict_with_bias(model: SVC, Xv: np.ndarray, weak_bias: float, neutral_bias: float) -> np.ndarray:
    scores = model.decision_function(Xv)
    if scores.ndim != 2:
        return model.predict(Xv)
    scores = np.asarray(scores, dtype=np.float64).copy()
    biases = make_biases(weak_bias, neutral_bias)
    for class_index, bias in biases.items():
        matches = np.flatnonzero(model.classes_ == class_index)
        if len(matches):
            scores[:, matches[0]] += bias
    return model.classes_[np.argmax(scores, axis=1)]

def build_features(cache: np.ndarray, layers: tuple[int, ...], pooling: str) -> np.ndarray:
    fused = np.asarray(cache[:, layers, :], np.float32).mean(axis=1)
    if pooling == 'mean':
        return fused[:, :768]
    if pooling == 'mean_std':
        return fused[:, :1536]
    raise ValueError(pooling)

def evaluate_prediction(y_true: np.ndarray, pred: np.ndarray, split: dict, config: dict, seed: int) -> dict:
    used = [LABELS.index(x) for x in split['emotion_labels']]
    rep = classification_report(y_true, pred, labels=used, target_names=split['emotion_labels'], output_dict=True, zero_division=0)
    per_class_f1 = {x: float(rep[x]['f1-score']) for x in split['emotion_labels']}
    weak_class_score = float(np.mean([per_class_f1[x] for x in WEAK_CLASSES]))
    config_doc = dict(config)
    run_id = hashlib.sha256(json.dumps(config_doc, sort_keys=True).encode('utf8')).hexdigest()[:16]
    return {
        'run_id': run_id,
        'protocol_name': 'wavlm_clean_v4',
        'used_test_set': False,
        'test_metrics_used': False,
        'split_file': str(config['split_file']),
        'dataset_names': split['dataset_names'],
        'emotion_labels': split['emotion_labels'],
        'weak_classes': WEAK_CLASSES,
        'random_seed': seed,
        'train_speakers_count': len(split['train_speakers']),
        'val_speakers_count': len(split['val_speakers']),
        'sealed_test_speakers_count': len(split['test_speakers']),
        'train_sample_count': split['sample_counts']['train'],
        'val_sample_count': split['sample_counts']['validation'],
        'sealed_test_sample_count': split['sample_counts']['test'],
        'wavlm_model_name': 'microsoft/wavlm-base-plus',
        'wavlm_layers': list(config['layers']),
        'layer_fusion': 'arithmetic_mean',
        'pooling': config['pooling'],
        'feature_dim_before_pca': config['feature_dim_before_pca'],
        'pca_components': config['pca_components'],
        'classifier': 'rbf_svm',
        'C': config['C'],
        'gamma': config['gamma'],
        'class_weight_name': config['class_weight_name'],
        'class_weight': config['class_weight_record'],
        'bias_search': {'fear': config['weak_bias'], 'sadness': config['weak_bias'], 'disgust': config['weak_bias'], 'neutral': config['neutral_bias']},
        'scaler_type': 'StandardScaler fitted on train only',
        'pca_type': None if config['pca_components'] is None else 'PCA fitted on train only',
        'val_accuracy': float(accuracy_score(y_true, pred)),
        'val_macro_f1': float(f1_score(y_true, pred, labels=used, average='macro', zero_division=0)),
        'val_weighted_f1': float(f1_score(y_true, pred, average='weighted', zero_division=0)),
        'weak_class_score': weak_class_score,
        'val_per_class_precision': {x: float(rep[x]['precision']) for x in split['emotion_labels']},
        'val_per_class_recall': {x: float(rep[x]['recall']) for x in split['emotion_labels']},
        'val_per_class_f1': per_class_f1,
        'val_confusion_matrix': confusion_matrix(y_true, pred, labels=used).tolist(),
        'created_at': datetime.now(timezone.utc).isoformat(),
        'notes': 'Validation-only v4 candidate. The v4 test index and test metrics were not constructed or evaluated.',
    }

def main():
    args = parse_args()
    for out in [args.output_jsonl, args.output_summary_md, args.output_summary_json, args.frozen_plan]:
        if out.exists() and not args.overwrite:
            raise SystemExit(f'Refusing to overwrite existing file; pass --overwrite: {out}')
    split = json.loads(args.split_file.read_text(encoding='utf8'))
    if split.get('protocol_name') != 'wavlm_clean_v4' or split.get('test_status') != 'sealed':
        raise RuntimeError('v4 split must exist with test_status="sealed" before validation sweep')
    if split.get('emotion_labels') != EMOTION_LABELS:
        raise RuntimeError('Unexpected v4 emotion labels')

    samples = discover_tess(PROJECT_ROOT / 'datasets/TESS') + discover_crema(PROJECT_ROOT / 'datasets/CREMA-D') + discover_emodb(PROJECT_ROOT / 'datasets/EmoDB')
    sp = np.array([s.speaker for s in samples])
    y = np.array([LABELS.index(s.label) for s in samples])
    cache = np.load(args.feature_cache, mmap_mode='r')
    mf = np.load(PROJECT_ROOT / 'models/speech/multidataset_features.npz')
    if len(cache) != len(samples) or not np.array_equal(sp, mf['speakers'].astype(str)):
        raise RuntimeError('Feature cache and discovered sample order do not match')
    tr = np.flatnonzero(np.isin(sp, split['train_speakers']))
    va = np.flatnonzero(np.isin(sp, split['val_speakers']))
    if len(tr) != split['sample_counts']['train'] or len(va) != split['sample_counts']['validation']:
        raise RuntimeError('Train/validation sample counts do not match split file')
    # Deliberately never construct a v4 test index and never compute test-set metrics here.

    # v4 staged validation sweep: keep the user-prioritized candidates, but avoid an
    # exhaustive RBF-SVM grid that is prohibitively slow on 5k+ speaker-independent
    # training samples. Biases are swept only after fitting each prioritized base model.
    layer_groups = [(7,), (7, 8), (7, 9), (7, 8, 9)]
    pca_options = [None, 512, 768]
    class_weight_names = ['balanced', 'manual_v1', 'manual_v2', 'manual_v3']
    C_values = [3.0, 5.0]
    gamma_values = ['0.25', 'scale']
    weak_bias_values = [0.0, 0.10, 0.20]
    neutral_bias_values = [0.0, -0.10, -0.20]

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.frozen_plan.parent.mkdir(parents=True, exist_ok=True)
    if args.output_jsonl.exists():
        args.output_jsonl.unlink()

    records = []
    with args.output_jsonl.open('w', encoding='utf8') as fout:
        for layers in layer_groups:
            raw = build_features(cache, layers, 'mean_std')
            scaler = StandardScaler()
            Xt0 = scaler.fit_transform(raw[tr])
            Xv0 = scaler.transform(raw[va])
            for pca_components in pca_options:
                if pca_components is None:
                    Xt, Xv = Xt0, Xv0
                else:
                    pca = PCA(n_components=pca_components, random_state=args.seed, svd_solver='randomized')
                    Xt = pca.fit_transform(Xt0)
                    Xv = pca.transform(Xv0)
                feature_dim = int(Xt.shape[1])
                for class_weight_name in class_weight_names:
                    # Raw 1536-dim RBF-SVM is very slow; run it only for the required
                    # layer-7 balanced baseline. Manual weights and layer fusions are
                    # evaluated through train-only PCA 512/768.
                    if pca_components is None and not (layers == (7,) and class_weight_name == 'balanced'):
                        continue
                    cw = class_weight_value(class_weight_name)
                    for C in C_values:
                        if pca_components is None and C != 3.0:
                            continue
                        for gamma_label in gamma_values:
                            if pca_components is None and gamma_label != '0.25':
                                continue
                            gamma = 'scale' if gamma_label == 'scale' else float(gamma_label) / feature_dim
                            model = SVC(C=C, gamma=gamma, kernel='rbf', class_weight=cw, cache_size=1024, decision_function_shape='ovr', random_state=args.seed)
                            model.fit(Xt, y[tr])
                            for weak_bias in weak_bias_values:
                                for neutral_bias in neutral_bias_values:
                                    config = {
                                        'split_file': str(args.split_file.resolve()),
                                        'layers': layers,
                                        'pooling': 'mean_std',
                                        'feature_dim_before_pca': int(Xt0.shape[1]),
                                        'pca_components': pca_components,
                                        'C': C,
                                        'gamma': gamma_label,
                                        'class_weight_name': class_weight_name,
                                        'class_weight_record': class_weight_for_record(class_weight_name),
                                        'weak_bias': weak_bias,
                                        'neutral_bias': neutral_bias,
                                    }
                                    pred = predict_with_bias(model, Xv, weak_bias, neutral_bias)
                                    rec = evaluate_prediction(y[va], pred, split, config, args.seed)
                                    if layers == (7,) and pca_components is None and class_weight_name == 'balanced' and C == 3.0 and gamma_label == '0.25' and weak_bias == 0.0 and neutral_bias == 0.0:
                                        rec['candidate_tag'] = 'v4-baseline'
                                    fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
                                    records.append(rec)
            current_best = max(records, key=lambda r: (r['val_macro_f1'], r['weak_class_score'], r['val_accuracy']))
            print(json.dumps({'finished_layers': list(layers), 'records_so_far': len(records), 'current_best_macro_f1': round(current_best['val_macro_f1'], 4), 'current_best_weak': round(current_best['weak_class_score'], 4)}, ensure_ascii=False), flush=True)

    best_macro = max(r['val_macro_f1'] for r in records)
    eligible = [r for r in records if r['val_macro_f1'] >= best_macro - 0.005]
    selected = max(eligible, key=lambda r: (r['weak_class_score'], r['val_macro_f1'], r['val_accuracy']))
    ranked_macro = sorted(records, key=lambda r: (r['val_macro_f1'], r['weak_class_score'], r['val_accuracy']), reverse=True)
    ranked_selected = sorted(records, key=lambda r: (r['weak_class_score'] if r['val_macro_f1'] >= best_macro - 0.005 else -1, r['val_macro_f1'], r['val_accuracy']), reverse=True)
    baseline = next((r for r in records if r.get('candidate_tag') == 'v4-baseline'), None)

    summary = {
        'protocol_name': 'wavlm_clean_v4',
        'split_file': str(args.split_file.resolve()),
        'used_test_set': False,
        'test_metrics_used': False,
        'test_status': split['test_status'],
        'selection_rule': 'Find best validation Macro-F1; among candidates within 0.005 absolute Macro-F1 of that best value, choose highest weak_class_score, then Macro-F1, then Accuracy.',
        'weak_classes': WEAK_CLASSES,
        'candidate_count': len(records),
        'feature_cache': str(args.feature_cache.resolve()),
        'feature_cache_size_bytes': args.feature_cache.stat().st_size,
        'validation_jsonl_sha256': sha256_file(args.output_jsonl),
        'best_by_macro_f1': ranked_macro[0],
        'selected_best_config': selected,
        'baseline_config': baseline,
        'top_10_by_macro_f1': ranked_macro[:10],
        'top_10_by_selection_rule': ranked_selected[:10],
        'created_at': datetime.now(timezone.utc).isoformat(),
    }
    args.output_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')

    frozen_plan = {
        'protocol_name': 'wavlm_clean_v4',
        'frozen_at': datetime.now(timezone.utc).isoformat(),
        'split_file': str(args.split_file.resolve()),
        'split_sha256': sha256_file(args.split_file),
        'validation_sweep_jsonl': str(args.output_jsonl.resolve()),
        'validation_sweep_jsonl_sha256': summary['validation_jsonl_sha256'],
        'test_status_at_freeze': split['test_status'],
        'used_test_set': False,
        'test_metrics_used': False,
        'test_evaluation_status': 'not_evaluated',
        'statement': 'No v4 test-set metrics were computed or used. Model selection used validation Macro-F1 and weak_class_score only.',
        'selection_rule': summary['selection_rule'],
        'selected_validation_result': selected,
        'sealed_test_speakers_count': len(split['test_speakers']),
        'sealed_test_sample_count': split['sample_counts']['test'],
        'future_test_instruction': 'For final testing, load this frozen plan, refit the selected configuration on v4 train+validation speakers, and evaluate the sealed v4 test set exactly once.',
    }
    args.frozen_plan.write_text(json.dumps(frozen_plan, ensure_ascii=False, indent=2), encoding='utf8')

    def pct(x: float) -> str:
        return f'{x * 100:.2f}%'
    rows = []
    for i, r in enumerate(ranked_macro[:10], 1):
        rows.append('| {rank} | {layers} | {pca} | {weight} | C={C:g}, gamma={gamma} | weak={wb:+.2f}, neutral={nb:+.2f} | {acc} | {mf1} | {weak} | {run} |'.format(
            rank=i, layers='+'.join(map(str, r['wavlm_layers'])), pca='none' if r['pca_components'] is None else r['pca_components'], weight=r['class_weight_name'], C=r['C'], gamma=r['gamma'], wb=r['bias_search']['fear'], nb=r['bias_search']['neutral'], acc=pct(r['val_accuracy']), mf1=pct(r['val_macro_f1']), weak=pct(r['weak_class_score']), run=r['run_id']))
    selected_line = '| selected | {layers} | {pca} | {weight} | C={C:g}, gamma={gamma} | weak={wb:+.2f}, neutral={nb:+.2f} | {acc} | {mf1} | {weak} | {run} |'.format(
        layers='+'.join(map(str, selected['wavlm_layers'])), pca='none' if selected['pca_components'] is None else selected['pca_components'], weight=selected['class_weight_name'], C=selected['C'], gamma=selected['gamma'], wb=selected['bias_search']['fear'], nb=selected['bias_search']['neutral'], acc=pct(selected['val_accuracy']), mf1=pct(selected['val_macro_f1']), weak=pct(selected['weak_class_score']), run=selected['run_id'])
    baseline_text = 'not found'
    if baseline:
        baseline_text = f"Accuracy {pct(baseline['val_accuracy'])}, Macro-F1 {pct(baseline['val_macro_f1'])}, weak_class_score {pct(baseline['weak_class_score'])}, run_id `{baseline['run_id']}`"
    md = f'''# WavLM clean v4 validation sweep summary

生成时间：{datetime.now().isoformat(timespec='seconds')}

## Protocol guardrail

- v4 test_status: `{split['test_status']}`
- `used_test_set=false` for every validation candidate.
- No v4 test-set metrics were computed or used.
- No v2/v3 test-set files or metrics are required by this v4 sweep.
- EXE was not repackaged.

## Split

- Train speakers: {len(split['train_speakers'])}, validation speakers: {len(split['val_speakers'])}, sealed test speakers: {len(split['test_speakers'])}
- Train samples: {split['sample_counts']['train']}, validation samples: {split['sample_counts']['validation']}, sealed test samples: {split['sample_counts']['test']}
- Datasets: CREMA-D + EmoDB; TESS excluded because it has only two speakers.

## Search space

- Features: WavLM mean+std pooling
- Layers: `[7]`, `[7,8]`, `[7,9]`, `[7,8,9]`
- PCA: none for required layer-7 baseline; 512 and 768 for manual weights and layer fusion candidates
- Class weights: balanced, manual_v1, manual_v2, manual_v3
- RBF-SVM: C = 3, 5; gamma = scale or 0.25/feature_dim; baseline fixed at C=3, gamma=0.25/feature_dim
- Bias search: fear/sadness/disgust positive bias; neutral negative bias
- Candidate records: {len(records)}

## Selection rule

{summary['selection_rule']}

Weak classes: {', '.join(WEAK_CLASSES)}. `weak_class_score` is the mean validation F1 over those classes.

## Baseline

v4-baseline (`layer 7 + mean_std + balanced RBF-SVM + no PCA + no bias`): {baseline_text}

## Selected frozen validation config

| Pick | Layers | PCA | Class weight | SVM | Bias | Val Accuracy | Val Macro-F1 | weak_class_score | run_id |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- |
{selected_line}

## Top 10 by validation Macro-F1

| Rank | Layers | PCA | Class weight | SVM | Bias | Val Accuracy | Val Macro-F1 | weak_class_score | run_id |
| ---: | --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- |
{chr(10).join(rows)}

## Frozen plan

Frozen plan written to `{args.frozen_plan}`. It explicitly records that no test-set metrics were used and that final v4 test evaluation is still **not evaluated**.
'''
    args.output_summary_md.write_text(md, encoding='utf8')
    print('SELECTED', json.dumps({'run_id': selected['run_id'], 'layers': selected['wavlm_layers'], 'pca': selected['pca_components'], 'class_weight': selected['class_weight_name'], 'C': selected['C'], 'gamma': selected['gamma'], 'bias': selected['bias_search'], 'val_accuracy': selected['val_accuracy'], 'val_macro_f1': selected['val_macro_f1'], 'weak_class_score': selected['weak_class_score']}, ensure_ascii=False))

if __name__ == '__main__':
    main()
