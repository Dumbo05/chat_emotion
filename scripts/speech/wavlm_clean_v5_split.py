from __future__ import annotations
import argparse, json, random, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.train_speech_model import discover_crema, discover_emodb

EMOTION_LABELS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'neutral']

def parse_args():
    p = argparse.ArgumentParser(description='Create sealed WavLM clean v5 speaker-independent split')
    p.add_argument('--split-file', type=Path, default=PROJECT_ROOT / 'models/speech/wavlm_clean_split_v5.json')
    p.add_argument('--crema-root', type=Path, default=PROJECT_ROOT / 'datasets/CREMA-D')
    p.add_argument('--emodb-root', type=Path, default=PROJECT_ROOT / 'datasets/EmoDB')
    p.add_argument('--seed', type=int, default=2028)
    p.add_argument('--crema-val-speakers', type=int, default=15)
    p.add_argument('--crema-test-speakers', type=int, default=15)
    p.add_argument('--emodb-val-speakers', type=int, default=2)
    p.add_argument('--emodb-test-speakers', type=int, default=2)
    return p.parse_args()

def class_distribution(samples, speakers):
    rows = [s for s in samples if s.speaker in speakers]
    dist = Counter(s.label for s in rows)
    missing = sorted(set(EMOTION_LABELS) - set(dist))
    return rows, {label: int(dist.get(label, 0)) for label in EMOTION_LABELS}, missing

def score_candidate(samples, parts):
    # Prefer all classes present and roughly balanced validation/test class counts.
    penalty = 0.0
    for key in ('val', 'test'):
        _, dist, missing = class_distribution(samples, set(parts[key]))
        penalty += 1000.0 * len(missing)
        vals = [dist[x] for x in EMOTION_LABELS]
        penalty += float(max(vals) - min(vals))
    return penalty

def main():
    args = parse_args()
    if args.split_file.exists():
        raise SystemExit(f'Refusing to overwrite existing split: {args.split_file}')
    samples = discover_crema(args.crema_root) + discover_emodb(args.emodb_root)
    by_dataset = {
        'CREMA-D': sorted({s.speaker for s in samples if s.dataset == 'CREMA-D'}),
        'EmoDB': sorted({s.speaker for s in samples if s.dataset == 'EmoDB'}),
    }
    rng = random.Random(args.seed)
    best = None
    best_score = float('inf')
    for _ in range(2000):
        crema = by_dataset['CREMA-D'][:]
        emodb = by_dataset['EmoDB'][:]
        rng.shuffle(crema)
        rng.shuffle(emodb)
        parts = {
            'test': sorted(crema[:args.crema_test_speakers] + emodb[:args.emodb_test_speakers]),
            'val': sorted(crema[args.crema_test_speakers:args.crema_test_speakers + args.crema_val_speakers] + emodb[args.emodb_test_speakers:args.emodb_test_speakers + args.emodb_val_speakers]),
            'train': sorted(crema[args.crema_test_speakers + args.crema_val_speakers:] + emodb[args.emodb_test_speakers + args.emodb_val_speakers:]),
        }
        sets = {k: set(v) for k, v in parts.items()}
        if sets['train'] & sets['val'] or sets['train'] & sets['test'] or sets['val'] & sets['test']:
            continue
        score = score_candidate(samples, parts)
        if score < best_score:
            best_score = score
            best = parts
            if score == 0:
                break
    if best is None:
        raise RuntimeError('Failed to create a disjoint v5 split')
    sets = {k: set(v) for k, v in best.items()}
    metadata = {}
    for key in ('train', 'val', 'test'):
        rows, dist, missing = class_distribution(samples, sets[key])
        if missing:
            raise RuntimeError(f'{key} split misses classes: {missing}')
        metadata[key] = {'sample_count': len(rows), 'class_distribution': dist, 'dataset_distribution': dict(Counter(s.dataset for s in rows))}
    doc = {
        'protocol_name': 'wavlm_clean_v5',
        'dataset_names': ['CREMA-D', 'EmoDB'],
        'excluded_datasets': {'TESS': 'Only two speakers; cannot create mutually exclusive speaker-independent train/validation/test partitions.'},
        'emotion_labels': EMOTION_LABELS,
        'random_seed': args.seed,
        'split_strategy': 'Speaker-independent split by dataset with repeated seeded search for validation/test class coverage and balance.',
        'train_speakers': best['train'],
        'val_speakers': best['val'],
        'test_speakers': best['test'],
        'train_sample_count': metadata['train']['sample_count'],
        'val_sample_count': metadata['val']['sample_count'],
        'test_sample_count': metadata['test']['sample_count'],
        'per_split_class_distribution': {k: metadata[k]['class_distribution'] for k in ('train', 'val', 'test')},
        'sample_counts_by_dataset': {k: metadata[k]['dataset_distribution'] for k in ('train', 'val', 'test')},
        'speaker_counts': {k: len(best[k]) for k in ('train', 'val', 'test')},
        'test_status': 'sealed',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'statement': 'Train, validation, and test speakers are mutually exclusive. v5 test set remains sealed and must not be used during validation sweep or model selection.',
    }
    args.split_file.parent.mkdir(parents=True, exist_ok=True)
    args.split_file.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'split_file': str(args.split_file), 'speaker_counts': doc['speaker_counts'], 'sample_counts': {'train': doc['train_sample_count'], 'val': doc['val_sample_count'], 'test': doc['test_sample_count']}, 'test_status': doc['test_status']}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
