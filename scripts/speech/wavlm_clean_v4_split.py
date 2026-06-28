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
    p = argparse.ArgumentParser(description='Create sealed WavLM clean v4 speaker-independent split')
    p.add_argument('--split-file', type=Path, default=PROJECT_ROOT / 'models/speech/wavlm_clean_split_v4.json')
    p.add_argument('--crema-root', type=Path, default=PROJECT_ROOT / 'datasets/CREMA-D')
    p.add_argument('--emodb-root', type=Path, default=PROJECT_ROOT / 'datasets/EmoDB')
    p.add_argument('--seed', type=int, default=2027)
    p.add_argument('--crema-val-speakers', type=int, default=15)
    p.add_argument('--crema-test-speakers', type=int, default=15)
    p.add_argument('--emodb-val-speakers', type=int, default=2)
    p.add_argument('--emodb-test-speakers', type=int, default=2)
    return p.parse_args()

def split_dataset(speakers: list[str], val_n: int, test_n: int, rng: random.Random):
    speakers = sorted(speakers)
    if len(speakers) < val_n + test_n + 1:
        raise RuntimeError(f'Not enough speakers: {len(speakers)} for val={val_n}, test={test_n}')
    shuffled = speakers[:]
    rng.shuffle(shuffled)
    test = sorted(shuffled[:test_n])
    val = sorted(shuffled[test_n:test_n + val_n])
    train = sorted(shuffled[test_n + val_n:])
    return train, val, test

def main():
    args = parse_args()
    if args.split_file.exists():
        raise SystemExit(f'Refusing to overwrite existing split: {args.split_file}')
    samples = discover_crema(args.crema_root) + discover_emodb(args.emodb_root)
    rng = random.Random(args.seed)
    speakers_by_dataset = {
        'CREMA-D': sorted({s.speaker for s in samples if s.dataset == 'CREMA-D'}),
        'EmoDB': sorted({s.speaker for s in samples if s.dataset == 'EmoDB'}),
    }
    crema_train, crema_val, crema_test = split_dataset(
        speakers_by_dataset['CREMA-D'], args.crema_val_speakers, args.crema_test_speakers, rng
    )
    emodb_train, emodb_val, emodb_test = split_dataset(
        speakers_by_dataset['EmoDB'], args.emodb_val_speakers, args.emodb_test_speakers, rng
    )
    parts = {
        'train': sorted(crema_train + emodb_train),
        'val': sorted(crema_val + emodb_val),
        'test': sorted(crema_test + emodb_test),
    }
    sets = {k: set(v) for k, v in parts.items()}
    if sets['train'] & sets['val'] or sets['train'] & sets['test'] or sets['val'] & sets['test']:
        raise RuntimeError('Speaker partitions are not disjoint')

    def metadata(key: str):
        rows = [s for s in samples if s.speaker in sets[key]]
        dist = Counter(s.label for s in rows)
        if set(dist) != set(EMOTION_LABELS):
            raise RuntimeError(f'{key} does not cover all six emotions: {dist}')
        by_dataset = Counter(s.dataset for s in rows)
        return len(rows), {label: int(dist[label]) for label in EMOTION_LABELS}, dict(by_dataset)

    train_n, train_d, train_by_dataset = metadata('train')
    val_n, val_d, val_by_dataset = metadata('val')
    test_n, test_d, test_by_dataset = metadata('test')
    doc = {
        'protocol_name': 'wavlm_clean_v4',
        'dataset_names': ['CREMA-D', 'EmoDB'],
        'excluded_datasets': {
            'TESS': 'Only two speakers; excluded to preserve mutually exclusive speaker-independent train/validation/test partitions.'
        },
        'emotion_labels': EMOTION_LABELS,
        'random_seed': args.seed,
        'split_strategy': 'Speaker-independent split by dataset: CREMA-D 15 validation speakers and 15 sealed test speakers; EmoDB 2 validation speakers and 2 sealed test speakers; all remaining speakers train.',
        'train_speakers': parts['train'],
        'val_speakers': parts['val'],
        'test_speakers': parts['test'],
        'speaker_counts': {k: len(v) for k, v in parts.items()},
        'sample_counts': {'train': train_n, 'validation': val_n, 'test': test_n},
        'sample_counts_by_dataset': {
            'train': train_by_dataset,
            'validation': val_by_dataset,
            'test': test_by_dataset,
        },
        'per_split_class_distribution': {'train': train_d, 'validation': val_d, 'test': test_d},
        'test_status': 'sealed',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'statement': 'Train, validation, and test speakers are mutually exclusive. v4 test labels are sealed for future one-time evaluation and must not be used for model selection.',
        'prohibited_during_validation': ['Do not evaluate v4 test set', 'Do not use test-set metrics', 'Do not modify v2/v3 artifacts', 'Do not repackage exe'],
    }
    args.split_file.parent.mkdir(parents=True, exist_ok=True)
    args.split_file.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({
        'split_file': str(args.split_file),
        'protocol_name': doc['protocol_name'],
        'speaker_counts': doc['speaker_counts'],
        'sample_counts': doc['sample_counts'],
        'test_status': doc['test_status'],
    }, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
