from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from emotion_app.domain import EMOTIONS, RecognitionResult
from emotion_app.recognizers.image import ImageRecognizer
from emotion_app.recognizers.speech import SpeechRecognizer
from emotion_app.recognizers.text import TextRecognizer
from scripts.fusion._fusion_common import empty_probs, read_csv_rows, write_jsonl


def result_payload(result: RecognitionResult) -> tuple[bool, dict[str, float], str | None, str | None, float]:
    probs = empty_probs()
    if result.probabilities:
        for label in EMOTIONS:
            probs[label] = float(result.probabilities.get(label, 0.0))
    return result.ok, probs, result.emotion, result.error, float(result.confidence)


def load_existing(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                done.add(str(json.loads(line)["sample_id"]))
            except Exception:
                continue
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache text/speech/image probabilities for multimodal fusion.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, default=None)
    parser.add_argument("--speech-model", type=Path, default=None)
    parser.add_argument("--image-model", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rows = read_csv_rows(args.input_csv)
    if args.limit > 0:
        rows = rows[: args.limit]
    done = load_existing(args.output_jsonl) if args.resume else set()

    text_recognizer = TextRecognizer(args.text_model) if args.text_model else TextRecognizer()
    speech_recognizer = SpeechRecognizer(args.speech_model) if args.speech_model else SpeechRecognizer()
    image_recognizer = ImageRecognizer(args.image_model) if args.image_model else ImageRecognizer()

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    count = 0
    with args.output_jsonl.open(mode, encoding="utf-8", newline="\n") as handle:
        for row in rows:
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id or sample_id in done:
                continue
            errors: dict[str, str] = {}

            text_result = text_recognizer.predict(row.get("text", ""))
            text_ok, text_probs, text_pred, text_error, text_conf = result_payload(text_result)
            if text_error:
                errors["text"] = text_error

            audio_path = str(row.get("audio_path", "")).strip()
            speech_result = speech_recognizer.predict(audio_path) if audio_path else RecognitionResult.failure("missing audio_path", "speech")
            speech_ok, speech_probs, speech_pred, speech_error, speech_conf = result_payload(speech_result)
            if speech_error:
                errors["speech"] = speech_error

            image_path = str(row.get("image_path", "")).strip()
            image_result = image_recognizer.predict(image_path) if image_path else RecognitionResult.failure("missing image_path", "image")
            image_ok, image_probs, image_pred, image_error, image_conf = result_payload(image_result)
            if image_error:
                errors["image"] = image_error

            payload = {
                "sample_id": sample_id,
                "label": row.get("label", ""),
                "text_probs": text_probs,
                "speech_probs": speech_probs,
                "image_probs": image_probs,
                "text_ok": text_ok,
                "speech_ok": speech_ok,
                "image_ok": image_ok,
                "text_pred": text_pred,
                "speech_pred": speech_pred,
                "image_pred": image_pred,
                "text_confidence": text_conf,
                "speech_confidence": speech_conf,
                "image_confidence": image_conf,
                "errors": errors,
            }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
            if count % 50 == 0:
                print(json.dumps({"cached": count, "last_sample_id": sample_id}, ensure_ascii=False), flush=True)

    print(json.dumps({"cached_new": count, "output": str(args.output_jsonl)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
