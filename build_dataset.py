from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import tempfile
from collections import Counter
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "validation", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_source(source_dir: Path) -> tuple[list[dict], dict]:
    metadata = json.loads((source_dir / "source_manifest.json").read_text(encoding="utf-8"))
    path = source_dir / "wordpress_records.jsonl"
    if sha256(path) != metadata["source_sha256"]:
        raise ValueError("Контрольная сумма исходных записей не совпадает с source_manifest.json")
    with path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if len(records) != metadata["exported_records"]:
        raise ValueError("Изменилось число исходных записей")
    if any(not record["code"].strip() for record in records):
        raise ValueError("В исходных записях есть пустые фрагменты кода")
    if len({record["record_id"] for record in records}) != len(records):
        raise ValueError("В исходных записях повторяется record_id")
    return records, metadata


def make_splits(records: list[dict]) -> dict[str, list[dict]]:
    partitions = {"train": [], "test": []}
    for record in records:
        partition = record["benchmark_partition"]
        if partition not in partitions:
            raise ValueError(f"Неизвестная исходная выборка: {partition}")
        partitions[partition].append(record)
    training_pool = partitions["train"]
    groups = [record["plugin_slug"] for record in training_pool]
    if not all(groups):
        raise ValueError("Для группового разбиения нужен plugin_slug у каждой записи")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
    _, validation_positions = next(splitter.split(training_pool, groups=groups))
    validation_positions = set(validation_positions)
    splits = {
        "train": [record for i, record in enumerate(training_pool) if i not in validation_positions],
        "validation": [record for i, record in enumerate(training_pool) if i in validation_positions],
        "test": partitions["test"],
    }
    for name, items in splits.items():
        if not items:
            raise ValueError(f"Пустая выборка: {name}")
    for left, right in itertools.combinations(SPLITS, 2):
        for field in ("record_id", "plugin_slug", "code_hash", "clone_family_id"):
            left_values = {record[field] for record in splits[left] if record[field]}
            right_values = {record[field] for record in splits[right] if record[field]}
            if left_values & right_values:
                raise ValueError(f"Пересечение {field} между {left} и {right}")
    return splits


def write_csv(path: Path, records: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def build_summary(splits: dict[str, list[dict]], metadata: dict) -> dict:
    records = list(itertools.chain.from_iterable(splits.values()))
    summary = {
        "dataset": "WordPress CVE Benchmark v2 Variant B",
        "selected_fold": "0",
        "records_loaded": metadata["benchmark_records_loaded"],
        "records_with_code": len(records),
        "records_final": len(records),
        **{name: len(splits[name]) for name in SPLITS},
        "plugins": len({record["plugin_slug"] for record in records if record["plugin_slug"]}),
        "class_distribution": {},
        "plugin_overlap": {"train_validation": 0, "train_test": 0, "validation_test": 0},
    }
    for name, items in splits.items():
        counts = Counter(int(record["label_binary"]) for record in items)
        summary["class_distribution"][name] = {"vulnerable": counts[1], "safe": counts[0]}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "source")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    records, metadata = read_source(args.source_dir.resolve())
    splits = make_splits(records)
    for name, items in splits.items():
        for record in items:
            record["split"] = name

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dataset-", dir=output_dir) as temporary:
        staging = Path(temporary)
        for name, items in splits.items():
            write_csv(staging / f"wordpress_xai_{name}.csv", items, metadata["columns"])
        combined = list(itertools.chain.from_iterable(splits.values()))
        write_csv(staging / "wordpress_xai_dataset.csv", combined, metadata["columns"])
        with (staging / "dataset_summary.json").open("w", encoding="utf-8") as stream:
            json.dump(build_summary(splits, metadata), stream, ensure_ascii=False, indent=2)
        for filename, expected in metadata["expected_files_sha256"].items():
            actual = sha256(staging / filename)
            if actual != expected:
                raise ValueError(f"{filename}: результат не совпал с исходным файлом; запись отменена")
        for filename in metadata["expected_files_sha256"]:
            (staging / filename).replace(output_dir / filename)

    for name, items in splits.items():
        counts = Counter(int(record["label_binary"]) for record in items)
        print(f"{name:10}: {len(items):5} записей; уязвимых {counts[1]}, безопасных {counts[0]}")



if __name__ == "__main__":
    main()
