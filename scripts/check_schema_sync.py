"""Generate schemas intentionally or verify committed schemas byte-for-byte."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_inference_systems.canonical import canonical_json
from llm_inference_systems.schema_io import SCHEMA_MODELS, schema_sync_mismatches, write_schemas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write generated schemas")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = root / "schemas"
    if args.write:
        write_schemas(directory)
        print(canonical_json({"schema_count": len(SCHEMA_MODELS), "status": "written"}))
        return 0
    mismatches = schema_sync_mismatches(directory)
    if mismatches:
        print(canonical_json({"mismatches": list(mismatches), "status": "out_of_sync"}))
        return 1
    print(canonical_json({"schema_count": len(SCHEMA_MODELS), "status": "synchronized"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
