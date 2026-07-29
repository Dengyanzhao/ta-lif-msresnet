#!/usr/bin/env python3
"""Validate dependencies, frozen choices, analysis margins, and data paths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SRC_PATH = str(SRC_ROOT)
if SRC_PATH in sys.path:
    sys.path.remove(SRC_PATH)
sys.path.insert(0, SRC_PATH)

from talif_msresnet.preflight import check_protocol  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(PROJECT_ROOT / "configs" / "protocol.yaml"))
    parser.add_argument(
        "--mode",
        choices=("smoke", "pilot", "full", "final-test"),
        default="full",
    )
    parser.add_argument("--json", action="store_true", help="Print a machine-readable report")
    parser.add_argument(
        "--skip-dependency-check",
        action="store_true",
        help="Validate only protocol fields and paths",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = check_protocol(
            args.protocol,
            mode=args.mode,
            project_root=PROJECT_ROOT,
            check_dependencies=not args.skip_dependency_check,
        )
    except Exception as exc:
        print(f"ERROR: preflight could not load the protocol: {type(exc).__name__}: {exc}")
        return 2
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))
    else:
        print(f"Preflight mode={report.mode} protocol_hash={report.protocol_hash}")
        for warning in report.warnings:
            print(f"WARNING: {warning}")
        for error in report.errors:
            print(f"ERROR: {error}")
        print("PASS" if report.ok else f"BLOCKED ({len(report.errors)} error(s))")
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
