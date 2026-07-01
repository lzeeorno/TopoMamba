#!/usr/bin/env python
"""Command-line wrapper for TopoMamba-3D Synapse testing."""

from __future__ import annotations

from tools.synapse3d_pipeline import build_test_parser, run_synapse3d_test


def main() -> None:
    args = build_test_parser().parse_args()
    run_synapse3d_test(args)


if __name__ == "__main__":
    main()
