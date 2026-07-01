#!/usr/bin/env python
"""Command-line wrapper for TopoMamba-3D Synapse training."""

from __future__ import annotations

from tools.synapse3d_pipeline import build_train_parser, run_synapse3d_training


def main() -> None:
    args = build_train_parser().parse_args()
    run_synapse3d_training(args)


if __name__ == "__main__":
    main()
