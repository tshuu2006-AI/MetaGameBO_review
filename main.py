"""Terminal entry point for the LFBO/GP held-out Forrester experiment."""
from __future__ import annotations


def main(argv=None):
    from experiments.lfbo_gp_unified.cli import main as run
    return run(argv)


if __name__ == '__main__':
    raise SystemExit(main())
