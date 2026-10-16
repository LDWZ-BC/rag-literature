"""CLI entry point for RAG literature pipeline."""

import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(description="RAG Literature Pipeline")
    parser.add_argument("--no-rag", action="store_true", help="Skip RAG update step")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from .pipeline.orchestrator import process_all

    process_all(update_rag=not args.no_rag)


if __name__ == "__main__":
    main()
