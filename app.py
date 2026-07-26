"""Hugging Face Spaces entry point for the Fashion RAG Assistant."""

from fashion_rag.config import Settings
from fashion_rag.ui import build_application


demo = build_application(Settings.from_env())


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=8).launch()
