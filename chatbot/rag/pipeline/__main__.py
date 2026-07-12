"""Entry point: `python -m chatbot.rag.pipeline` runs the stdin/stdout demo."""

import os
print(os.getcwd())

import sys
print(sys.path)
from chatbot.rag.pipeline.pipeline import _main

if __name__ == "__main__":
    _main()
