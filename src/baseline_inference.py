"""Compatibility entry point for the explicit free-form inference prototype."""

import logging

from vlm_finetuning.prototype_inference import main

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
