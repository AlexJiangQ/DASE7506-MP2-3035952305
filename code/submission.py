"""Submission entry point for the frozen Stage 3 CREO selector.

From code/, use with --checkpoint ../submission_assets/checkpoint.
The implementation is kept in submission_stage3.py, whose dependencies match
the source hashes recorded in the Stage 3 freeze manifest.
"""

from submission_stage3 import predict

__all__ = ["predict"]
