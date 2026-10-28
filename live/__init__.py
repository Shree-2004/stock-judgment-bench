"""The live pipeline: forward-only, append-only, separate from the frozen v1
benchmark. Nothing under `live/` may change what `scripts.report` prints —
`tests/test_frozen_v1.py` enforces that.
"""
