"""Anchor benchmark exports.

Everything in this package produces artifacts under data/anchor/ and
results/anchor/ ONLY. Anchor artifacts follow the official per-user
leave-one-out protocol, which ignores the global temporal cutoffs used by the
production pipelines — anchor training rows fall inside the production locked
test window. Nothing produced here may enter a production candidate / ranker /
calibration pipeline. See the isolation declaration in each manifest.
"""
