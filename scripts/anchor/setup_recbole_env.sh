#!/bin/bash
# Reproducible setup for the anchor RecBole venv .
#
#   bash scripts/anchor/setup_recbole_env.sh
#
# Creates ${ANCHOR_VENV:-$HOME/envs/recbole-anchor} -- a GPFS venv visible from
# compute nodes (a /tmp venv on the login node is NOT; that was the
# lesson that motivated this location). Run on the LOGIN node: compute nodes
# have no internet, so all pip downloads must happen here.
#
# Why not a plain `pip install recbole==1.2.1 "ray>=2.30"`:
#   recbole 1.2.1 pins ray<=2.6.3,>=1.13.0, which conflicts with the
#   spec-mandated ray>=2.30 -- pip's resolver refuses the pair outright
#   (verified 2026-08-11: "ResolutionImpossible ... recbole 1.2.1 depends on
#   ray<=2.6.3"). The deliberate deviation: install recbole --no-deps, install
#   ray>=2.30 alongside recbole's other declared deps explicitly. `pip check`
#   afterwards reports exactly one expected complaint (the ray pin) and
#   nothing else.
#
# Shims (installed as sitecustomize.py, canonical copy in
# scripts/anchor/recbole_anchor_sitecustomize.py):
#   * numpy scalar aliases (np.float / np.object / ...) removed in numpy>=1.24
#     that recbole 1.2.1 still touches -- re-injected post-import.
#   * torch.load defaults to weights_only=False (torch>=2.6 flipped the
#     default, which breaks RecBole checkpoint load).
#
# Versions captured at first successful build (2026-08-11):
#   python 3.10.18 (seeded from the tae conda env's interpreter)
#   recbole 1.2.1, ray 2.57.0, torch 2.13.0+cu130, numpy 2.2.6,
#   pandas 2.3.3, scipy 1.15.3, scikit-learn 1.7.2
#
# Verified post-install (see P1.1 report):
#   * import recbole/ray OK; torch.load shim demonstrably refuses/allows an
#     arbitrary pickled object with/without explicit weights_only=True.
#   * RecBole `order: TO` sorts with np.argsort(kind="stable")
#     (recbole/data/interaction.py:347, dataset.py:1829) -- required for the
#     pre-sorted anchor .inter files to reproduce our deterministic tie-break.

set -euo pipefail

VENV=${ANCHOR_VENV:-$HOME/envs/recbole-anchor}
REPO=${REPO_ROOT:-$(pwd)}

source ~/.bashrc
conda activate tae   # only to borrow its python 3.10 interpreter for venv creation

mkdir -p "$(dirname "$VENV")"
python -m venv "$VENV"

"$VENV/bin/pip" install --no-cache-dir --upgrade pip
"$VENV/bin/pip" install --no-cache-dir --no-deps "recbole==1.2.1"
# recbole 1.2.1's declared deps (from its wheel metadata), minus the ray pin,
# plus the spec-mandated ray>=2.30:
"$VENV/bin/pip" install --no-cache-dir \
    "ray>=2.30" \
    "torch>=1.10.0" "numpy>=1.17.2" "scipy>=1.6.0" "pandas>=1.3.0" \
    "tqdm>=4.48.2" "colorlog==4.7.2" "colorama==0.4.4" \
    "scikit-learn>=0.23.2" "pyyaml>=5.1.0" "tensorboard>=2.5.0" \
    "thop>=0.1.1.post2207130030" "tabulate>=0.8.10" "plotly>=4.0.0" \
    "texttable>=0.9.0" "psutil>=5.9.0"

SP="$("$VENV/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")"
cp "$REPO/scripts/anchor/recbole_anchor_sitecustomize.py" "$SP/sitecustomize.py"

"$VENV/bin/python" -W error::FutureWarning -c "
import recbole, ray, numpy as np, torch
assert np.float is float and np.object is object, 'numpy alias shim failed'
assert getattr(torch.load, '_anchor_weights_only_shim', False), 'torch.load shim failed'
print('recbole', recbole.__version__, '| ray', ray.__version__,
      '| torch', torch.__version__, '| numpy', np.__version__)
print('recbole-anchor venv OK:', '$VENV')
"
