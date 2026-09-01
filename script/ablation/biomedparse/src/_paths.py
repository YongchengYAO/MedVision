"""Path resolution shared by every script in src/.

Nothing here depends on the working directory: all locations derive from this
file's position inside the MedVision repository, with environment overrides.
"""

import os
import sys

# <repo>/script/ablation/biomedparse
ABLATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# <repo>
REPO_ROOT = os.path.abspath(os.path.join(ABLATION_DIR, "..", "..", ".."))
# Upstream microsoft/BiomedParse checkout (populated by setup.sh)
BIOMEDPARSE_DIR = os.environ.get(
    "BIOMEDPARSE_DIR", os.path.join(ABLATION_DIR, "third_party", "BiomedParse")
)


def add_medvision_to_path():
    """Make `medvision_bm` (repo source) and `medvision_ds` (installed by
    install_medvision_ds into <repo>/Data/src) importable, source first."""
    for path in (os.path.join(REPO_ROOT, "Data", "src"), os.path.join(REPO_ROOT, "src")):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def add_biomedparse_to_path():
    """Make the upstream modules (`utils`, `inference`, `src.*`) importable."""
    if not os.path.isdir(BIOMEDPARSE_DIR):
        sys.exit(
            f"Upstream BiomedParse not found at {BIOMEDPARSE_DIR}. "
            "Run setup.sh, or point BIOMEDPARSE_DIR at an existing checkout."
        )
    if BIOMEDPARSE_DIR not in sys.path:
        sys.path.insert(0, BIOMEDPARSE_DIR)
