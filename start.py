"""First-time setup and server launcher. Run: python start.py"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent
ENVIRONMENT = ROOT / ".venv"
PYTHON = ENVIRONMENT / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
REQUIREMENTS = ROOT / "requirements-neural.txt"
STAMP = ENVIRONMENT / ".mailcheck-dependencies"
MODEL_FILES = (
    "src/naive_bayes/artifacts/naive_bayes_pipeline.joblib",
    "src/logistic_regression/artifacts/logistic_regression_pipeline.joblib",
    "src/neural_network/nn_artifacts/model/spam_nn.keras",
    "src/neural_network/nn_artifacts/model/tfidf_vectorizer.joblib",
    "src/neural_network/nn_artifacts/results.json",
)


def run(*arguments):
    subprocess.run([str(PYTHON), *arguments], cwd=ROOT, check=True)


def prepare():
    missing = [name for name in MODEL_FILES if not (ROOT / name).is_file()]
    if missing:
        raise RuntimeError(
            "Copy the supplied saved model files into this project first:\n  "
            + "\n  ".join(missing)
            + "\nThe launcher installs dependencies; it does not download or train models."
        )

    if not PYTHON.is_file():
        print("Creating the Python environment...", flush=True)
        venv.EnvBuilder(with_pip=True).create(ENVIRONMENT)

    # Validate installed versions against both runtime requirement files.
    probe = """
from importlib.metadata import version
from pathlib import Path
from pip._vendor.packaging.requirements import Requirement
def check(path):
    for line in path.read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        if line.startswith('-r '):
            check(path.parent / line[3:].strip())
            continue
        requirement = Requirement(line)
        if requirement.marker is None or requirement.marker.evaluate():
            assert version(requirement.name) in requirement.specifier, line
check(Path('requirements-neural.txt'))
import numpy, scipy, pandas, sklearn, joblib, tensorflow
"""
    checked = subprocess.run(
        [str(PYTHON), "-c", probe], cwd=ROOT, capture_output=True, text=True
    )
    fingerprint = hashlib.sha256(
        (ROOT / "requirements.txt").read_bytes() + REQUIREMENTS.read_bytes()
    ).hexdigest()
    changed = STAMP.exists() and STAMP.read_text() != fingerprint
    if checked.returncode or changed:
        print("Installing model dependencies. First-time setup may take a few minutes...", flush=True)
        run("-m", "pip", "install", "-r", str(REQUIREMENTS))
        subprocess.run([str(PYTHON), "-c", probe], cwd=ROOT, check=True)
    STAMP.write_text(fingerprint)
    print("Environment and saved model files are ready.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--check", action="store_true", help="Verify setup and model loading, then exit.")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535.")
    prepare()
    if args.check:
        code = """
import json, sys
sys.path.insert(0, 'app')
from model_loader import Models
models = Models()
print(json.dumps(models.status(), indent=2))
sys.exit(1 if models.errors else 0)
"""
        run("-c", code)
        return
    with socket.socket() as listener:
        try:
            listener.bind(("127.0.0.1", args.port))
        except OSError as exc:
            raise RuntimeError(
                f"Port {args.port} is unavailable. If mailCheck is already running, open "
                f"http://127.0.0.1:{args.port}. Otherwise try python start.py --port {args.port + 1}."
            ) from exc
    print(f"Starting mailCheck at http://127.0.0.1:{args.port}", flush=True)
    print("Keep this window open. Press Ctrl+C to stop.\n", flush=True)
    run(str(ROOT / "app/app.py"), "--port", str(args.port))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"\nSetup/start failed: {exc}", file=sys.stderr)
        sys.exit(1)
