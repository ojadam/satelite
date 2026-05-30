import argparse
import subprocess
import sys


STEPS = {
    "download": ["-m", "src.download_catalog"],
    "coords": ["-m", "src.coordinates"],
    "density": ["-m", "src.density"],
    "patterns": ["-m", "src.patterns"],
    "viz": ["-m", "src.visualize"],
    "train": ["-m", "src.model_tf"],
    "analyze": ["-m", "src.analyze"],
}


def run_step(name: str) -> None:
    cmd = [sys.executable, *STEPS[name]]
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "step",
        nargs="?",
        default="all",
        choices=[*STEPS.keys(), "all"],
    )
    args = parser.parse_args()

    order = ["download", "coords", "density", "patterns", "viz", "train", "analyze"]
    if args.step == "all":
        for name in order:
            run_step(name)
    else:
        run_step(args.step)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError:
        sys.exit(1)
