# run_all_pipeline.py

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


DEFAULT_SOURCES = ["Digital_Music", "Movies_and_TV", "Video_Games"]
DEFAULT_TARGET = "Electronics"


def _split_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _find_config_for_pair(config_dir: Path, src: str, tgt: str) -> Path:
    """
    Try multiple common naming conventions.
    You can freely rename your configs; just pass --configs explicitly if needed.
    """
    candidates = [
        config_dir / f"{src}_to_{tgt}.yaml",
        config_dir / f"{src}_to_{tgt}.yml",
        config_dir / f"{src.lower()}_to_{tgt.lower()}.yaml",
        config_dir / f"{src.lower()}_to_{tgt.lower()}.yml",
    ]

    # common abbreviations people use
    abbr = {
        "Digital_Music": "dm",
        "Movies_and_TV": "mv",
        "Video_Games": "vg",
        "Electronics": "elec",
    }
    src_ab = abbr.get(src, src.lower())
    tgt_ab = abbr.get(tgt, tgt.lower())
    candidates += [
        config_dir / f"{src_ab}_to_{tgt_ab}.yaml",
        config_dir / f"{src_ab}_to_{tgt_ab}.yml",
    ]

    for p in candidates:
        if p.exists():
            return p

    # If still not found, show what we tried
    tried = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Config for pair {src}->{tgt} not found under {config_dir}.\nTried:\n{tried}\n"
        f"Fix: pass --configs explicitly, e.g.\n"
        f"  --configs configs/dm_to_elec.yaml,configs/mv_to_elec.yaml,configs/vg_to_elec.yaml"
    )


def run_one(
    run_pipeline_path: Path,
    config_path: Path,
    stages: str,
    extra_args: Optional[List[str]] = None,
    python_exec: str = sys.executable,
) -> int:
    """
    Call: python run_pipeline.py --config <config> --stages <stages>
    """
    cmd = [
        python_exec,
        str(run_pipeline_path),
        "--config",
        str(config_path),
        "--stages",
        stages,
    ]
    if extra_args:
        cmd += extra_args

    print(f"\n=== RUN: {config_path.name} | stages={stages} ===")
    print("CMD:", " ".join(cmd))

    proc = subprocess.run(cmd)
    return int(proc.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_pipeline",
        type=str,
        default="run_pipeline.py",
        help="Path to your single-entry pipeline script.",
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default="configs",
        help="Directory containing YAML configs.",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default="",
        help="Comma-separated explicit config paths (overrides auto pair->config search).",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default=",".join(DEFAULT_SOURCES),
        help="Comma-separated source domains.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=DEFAULT_TARGET,
        help="Target domain.",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="preprocess,tmn,tcf,tdar,evaluate",
        help="Stages to run, passed to run_pipeline.py.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue running remaining experiments even if one fails.",
    )
    # passthrough for any extra args you want to send to run_pipeline.py
    parser.add_argument(
        "--extra",
        type=str,
        default="",
        help="Extra args passed to run_pipeline.py, e.g. '--extra \"--some_flag 1\"'",
    )
    args = parser.parse_args()

    run_pipeline_path = Path(args.run_pipeline)
    if not run_pipeline_path.exists():
        raise FileNotFoundError(f"run_pipeline.py not found at: {run_pipeline_path.resolve()}")

    config_dir = Path(args.config_dir)
    if not config_dir.exists() and not args.configs:
        raise FileNotFoundError(f"config_dir not found: {config_dir.resolve()}")

    stages = args.stages.strip()
    extra_args = args.extra.strip().split() if args.extra.strip() else []

    # Determine configs to run
    configs: List[Path] = []
    if args.configs.strip():
        for p in _split_csv(args.configs):
            cp = Path(p)
            if not cp.exists():
                raise FileNotFoundError(f"Config not found: {cp.resolve()}")
            configs.append(cp)
    else:
        sources = _split_csv(args.sources)
        tgt = args.target.strip()
        for src in sources:
            configs.append(_find_config_for_pair(config_dir, src, tgt))

    # Run sequentially
    failed = False
    for idx, cfg_path in enumerate(configs, start=1):
        print(f"\n##### EXP {idx}/{len(configs)}: {cfg_path} #####")
        rc = run_one(
            run_pipeline_path=run_pipeline_path,
            config_path=cfg_path,
            stages=stages,
            extra_args=extra_args,
        )
        if rc != 0:
            print(f"[ERROR] pipeline failed for config={cfg_path} (returncode={rc})")
            failed = True
            if not args.continue_on_error:
                break

    if failed:
        sys.exit(1)
    print("\nALL EXPERIMENTS FINISHED OK.")
    sys.exit(0)


if __name__ == "__main__":
    main()