# run_pipeline.py
import argparse
import yaml
from pathlib import Path

from src.data.preprocess_amazon import preprocess_amazon_reviews
from src.train.train_tmn import run_train_tmn
from src.train.train_tcf import run_train_tcf
from src.train.train_tdar import run_train_tdar
from src.train.evaluate import run_evaluate
from src.utils.seeds import set_global_seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--stages", type=str, default="all",
                        help="all | preprocess | tmn | tcf | tdar | eval | preprocess,tmn,...")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    stages = args.stages
    if stages == "all":
        stages = ["preprocess", "tmn", "tcf", "tdar", "eval"]
    else:
        stages = [s.strip() for s in stages.split(",")]

    set_global_seed(cfg.get("seed", 42))

    # ---------- preprocess ----------
    if "preprocess" in stages:
        for dom in [cfg["source_domain"], cfg["target_domain"]]:
            in_path = cfg["raw_paths"][dom]
            out_dir = Path(cfg["processed_root"]) / dom
            out_dir.mkdir(parents=True, exist_ok=True)

            preprocess_amazon_reviews(
                in_path=in_path,
                out_dir=str(out_dir),
                k_core=cfg["preprocess"]["k_core"],
                seed=cfg["seed"],
                vocab_size=cfg["preprocess"]["vocab_size"],
                min_freq=cfg["preprocess"]["min_freq"],
            )

    # ---------- train TMN ----------
    if "tmn" in stages:
        for dom in [cfg["source_domain"], cfg["target_domain"]]:
            run_train_tmn(cfg, domain=dom)

    # ---------- train TCF ----------
    if "tcf" in stages:
        for dom in [cfg["source_domain"], cfg["target_domain"]]:
            run_train_tcf(cfg, domain=dom)

    # ---------- train TDAR ----------
    if "tdar" in stages:
        run_train_tdar(cfg)

    # ---------- eval ----------
    if "eval" in stages:
        run_evaluate(cfg)

if __name__ == "__main__":
    main()
