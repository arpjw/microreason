"""Command-line entry point. Every command is noninteractive and scriptable (spec §9)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence

from pydantic import ValidationError

from microreason import __version__
from microreason.config import load_model_config, load_stage_config
from microreason.logging_utils import get_logger, setup_logging

log = get_logger("microreason.cli")

EXIT_OK = 0
EXIT_CONFIG_ERROR = 1
EXIT_NOT_IMPLEMENTED = 2


def _not_implemented(phase: int, what: str) -> Callable[[argparse.Namespace], int]:
    def run(_: argparse.Namespace) -> int:
        print(
            f"error: '{what}' is not implemented yet (planned for Phase {phase}).", file=sys.stderr
        )
        return EXIT_NOT_IMPLEMENTED

    return run


def _cmd_config_validate(args: argparse.Namespace) -> int:
    try:
        resolved = load_stage_config(args.config)
    except (ValidationError, ValueError, OSError) as exc:
        print(f"error: invalid config {args.config}:\n{exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    out = {
        "config_hash": resolved.config_hash(),
        "parameter_count_analytic": resolved.model.parameter_count(),
        "resolved": resolved.model_dump(mode="json"),
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return EXIT_OK


def _cmd_data_generate(args: argparse.Namespace) -> int:
    from microreason.data import build

    try:
        cfg = build.load_data_config(args.config)
    except (ValidationError, ValueError, OSError) as exc:
        print(f"error: invalid data config {args.config}:\n{exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    updates = {}
    if args.out:
        updates["out_dir"] = args.out
    if args.manifest_dir:
        updates["manifest_dir"] = args.manifest_dir
    cfg = cfg.model_copy(update=updates)
    try:
        result = build.build_dataset(cfg)
        build.guard_against_frozen(cfg, result)
    except (build.CapacityError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    for path in build.write_dataset(cfg, result):
        log.info("wrote %s", path)
    print(json.dumps(build.summarize(result), indent=2, sort_keys=True))
    return EXIT_OK


def _cmd_data_audit(args: argparse.Namespace) -> int:
    from pathlib import Path

    from microreason.data import build

    report = build.audit(args.manifest_dir, args.version)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": report["passed"], "failures": report["failures"],
                      "report": args.report}, indent=2))  # fmt: skip
    return EXIT_OK if report["passed"] else EXIT_CONFIG_ERROR


def _cmd_data_freeze(args: argparse.Namespace) -> int:
    from microreason.data import build

    try:
        path = build.freeze(args.manifest_dir, args.version)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    print(path)
    return EXIT_OK


def _cmd_data_tokenize(args: argparse.Namespace) -> int:
    from pathlib import Path

    from microreason.data.tokenized import MANIFEST_FILE, tokenize_general_text, tokenize_math
    from microreason.tokenizer import MicroTokenizer

    tok = MicroTokenizer.load(args.tokenizer)
    jobs = {
        "general_text_train_v1": lambda d: tokenize_general_text(tok, "train", d),
        "general_text_val_v1": lambda d: tokenize_general_text(tok, "val", d),
        "math_train_v1": lambda d: tokenize_math(tok, "train", d),
        "math_val_v1": lambda d: tokenize_math(tok, "val", d),
    }
    out = {}
    for name, job in jobs.items():
        d = Path(args.out) / name
        if (d / MANIFEST_FILE).exists():
            log.info("exists, skipping %s", d)
            m = json.loads((d / MANIFEST_FILE).read_text())
        else:
            log.info("tokenizing %s", name)
            m = job(d)
        out[name] = {"documents": m["documents"], "tokens": m["tokens"]}
    print(json.dumps(out, indent=2, sort_keys=True))
    return EXIT_OK


def _cmd_data_split_text(args: argparse.Namespace) -> int:
    from microreason.data.text_split import build_text_split_manifest

    try:
        m = build_text_split_manifest(args.sources, args.raw_dir, args.out)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    print(json.dumps(m, indent=2, sort_keys=True))
    return EXIT_OK


def _cmd_tokenizer_train(args: argparse.Namespace) -> int:
    from microreason import tokenizer as T

    try:
        cfg = T.load_tokenizer_config(args.config)
        texts, stats = T.collect_training_texts(cfg)
    except (ValidationError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    log.info("training tokenizer on %d texts (math fraction %.3f)", len(texts),
             stats["math_char_fraction"])  # fmt: skip
    path = T.save_tokenizer(T.train_tokenizer(cfg, iter(texts)), cfg, stats)
    print((path.parent / T.MANIFEST_FILE).read_text())
    return EXIT_OK


def _cmd_tokenizer_inspect(args: argparse.Namespace) -> int:
    from pathlib import Path

    from microreason.inspect_report import tokenizer_report

    report = tokenizer_report(args.tokenizer, include_corpus=not args.skip_corpus)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": report["passed"], "failures": report["failures"],
                      "report": args.report}, indent=2))  # fmt: skip
    return EXIT_OK if report["passed"] else EXIT_CONFIG_ERROR


def _cmd_data_fetch(args: argparse.Namespace) -> int:
    from pathlib import Path

    from microreason.data.fetch import ChecksumError, fetch_source
    from microreason.data.sources import load_sources

    try:
        cfg = load_sources(args.config)
    except (ValidationError, ValueError, OSError) as exc:
        print(f"error: invalid sources config {args.config}:\n{exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    try:
        for source in cfg.sources:
            paths = fetch_source(source, Path(args.out))
            for p in paths:
                print(p)
    except ChecksumError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    return EXIT_OK


def _cmd_model_inspect(args: argparse.Namespace) -> int:
    from pathlib import Path

    from microreason.inspect_report import model_report

    try:
        load_model_config(args.config)
    except (ValidationError, ValueError, OSError) as exc:
        print(f"error: invalid model config {args.config}:\n{exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    out = model_report(args.config, measure_mps=args.measure)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))
    return EXIT_OK if out["counts_match"] else EXIT_CONFIG_ERROR


def _cmd_train(args: argparse.Namespace) -> int:
    from datetime import UTC, datetime
    from pathlib import Path

    from microreason.checkpoint import latest_checkpoint
    from microreason.train.pretrain import CAUSAL_LM_STAGES, build_trainer, default_device

    try:
        resolved = load_stage_config(args.config)
        if args.parent:
            stage = resolved.stage.model_copy(update={"parent": args.parent})
            resolved = resolved.model_copy(update={"stage": stage})
    except (ValidationError, ValueError, OSError) as exc:
        print(f"error: invalid config {args.config}:\n{exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    stage = resolved.stage
    if stage.stage not in CAUSAL_LM_STAGES:
        print(f"error: stage '{stage.stage}' is not implemented yet.", file=sys.stderr)
        return EXIT_NOT_IMPLEMENTED
    if stage.budget.paid_compute_allowed:
        print("error: paid compute is gated until an explicitly authorized pilot.", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    chash = resolved.config_hash()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.run_dir or f"runs/{stage.stage}_{stamp}_{chash[7:15]}")
    device = args.device or default_device()
    resume = None
    if args.resume:
        resume = latest_checkpoint(run_dir) if args.resume == "latest" else Path(args.resume)
        if resume is None:
            print(f"error: no checkpoint to resume in {run_dir}", file=sys.stderr)
            return EXIT_CONFIG_ERROR

    try:
        trainer = build_trainer(resolved, run_dir, device)
    except (ValueError, FileNotFoundError, NotImplementedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    summary = {
        "run_dir": str(run_dir),
        "config_hash": chash,
        "parent": stage.parent,
        "device": device,
        "total_steps": trainer.total_steps,
        "planned_tokens": trainer.total_steps * stage.effective_batch_tokens,
        "tokenizer_sha256": trainer.provenance["tokenizer_sha256"],
        "train_datasets": trainer.provenance["train_datasets"],
        "projected_spend_usd": 0.0,
        "paid_compute_allowed": stage.budget.paid_compute_allowed,
        "resume_from": str(resume) if resume else None,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    result = trainer.train(resume_from=resume)
    final = {
        "final_checkpoint": str(result.final_checkpoint),
        "steps": result.steps,
        "tokens": result.tokens,
        "final_eval": result.final_eval,
    }
    print(json.dumps(final, indent=2))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="microreason",
        description="MicroReason experiment pipeline. Phase 1 scaffold: most commands are stubs.",
    )
    p.add_argument("--version", action="version", version=f"microreason {__version__}")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # config
    cfg = sub.add_parser("config", help="Validate and resolve configs.")
    cfg_sub = cfg.add_subparsers(dest="config_command", required=True, metavar="SUBCOMMAND")
    v = cfg_sub.add_parser(
        "validate", help="Validate a stage config; print resolved JSON and hash."
    )
    v.add_argument("--config", required=True)
    v.set_defaults(func=_cmd_config_validate)

    # model
    mdl = sub.add_parser("model", help="Model utilities.")
    mdl_sub = mdl.add_subparsers(dest="model_command", required=True, metavar="SUBCOMMAND")
    mi = mdl_sub.add_parser(
        "inspect", help="Print the analytic parameter count for a model config."
    )
    mi.add_argument("--config", required=True)
    mi.add_argument("--report", default=None, help="Also write the report JSON here.")
    mi.add_argument("--measure", action="store_true", help="Time a training step on MPS.")
    mi.set_defaults(func=_cmd_model_inspect)

    # data
    data = sub.add_parser("data", help="Generate and audit datasets (Phase 2).")
    data_sub = data.add_subparsers(dest="data_command", required=True, metavar="SUBCOMMAND")
    dg = data_sub.add_parser("generate", help="Generate synthetic math data and manifests.")
    dg.add_argument("--config", default="configs/data.yaml")
    dg.add_argument("--out", default=None, help="Override out_dir from the config.")
    dg.add_argument("--manifest-dir", default=None, help="Override manifest_dir.")
    dg.set_defaults(func=_cmd_data_generate)
    da = data_sub.add_parser("audit", help="Audit manifests: hashes, checks, split overlap.")
    da.add_argument("--manifest-dir", default="data/manifests")
    da.add_argument("--version", default="v1")
    da.add_argument("--report", default="reports/data_audit_v1.json")
    da.set_defaults(func=_cmd_data_audit)
    dt = data_sub.add_parser("tokenize", help="Pre-tokenize train/val text and math datasets.")
    dt.add_argument("--tokenizer", default="artifacts/tokenizer/v1")
    dt.add_argument("--out", default="data/tokenized")
    dt.set_defaults(func=_cmd_data_tokenize)
    ds = data_sub.add_parser("split-text", help="Hash-split general text into train/val.")
    ds.add_argument("--sources", default="configs/data_sources.yaml")
    ds.add_argument("--raw-dir", default="data/raw")
    ds.add_argument("--out", default="data/manifests/general_text_split_v1.json")
    ds.set_defaults(func=_cmd_data_split_text)
    dz = data_sub.add_parser("freeze", help="Freeze evaluation suite hashes after review.")
    dz.add_argument("--manifest-dir", default="data/manifests")
    dz.add_argument("--version", default="v1")
    dz.set_defaults(func=_cmd_data_freeze)
    df = data_sub.add_parser("fetch-sources", help="Download pinned text sources; verify sha256.")
    df.add_argument("--config", default="configs/data_sources.yaml")
    df.add_argument("--out", default="data/raw")
    df.set_defaults(func=_cmd_data_fetch)

    # tokenizer
    tok = sub.add_parser("tokenizer", help="Tokenizer utilities (Phase 3).")
    tok_sub = tok.add_subparsers(dest="tokenizer_command", required=True, metavar="SUBCOMMAND")
    tt = tok_sub.add_parser("train", help="Train the byte-level BPE tokenizer.")
    tt.add_argument("--config", default="configs/tokenizer.yaml")
    tt.set_defaults(func=_cmd_tokenizer_train)
    ti = tok_sub.add_parser("inspect", help="Tokenizer diagnostics and corpus token counts.")
    ti.add_argument("--tokenizer", default="artifacts/tokenizer/v1")
    ti.add_argument("--report", default="reports/tokenizer_v1.json")
    ti.add_argument("--skip-corpus", action="store_true", help="Skip general-text token counts.")
    ti.set_defaults(func=_cmd_tokenizer_inspect)

    # train
    tr = sub.add_parser("train", help="Run a causal-LM training stage locally.")
    tr.add_argument("--config", required=True)
    tr.add_argument("--parent", default=None, help="Parent checkpoint directory.")
    tr.add_argument("--run-dir", default=None)
    tr.add_argument("--device", default=None, help="cpu, mps or cuda (default: best available).")
    tr.add_argument("--resume", default=None, help="Checkpoint path, or 'latest' in --run-dir.")
    tr.set_defaults(func=_cmd_train)

    # eval / compare / serve
    ev = sub.add_parser("eval", help="Evaluate a checkpoint on a frozen suite.")
    ev.add_argument("--suite", required=True)
    ev.add_argument("--checkpoint", required=True)
    ev.set_defaults(func=_not_implemented(8, "eval"))

    cmp_ = sub.add_parser("compare", help="Compare runs with paired intervals.")
    cmp_.add_argument("--runs", required=True)
    cmp_.set_defaults(func=_not_implemented(8, "compare"))

    sv = sub.add_parser("serve", help="Serve a checkpoint locally (loopback by default).")
    sv.add_argument("--checkpoint", required=True)
    sv.add_argument("--host", default="127.0.0.1")
    sv.set_defaults(func=_not_implemented(8, "serve"))

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
