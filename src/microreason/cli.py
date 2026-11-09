"""Command-line entry point. Every command is noninteractive and scriptable (spec §9)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

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


def _cmd_eval(args: argparse.Namespace) -> int:
    from microreason.eval.report import SealedSuiteError, run_eval
    from microreason.train.pretrain import default_device

    try:
        summary = run_eval(
            args.suite, args.checkpoint, args.format, args.out,
            tokenizer_dir=args.tokenizer, allow_test=args.allow_test, samples=args.samples,
            temperature=args.temperature, seed=args.seed, max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size, limit=args.limit, device=args.device or default_device(),
            data_version=args.data_version, kv_cache=args.kv_cache,
        )  # fmt: skip
    except (SealedSuiteError, ValueError, FileExistsError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    print(json.dumps(summary, indent=2, sort_keys=True))
    return EXIT_OK


def _cmd_compare(args: argparse.Namespace) -> int:
    from microreason.eval.report import compare_runs

    try:
        result = compare_runs(args.a, args.b, args.resamples, args.seed)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    print(json.dumps(result, indent=2, sort_keys=True))
    return EXIT_OK


def _cmd_data_generate_stream(args: argparse.Namespace) -> int:
    from microreason.data.stream import generate_stream, load_stream_config

    try:
        m = generate_stream(load_stream_config(args.config))
    except (ValidationError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    keys = ("record_count", "achieved", "capacity_limited", "by_difficulty", "shards", "sha256")
    print(json.dumps({k: m[k] for k in keys}, indent=2, sort_keys=True))
    return EXIT_OK


def _cmd_data_audit_stream(args: argparse.Namespace) -> int:
    from pathlib import Path

    from microreason.data.stream import audit_stream, load_stream_config

    report = audit_stream(load_stream_config(args.config), args.sample)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return EXIT_OK if report["passed"] else EXIT_CONFIG_ERROR


def _cmd_data_tokenize_stream(args: argparse.Namespace) -> int:
    from pathlib import Path

    from microreason.data.stream import load_stream_config
    from microreason.data.tokenized import MANIFEST_FILE, tokenize_math_stream
    from microreason.tokenizer import MicroTokenizer

    cfg = load_stream_config(args.config)
    tok = MicroTokenizer.load(args.tokenizer)
    pre, ver = cfg.tokenized_prefix, cfg.version
    jobs = {
        f"{pre}_a_{ver}": ("a", None),
        f"{pre}_b_easy_{ver}": ("b", (1, 2)),
        f"{pre}_b_hard_{ver}": ("b", (3, 4)),
    }
    out = {}
    for name, (shard, levels) in jobs.items():
        d = Path(args.out) / name
        if (d / MANIFEST_FILE).exists():
            m = json.loads((d / MANIFEST_FILE).read_text())
        else:
            log.info("tokenizing %s", name)
            m = tokenize_math_stream(tok, cfg.manifest_path, shard, d, levels, name)
        out[name] = {"documents": m["documents"], "tokens": m["tokens"]}
    print(json.dumps(out, indent=2, sort_keys=True))
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
    if Path("data/manifests/math_val_v3.json").exists():
        jobs["math_train_v3"] = lambda d: tokenize_math(tok, "train", d, "v3")
        jobs["math_val_v3"] = lambda d: tokenize_math(tok, "val", d, "v3")
    stream_manifest = Path("data/manifests/math_train_stream_v1.json")
    if stream_manifest.exists():
        from microreason.data.tokenized import tokenize_math_stream

        for shard in ("a", "b"):
            jobs[f"math_stream_{shard}_v1"] = lambda d, sh=shard: tokenize_math_stream(
                tok, stream_manifest, sh, d
            )
        # Curriculum buckets of the mid-training shard.
        for bucket, levels in (("easy", (1, 2)), ("hard", (3, 4))):
            jobs[f"math_stream_b_{bucket}_v1"] = lambda d, lv=levels, b=bucket: (
                tokenize_math_stream(tok, stream_manifest, "b", d, lv, f"math_stream_b_{b}")
            )
    out = {}
    if args.datasets:
        wanted = set(args.datasets.split(","))
        unknown = wanted - set(jobs)
        if unknown:
            print(f"error: unknown datasets {sorted(unknown)}", file=sys.stderr)
            return EXIT_CONFIG_ERROR
        jobs = {k: v for k, v in jobs.items() if k in wanted}
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


def _cmd_data_generate_v3(args: argparse.Namespace) -> int:
    from microreason.data import build_v3 as b3

    try:
        cfg = b3.load_v3_config(args.config)
        result = b3.build_v3(cfg)
        b3.guard_against_frozen_v3(cfg, result)
    except (ValidationError, ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    for path in b3.write_v3(cfg, result):
        log.info("wrote %s", path)
    try:
        frozen = b3.freeze_v3(cfg.manifest_dir, cfg.version)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    print(json.dumps({"frozen": str(frozen), "records": {k: len(v) for k, v in
                      result["records"].items()}, "corpus_scan": result["corpus_scan"],
                      "coverage": result["coverage"]}, indent=2, sort_keys=True))  # fmt: skip
    return EXIT_OK


def _cmd_data_audit_v3(args: argparse.Namespace) -> int:
    from microreason.data import build_v3 as b3

    report = b3.audit_v3(args.manifest_dir, args.version, regenerate=args.regenerate)
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return EXIT_OK if report["passed"] else EXIT_CONFIG_ERROR


def _cmd_data_generate_stream_v3(args: argparse.Namespace) -> int:
    from microreason.data import build_v3 as b3

    try:
        m = b3.build_stream_v3(b3.load_v3_config(args.config))
    except (ValidationError, ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    print(json.dumps({k: m[k] for k in ("record_count", "sha256", "shards", "by_difficulty",
                                        "distinct_shapes")}, indent=2, sort_keys=True))  # fmt: skip
    return EXIT_OK


def _cmd_data_tokenize_stream_v3(args: argparse.Namespace) -> int:
    from pathlib import Path

    from microreason.data import build_v3 as b3
    from microreason.data.tokenized import MANIFEST_FILE, tokenize_math_stream
    from microreason.tokenizer import MicroTokenizer

    cfg = b3.load_v3_config(args.config)
    tok = MicroTokenizer.load(args.tokenizer)
    pre, ver = cfg.stream.tokenized_prefix, cfg.version
    jobs = {
        f"{pre}_a_{ver}": ("a", None),
        f"{pre}_b_easy_{ver}": ("b", (1, 2)),
        f"{pre}_b_hard_{ver}": ("b", (3, 4, 5)),
    }
    out = {}
    for name, (shard, levels) in jobs.items():
        d = Path(args.out) / name
        if (d / MANIFEST_FILE).exists():
            m = json.loads((d / MANIFEST_FILE).read_text())
        else:
            log.info("tokenizing %s", name)
            m = tokenize_math_stream(tok, cfg.stream.manifest_path, shard, d, levels, name)
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


def _rl_guard(budget, args, run_dir):
    """A budget guard for a paid RL run, or None for an unpaid one. Raises ValueError."""
    import os
    from datetime import datetime
    from pathlib import Path

    if not budget.paid_compute_allowed:
        return None
    from microreason.train.budget import BudgetGuard

    pod_id = args.pod_id or os.environ.get("RUNPOD_POD_ID")
    start = args.billing_start or os.environ.get("MICROREASON_BILLING_START")
    if not pod_id or not start:
        raise ValueError("paid runs need --pod-id and --billing-start (or RUNPOD_POD_ID and "
                         "MICROREASON_BILLING_START)")  # fmt: skip
    started = datetime.fromisoformat(start.replace("Z", "+00:00")).timestamp()
    return BudgetGuard(budget, started, pod_id, Path(run_dir) / "budget_ledger.json")


def _load_rl(args):
    import yaml

    from microreason.rl.config import load_rl_config

    over = {"parent": getattr(args, "parent", None)}
    if getattr(args, "budget", None):
        with open(args.budget, encoding="utf-8") as f:
            over["budget"] = yaml.safe_load(f)
    return load_rl_config(args.config, over)


def _cmd_rl_gate(args: argparse.Namespace) -> int:
    from microreason.rl.gate import run_gate
    from microreason.train.pretrain import default_device

    try:
        resolved = _load_rl(args)
    except (ValidationError, ValueError, OSError) as exc:
        print(f"error: invalid config {args.config}:\n{exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    report = run_gate(resolved, args.checkpoint, args.out, args.device or default_device())
    print(json.dumps({"passed": report["passed"], "failures": report["failures"],
                      "overall": report["overall"]}, indent=2))  # fmt: skip
    return EXIT_OK if report["passed"] else 3


def _cmd_rl_kv_check(args: argparse.Namespace) -> int:
    from microreason.rl.kvcheck import run_kv_check
    from microreason.train.pretrain import default_device

    try:
        resolved = _load_rl(args)
    except (ValidationError, ValueError, OSError) as exc:
        print(f"error: invalid config {args.config}:\n{exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    report = run_kv_check(resolved, args.checkpoint, args.out, args.device or default_device())
    print(json.dumps(report, indent=2, sort_keys=True))
    return EXIT_OK if report["passed"] else 5


def _cmd_rl_train(args: argparse.Namespace) -> int:
    from microreason.train.pretrain import default_device

    try:
        resolved = _load_rl(args)
        guard = _rl_guard(resolved.stage.budget, args, args.run_dir)
    except (ValidationError, ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    if resolved.stage.stage != args.expect_stage:
        print(f"error: {args.config} is a {resolved.stage.stage} config", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    device = args.device or default_device()
    if resolved.stage.stage == "grpo":
        from microreason.rl.grpo import GRPOTrainer

        trainer = GRPOTrainer(resolved, args.run_dir, device, guard)
    else:
        from microreason.rl.dpo import DPOTrainer

        trainer = DPOTrainer(resolved, args.run_dir, device, guard)
    result = trainer.train()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return EXIT_OK if result["status"] == "completed" else 4


def _cmd_train(args: argparse.Namespace) -> int:
    from datetime import UTC, datetime
    from pathlib import Path

    from microreason.checkpoint import latest_checkpoint
    from microreason.train.pretrain import CAUSAL_LM_STAGES, build_trainer, default_device

    try:
        resolved = load_stage_config(args.config)
        updates = {
            k: v
            for k, v in {
                "parent": args.parent,
                "micro_batch_size": args.micro_batch_size,
                "max_steps": args.max_steps,
                "max_wall_seconds": getattr(args, "max_wall_seconds", None),
            }.items()
            if v is not None
        }
        if args.budget:
            import yaml

            with open(args.budget, encoding="utf-8") as f:
                updates["budget"] = yaml.safe_load(f)
        if updates:
            from microreason.config import ResolvedConfig, StageConfig

            stage = StageConfig.model_validate({**resolved.stage.model_dump(), **updates})
            resolved = ResolvedConfig(stage=stage, model=resolved.model)
    except (ValidationError, ValueError, OSError) as exc:
        print(f"error: invalid config {args.config}:\n{exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    stage = resolved.stage
    if stage.stage not in (*CAUSAL_LM_STAGES, "sft", "extra_sft"):
        print(f"error: stage '{stage.stage}' is not implemented yet.", file=sys.stderr)
        return EXIT_NOT_IMPLEMENTED
    chash = resolved.config_hash()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.run_dir or f"runs/{stage.stage}_{stamp}_{chash[7:15]}")

    guard = None
    if stage.budget.paid_compute_allowed:
        import os

        from microreason.train.budget import BudgetGuard

        pod_id = args.pod_id or os.environ.get("RUNPOD_POD_ID")
        start = args.billing_start or os.environ.get("MICROREASON_BILLING_START")
        if not pod_id or not start:
            print("error: paid runs need --pod-id and --billing-start (or RUNPOD_POD_ID and "
                  "MICROREASON_BILLING_START).", file=sys.stderr)  # fmt: skip
            return EXIT_CONFIG_ERROR
        try:
            started = datetime.fromisoformat(start.replace("Z", "+00:00")).timestamp()
            guard = BudgetGuard(stage.budget, started, pod_id, run_dir / "budget_ledger.json")
        except (ValueError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_CONFIG_ERROR
    device = args.device or default_device()
    resume = None
    if args.resume:
        resume = latest_checkpoint(run_dir) if args.resume == "latest" else Path(args.resume)
        if resume is None:
            print(f"error: no checkpoint to resume in {run_dir}", file=sys.stderr)
            return EXIT_CONFIG_ERROR

    try:
        if stage.stage in ("sft", "extra_sft"):
            from microreason.train.sft import build_sft_trainer

            trainer = build_sft_trainer(resolved, run_dir, device, guard)
        else:
            trainer = build_trainer(resolved, run_dir, device, guard)
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
        "paid_compute_allowed": stage.budget.paid_compute_allowed,
        "accrued_spend_usd_so_far": round(guard.accrued_usd(), 4) if guard else 0.0,
        "approved_max_usd": stage.budget.max_spend_usd,
        "hard_stop_at_hours_after_billing_start": round(stage.budget.ceiling_hours(), 3),
        "resume_from": str(resume) if resume else None,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    result = trainer.train(resume_from=resume)
    final = {
        "stop_reason": result.stop_reason,
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
    dgs = data_sub.add_parser("generate-stream", help="Build the training-only math stream.")
    dgs.add_argument("--config", default="configs/data_stream.yaml")
    dgs.set_defaults(func=_cmd_data_generate_stream)
    das = data_sub.add_parser("audit-stream", help="Leakage check + sampled re-verification.")
    das.add_argument("--config", default="configs/data_stream.yaml")
    das.add_argument("--sample", type=float, default=0.02)
    das.add_argument("--report", default="reports/data_stream_audit_v1.json")
    das.set_defaults(func=_cmd_data_audit_stream)
    dv3 = data_sub.add_parser("generate-v3", help="Data v3: build, audit and freeze suites.")
    dv3.add_argument("--config", default="configs/data_v3.yaml")
    dv3.set_defaults(func=_cmd_data_generate_v3)
    dav3 = data_sub.add_parser("audit-v3", help="Data v3: cross-split audit + regeneration.")
    dav3.add_argument("--manifest-dir", default="data/manifests")
    dav3.add_argument("--version", default="v3")
    dav3.add_argument("--regenerate", type=int, default=2000)
    dav3.add_argument("--report", default=None)
    dav3.set_defaults(func=_cmd_data_audit_v3)
    dsv3 = data_sub.add_parser("generate-stream-v3", help="Data v3: training-only stream.")
    dsv3.add_argument("--config", default="configs/data_v3.yaml")
    dsv3.set_defaults(func=_cmd_data_generate_stream_v3)
    dtv3 = data_sub.add_parser("tokenize-stream-v3", help="Data v3: tokenize stream shards.")
    dtv3.add_argument("--config", default="configs/data_v3.yaml")
    dtv3.add_argument("--tokenizer", default="artifacts/tokenizer/v2")
    dtv3.add_argument("--out", default="data/tokenized_v2")
    dtv3.set_defaults(func=_cmd_data_tokenize_stream_v3)
    dt = data_sub.add_parser("tokenize", help="Pre-tokenize train/val text and math datasets.")
    dt.add_argument("--tokenizer", default="artifacts/tokenizer/v1")
    dt.add_argument("--out", default="data/tokenized")
    dt.add_argument("--datasets", default=None, help="Comma-separated subset of dataset names.")
    dt.set_defaults(func=_cmd_data_tokenize)
    dts = data_sub.add_parser("tokenize-stream", help="Tokenize a stream's shards and buckets.")
    dts.add_argument("--config", required=True)
    dts.add_argument("--tokenizer", default="artifacts/tokenizer/v1")
    dts.add_argument("--out", default="data/tokenized")
    dts.set_defaults(func=_cmd_data_tokenize_stream)
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
    tr.add_argument("--micro-batch-size", type=int, default=None, help="Override the config.")
    tr.add_argument("--max-steps", type=int, default=None, help="Override the config.")
    tr.add_argument(
        "--max-wall-seconds",
        type=float,
        default=None,
        help="Stop as completed after this much training-loop time.",
    )
    tr.add_argument("--pod-id", default=None, help="Paid runs: provider pod id.")
    tr.add_argument("--billing-start", default=None, help="Paid runs: ISO time billing began.")
    tr.add_argument("--budget", default=None, help="YAML budget that replaces the config's own.")
    tr.set_defaults(func=_cmd_train)

    # rl branches (gate, grpo, dpo)
    rl = sub.add_parser("rl", help="RL-stage branches: reward-variance gate, GRPO, DPO.")
    rl_sub = rl.add_subparsers(dest="rl_command", required=True)
    rg = rl_sub.add_parser("gate", help="Reward-variance gate on the RL training distribution.")
    rg.add_argument("--config", required=True, help="A grpo config (sampling settings + rule).")
    rg.add_argument("--checkpoint", required=True)
    rg.add_argument("--out", required=True)
    rg.add_argument("--device", default=None)
    rg.set_defaults(func=_cmd_rl_gate)
    rk = rl_sub.add_parser("kv-check", help="KV-cache sampler parity on the real checkpoint.")
    rk.add_argument("--config", required=True, help="A grpo config (sampling settings).")
    rk.add_argument("--checkpoint", required=True)
    rk.add_argument("--out", required=True)
    rk.add_argument("--device", default=None)
    rk.set_defaults(func=_cmd_rl_kv_check)
    for name in ("grpo", "dpo"):
        rt = rl_sub.add_parser(name, help=f"Train the {name} branch from v2-sft.")
        rt.add_argument("--config", required=True)
        rt.add_argument("--parent", default=None)
        rt.add_argument("--run-dir", required=True)
        rt.add_argument("--device", default=None)
        rt.add_argument("--budget", default=None)
        rt.add_argument("--pod-id", default=None)
        rt.add_argument("--billing-start", default=None)
        rt.set_defaults(func=_cmd_rl_train, expect_stage=name)

    # eval / compare / serve
    ev = sub.add_parser("eval", help="Evaluate a checkpoint on a frozen suite.")
    ev.add_argument("--suite", required=True, help="val, test_id, test_extrapolation, ...")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--format", required=True, help="plain-v1, sft-show-work-v1, sft-direct-v1")
    ev.add_argument("--out", required=True)
    ev.add_argument("--allow-test", action="store_true", help="Unseal a test suite.")
    ev.add_argument("--samples", type=int, default=0, help="Sampled pass@k with this many.")
    ev.add_argument("--temperature", type=float, default=0.8)
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--max-new-tokens", type=int, default=128)
    ev.add_argument("--batch-size", type=int, default=64)
    ev.add_argument("--limit", type=int, default=None, help="First N prompts (smoke only).")
    ev.add_argument("--device", default=None)
    ev.add_argument("--tokenizer", default="artifacts/tokenizer/v1")
    ev.add_argument("--data-version", default="v1", help="Frozen suite version (v1 or v3).")
    ev.add_argument("--kv-cache", action="store_true", help="Cached generator (eval-v2).")
    ev.set_defaults(func=_cmd_eval)

    cmp_ = sub.add_parser("compare", help="Paired bootstrap between two eval runs (b - a).")
    cmp_.add_argument("--a", required=True)
    cmp_.add_argument("--b", required=True)
    cmp_.add_argument("--resamples", type=int, default=10_000)
    cmp_.add_argument("--seed", type=int, default=0)
    cmp_.set_defaults(func=_cmd_compare)

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
