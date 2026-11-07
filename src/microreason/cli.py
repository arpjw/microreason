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


def _cmd_model_inspect(args: argparse.Namespace) -> int:
    try:
        model = load_model_config(args.config)
    except (ValidationError, ValueError, OSError) as exc:
        print(f"error: invalid model config {args.config}:\n{exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    out = {
        "name": model.name,
        "parameter_count_analytic": model.parameter_count(),
        "parameter_count_instantiated": None,
        "note": "Analytic count only; the model is not implemented until Phase 3.",
        "config": model.model_dump(mode="json"),
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return EXIT_OK


def _cmd_train(args: argparse.Namespace) -> int:
    try:
        resolved = load_stage_config(args.config)
    except (ValidationError, ValueError, OSError) as exc:
        print(f"error: invalid config {args.config}:\n{exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    budget = resolved.stage.budget
    log.info("stage=%s config_hash=%s", resolved.stage.stage, resolved.config_hash())
    log.info(
        "paid_compute_allowed=%s max_spend_usd=%.2f",
        budget.paid_compute_allowed,
        budget.max_spend_usd,
    )
    print("error: training is not implemented yet (planned for Phase 4).", file=sys.stderr)
    return EXIT_NOT_IMPLEMENTED


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
    mi.set_defaults(func=_cmd_model_inspect)

    # data
    data = sub.add_parser("data", help="Generate and audit datasets (Phase 2).")
    data_sub = data.add_subparsers(dest="data_command", required=True, metavar="SUBCOMMAND")
    dg = data_sub.add_parser("generate", help="Generate synthetic math data.")
    dg.add_argument("--config", required=True)
    dg.add_argument("--out", required=True)
    dg.set_defaults(func=_not_implemented(2, "data generate"))
    da = data_sub.add_parser("audit", help="Audit a data manifest for splits and overlap.")
    da.add_argument("--manifest", required=True)
    da.set_defaults(func=_not_implemented(2, "data audit"))

    # tokenizer
    tok = sub.add_parser("tokenizer", help="Tokenizer utilities (Phase 3).")
    tok_sub = tok.add_subparsers(dest="tokenizer_command", required=True, metavar="SUBCOMMAND")
    tt = tok_sub.add_parser("train", help="Train the byte-level BPE tokenizer.")
    tt.add_argument("--config", required=True)
    tt.set_defaults(func=_not_implemented(3, "tokenizer train"))

    # train
    tr = sub.add_parser("train", help="Run a training stage (Phase 4+).")
    tr.add_argument("--config", required=True)
    tr.add_argument("--parent", default=None)
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
