#!/usr/bin/env python3
"""Static, executable audit of SwitchCraft's four-stage optimizer schedule."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


def literal(node: ast.AST):
    return ast.literal_eval(node)


def function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found")


def constructor_defaults(init: ast.FunctionDef) -> dict[str, object]:
    names = [argument.arg for argument in init.args.args]
    defaults = [literal(value) for value in init.args.defaults]
    return dict(zip(names[-len(defaults) :], defaults))


def annealer_calls(optimize: ast.FunctionDef) -> list[dict[str, object]]:
    calls = []
    for node in ast.walk(optimize):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "Annealer":
            calls.append({keyword.arg: literal(keyword.value) for keyword in node.keywords})
    return calls


def run_audit(official_repo: Path) -> dict[str, object]:
    designer_path = official_repo / "designer.py"
    utils_path = official_repo / "utils" / "mydesign_utils.py"
    designer_tree = ast.parse(designer_path.read_text(encoding="utf-8"))
    utils_tree = ast.parse(utils_path.read_text(encoding="utf-8"))

    optimize = function(designer_tree, "optimize")
    annealer_init = function(utils_tree, "__init__")
    restype = function(designer_tree, "get_restype_from_logits")
    defaults = constructor_defaults(annealer_init)
    calls = annealer_calls(optimize)
    if len(calls) != 4:
        raise AssertionError(f"expected four Annealer stages, found {len(calls)}")

    alpha_argument = next(
        (argument for argument in restype.args.args if argument.arg == "alpha"), None
    )
    if alpha_argument is None:
        raise AssertionError("get_restype_from_logits has no alpha argument")
    positional_names = [argument.arg for argument in restype.args.args]
    positional_defaults = dict(
        zip(
            positional_names[-len(restype.args.defaults) :],
            [literal(value) for value in restype.args.defaults],
        )
    )
    logit_scale = float(positional_defaults["alpha"])
    if logit_scale != 2.0:
        raise AssertionError(f"unexpected logit scale: {logit_scale}")

    stages = []
    for index, overrides in enumerate(calls, start=1):
        config = defaults | overrides
        start_temp = float(config["temp"]) / logit_scale
        raw_end_temp = (
            config["temp"] if config["e_temp"] is None else config["e_temp"]
        )
        end_temp = float(raw_end_temp) / logit_scale
        stages.append(
            {
                "stage": index,
                "iterations": int(config["iters"]),
                "learning_rate": float(config["lr"]),
                "paper_beta_start": float(config["hard"]),
                "paper_beta_end": float(
                    config["hard"] if config["e_hard"] is None else config["e_hard"]
                ),
                "paper_gamma_start": float(config["soft"]),
                "paper_gamma_end": float(
                    config["soft"] if config["e_soft"] is None else config["e_soft"]
                ),
                "effective_tau_start": start_temp,
                "effective_tau_end": end_temp,
            }
        )

    expected = [
        (30, 0.2, 0.0, 0.0, 1.0, 1.0, 0.5, 0.5),
        (100, 0.1, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5),
        (100, 0.1, 0.0, 0.0, 1.0, 1.0, 0.5, 0.005),
        (10, 0.1, 1.0, 1.0, 1.0, 1.0, 0.005, 0.005),
    ]
    observed = [
        (
            stage["iterations"],
            stage["learning_rate"],
            stage["paper_beta_start"],
            stage["paper_beta_end"],
            stage["paper_gamma_start"],
            stage["paper_gamma_end"],
            stage["effective_tau_start"],
            stage["effective_tau_end"],
        )
        for stage in stages
    ]
    if observed != expected:
        raise AssertionError(f"schedule mismatch:\nobserved={observed}\nexpected={expected}")
    if sum(stage["iterations"] for stage in stages) != 240:
        raise AssertionError("the optimizer does not total 240 iterations")

    source = ast.unparse(optimize)
    if "self.logits = 2 * self.logits" not in source:
        raise AssertionError("missing z <- 2z transition after stage 2")
    if "self.get_restype_from_logits(self.logits, opt)['pseudo']" not in source:
        raise AssertionError("missing z <- z_pseudo transition after stage 1")

    return {
        "status": "passed",
        "official_repo": str(official_repo.resolve()),
        "source_sha256": {
            "designer.py": hashlib.sha256(designer_path.read_bytes()).hexdigest(),
            "utils/mydesign_utils.py": hashlib.sha256(
                utils_path.read_bytes()
            ).hexdigest(),
        },
        "paper_anchor": "Section 3.2, Algorithm 1, Appendix A.1",
        "logit_scale": logit_scale,
        "total_iterations": 240,
        "stages": stages,
        "transitions": ["stage1: z <- z_pseudo", "stage2: z <- 2z"],
        "note": (
            "Code temperatures are twice the paper temperatures because "
            "get_restype_from_logits multiplies logits by alpha=2 before softmax."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_audit(args.official_repo)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
