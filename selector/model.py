"""Adapted logistic context selector for fixed CARLA CNN–PID specialists."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import tempfile

import numpy as np


WEATHERS = ("sunny", "cloudy", "rainy")
TIMES = ("day", "sunset", "night")
FEATURE_ORDER = WEATHERS + TIMES + ("predicted_gap",)
CONTROLLERS = tuple(f"{time}_{weather}" for time in TIMES for weather in WEATHERS)
SPLITS = ("train", "validation", "test")


def encode_context(condition, gap):
    """Encode the weather preset and the fixed generalist's estimate in metres."""
    if condition not in CONTROLLERS:
        raise ValueError(f"Unknown condition: {condition!r}")
    if not math.isfinite(gap):
        raise ValueError("A selector gap estimate must be finite")
    time, weather = condition.split("_")
    return [int(weather == item) for item in WEATHERS] + [
        int(time == item) for item in TIMES
    ] + [float(gap)]


def sigmoid(logits):
    exp = np.exp(-np.abs(logits))
    return np.where(logits >= 0, 1.0 / (1.0 + exp), exp / (1.0 + exp))


class Selector:
    """One affine layer with an independent Bernoulli probability per specialist."""

    def __init__(self, weights, biases, threshold):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.biases = np.asarray(biases, dtype=np.float64)
        if self.weights.shape != (len(CONTROLLERS), len(FEATURE_ORDER)):
            raise ValueError("Selector W must have shape (9, 7)")
        if self.biases.shape != (len(CONTROLLERS),):
            raise ValueError("Selector b must have shape (9,)")
        if not np.isfinite(self.weights).all() or not np.isfinite(self.biases).all():
            raise ValueError("Selector parameters must be finite")
        self.threshold = float(threshold)
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Acceptance threshold must be between zero and one")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data["schema_version"] != 1 or tuple(data["feature_order"]) != FEATURE_ORDER:
            raise ValueError("Selector artifact has an incompatible schema or feature order")
        if tuple(data["output_order"]) != CONTROLLERS:
            raise ValueError("Selector artifact has an incompatible controller order")
        return cls(data["W"], data["b"], data["acceptance_threshold"])

    def score(self, condition, gap):
        features = encode_context(condition, gap)
        scores = sigmoid(self.weights @ features + self.biases)
        return dict(zip(CONTROLLERS, scores.tolist()))


def write_json(path, data):
    path.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def read_examples(path):
    records = []
    group_splits = {}
    seen = set()
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                split = row["split"]
                controller = row["controller"]
                case_id = row["case_id"]
                window = row["window_index"]
                condition = row["condition"]
                label = row["label"]
                if split not in SPLITS or controller not in CONTROLLERS or condition not in CONTROLLERS:
                    raise ValueError("Invalid split, controller, or condition")
                if type(window) is not int or window < 0 or type(case_id) is not str or not case_id:
                    raise ValueError("Invalid window index or case ID")
                if label is not None:
                    features = row["features"]
                    if type(label) is not int or label not in (0, 1):
                        raise ValueError("Label must be 0, 1 or null")
                    if len(features) != len(FEATURE_ORDER) or not np.isfinite(features).all():
                        raise ValueError("Expected seven finite features for a labeled window")
                    if features != encode_context(condition, features[-1]):
                        raise ValueError("Weather and time features do not match the condition")
                key = (case_id, controller, window)
                if key in seen:
                    raise ValueError(f"Duplicate example {key!r}")
                seen.add(key)
                groups = {
                    "case_id": case_id,
                    "source_route": row.get("source_route", row.get("route_id")),
                    "seed": row.get("seed"),
                }
                for field, group in groups.items():
                    if group is None:
                        continue
                    group_key = field, json.dumps(group, sort_keys=True)
                    if group_splits.setdefault(group_key, split) != split:
                        raise ValueError(f"{field} {group!r} appears in multiple splits")
                records.append(row)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number} {error}") from error
    return records


def label_counts(records):
    counts = Counter((row["split"], row["controller"], row["label"]) for row in records)
    return {
        split: {
            name: {
                "success": counts[split, name, 1],
                "failure": counts[split, name, 0],
                "censored": counts[split, name, None],
            }
            for name in CONTROLLERS
        }
        for split in SPLITS
    }


def fit_head(features, labels, l2):
    import torch
    from torch.nn.functional import binary_cross_entropy_with_logits

    x = torch.as_tensor(features, dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.float64)
    params = torch.zeros(len(FEATURE_ORDER) + 1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [params], max_iter=160, tolerance_grad=1e-9, tolerance_change=1e-13,
        line_search_fn="strong_wolfe",
    )

    def objective():
        optimizer.zero_grad()
        coefficients, intercept = params[:-1], params[-1]
        loss = binary_cross_entropy_with_logits(x @ coefficients + intercept, y)
        loss = loss + 0.5 * l2 * coefficients.square().sum()
        loss.backward()
        return loss

    optimizer.step(objective)
    fitted = params.detach().numpy().copy()
    if not np.isfinite(fitted).all():
        raise ValueError("Logistic optimization returned nonfinite coefficients")
    return fitted


def train_selector(records, l2):
    import torch

    torch.set_num_threads(1)
    rows = [row for row in records if row["split"] == "train" and row["label"] is not None]
    gaps = np.asarray([row["features"][-1] for row in rows], dtype=np.float64)
    mean = float(gaps.mean())
    std = float(gaps.std())
    scale = std or 1.0
    weights, biases = [], []
    for controller in CONTROLLERS:
        subset = [row for row in rows if row["controller"] == controller]
        x = np.asarray([row["features"] for row in subset], dtype=np.float64)
        y = [row["label"] for row in subset]
        x[:, -1] = (x[:, -1] - mean) / scale
        params = fit_head(x, y, l2)
        # Fold the train-only standardization into the affine map on raw inputs.
        weights.append(params[:-1].copy())
        weights[-1][-1] /= scale
        biases.append(params[-1] - weights[-1][-1] * mean)
    return Selector(weights, biases, 0.0), {
        "weather_and_time": "Fixed one-hot indicators in feature order",
        "gap_source": "Frozen generalist ResNet-18 inference in metres",
        "training_gap_mean_m": mean,
        "training_gap_scale_m": scale,
        "gap_scaler_folded_into_W_and_b": True,
    }


def matched_start_choices(records, selector):
    grouped = defaultdict(dict)
    for row in records:
        if row["window_index"] == 0:
            grouped[row["case_id"]][row["controller"]] = row
    choices = []
    for case_id, rows in sorted(grouped.items()):
        if set(rows) != set(CONTROLLERS) or any(row["label"] is None for row in rows.values()):
            continue
        condition = rows[CONTROLLERS[0]]["condition"]
        if any(row["condition"] != condition for row in rows.values()):
            raise ValueError(f"Inconsistent conditions in matched case {case_id}")
        # A fixed reference before any candidate command yields one context used
        # to score all nine. Later windows follow different ego trajectories.
        gap = rows[condition]["features"][-1]
        probs = selector.score(condition, gap)
        selected = max(probs, key=probs.get)
        gaps = [row["features"][-1] for row in rows.values()]
        choices.append({
            "case_id": case_id,
            "condition": condition,
            "context_controller": condition,
            "controller": selected,
            "probability": probs[selected],
            "selected_label": rows[selected]["label"],
            "weather_matched_label": rows[condition]["label"],
            "best_observed_label": max(row["label"] for row in rows.values()),
            "generalist_gap_range_m": max(gaps) - min(gaps),
        })
    return choices, len(grouped)


def probability_metrics(labels, probabilities):
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    if not len(y):
        return {"n": 0}
    safe_p = p.clip(1e-12, 1 - 1e-12)
    return {
        "n": len(y),
        "log_loss": float(-(y * np.log(safe_p) + (1 - y) * np.log1p(-safe_p)).mean()),
        "brier": float(((y - p) ** 2).mean()),
        "accuracy_at_0_5": float(((p >= 0.5) == y).mean()),
    }


def acceptance_metrics(labels, probabilities, threshold):
    y = np.asarray(labels, dtype=np.bool_)
    accepted = np.asarray(probabilities) >= threshold
    successes = int(y.sum())
    failures = len(y) - successes
    n_accepted = int(accepted.sum())
    failures_accepted = int((accepted & ~y).sum())
    sensitivity = int((accepted & y).sum()) / successes if successes else None
    specificity = int((~accepted & ~y).sum()) / failures if failures else None
    return {
        "threshold": float(threshold),
        "successes": successes,
        "failures": failures,
        "accepted": n_accepted,
        "coverage": n_accepted / len(y),
        "accepted_failures": failures_accepted,
        "failure_rate_when_accepted": failures_accepted / n_accepted if n_accepted else None,
        "balanced_accuracy": (sensitivity + specificity) / 2 if successes and failures else None,
    }


def individual_scores(records, selector):
    rows = [row for row in records if row["label"] is not None]
    scores = []
    for row in rows:
        score = selector.score(row["condition"], row["features"][-1])
        scores.append(score[row["controller"]])
    return rows, scores


def select_threshold(records, selector):
    choices, _ = matched_start_choices(records, selector)
    labels = [choice["selected_label"] for choice in choices]
    probabilities = [choice["probability"] for choice in choices]
    source = "Highest score on complete matched starting cases"
    if len(set(labels)) < 2:
        rows, probabilities = individual_scores(records, selector)
        labels = [row["label"] for row in rows]
        source = "All labeled validation windows scored by the observed controller"
    if not labels:
        raise ValueError("No labeled validation windows for acceptance threshold selection")
    candidates = sorted(set(probabilities) | {0.0, 1.0})
    operating_points = [acceptance_metrics(labels, probabilities, value) for value in candidates]
    if len(set(labels)) == 2:
        chosen = max(operating_points, key=lambda row: (row["balanced_accuracy"], row["coverage"]))
        criterion = "Highest validation balanced accuracy, then maximum coverage"
    else:
        chosen = operating_points[len(operating_points) - 1 if labels[0] == 0 else 0]
        criterion = "Validation had one class; accept all observed successes or reject observed failures"
    return chosen["threshold"], {
        "source": source,
        "criterion": criterion,
        "selected": chosen,
        "operating_points": operating_points,
        "active_live_gate": False,
    }


def evaluate(records, selector):
    rows, scores = individual_scores(records, selector)
    metrics = probability_metrics([row["label"] for row in rows], scores)
    conditioned = defaultdict(list)
    for row, score in zip(rows, scores):
        conditioned[row["controller"]].append((row["label"], score))
    by_controller = {}
    for name in CONTROLLERS:
        pairs = conditioned[name]
        labels, probabilities = zip(*pairs) if pairs else ([], [])
        by_controller[name] = probability_metrics(labels, probabilities)
    choices, n_groups = matched_start_choices(records, selector)
    n_matched = len(choices)
    matched = {"starting_cases": n_groups, "complete_cases": n_matched}
    if n_matched:
        labels = [choice["selected_label"] for choice in choices]
        probabilities = [choice["probability"] for choice in choices]
        matched.update({
            "selected_successes": sum(labels),
            "weather_matched_successes": sum(row["weather_matched_label"] for row in choices),
            "best_observed_successes": sum(row["best_observed_label"] for row in choices),
            "max_shared_estimator_start_gap_spread_m": max(
                row["generalist_gap_range_m"] for row in choices
            ),
            "probability_metrics": probability_metrics(labels, probabilities),
            "advisory_threshold": acceptance_metrics(labels, probabilities, selector.threshold),
            "cases": choices,
        })
    return {"pooled": metrics, "by_controller": by_controller, "matched": matched}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--l2", default=0.01, type=float)
    args = parser.parse_args()
    if not math.isfinite(args.l2) or args.l2 < 0:
        parser.error("--l2 must be finite and nonnegative")
    records = read_examples(args.examples)
    counts = label_counts(records)
    missing = [
        controller for controller, by_label in counts["train"].items()
        if not by_label["success"] or not by_label["failure"]
    ]
    report_path = args.output.with_name(f"{args.output.stem}_report.json")
    report = {
        "label_counts": counts, "controllers_without_two_train_classes": missing,
        "examples": str(args.examples),
    }
    if missing:
        write_json(report_path, report)
        print(f"Cannot fit controllers without both train classes: {', '.join(missing)}")
        return 2
    selector, preprocessing = train_selector(records, args.l2)
    valid = [row for row in records if row["split"] == "validation"]
    selector.threshold, report["validation_threshold"] = select_threshold(valid, selector)
    for split in ("validation", "test"):
        report[split] = evaluate([row for row in records if row["split"] == split], selector)
    artifact = {
        "schema_version": 1,
        "method": "Adapted logistic context selector",
        "feature_order": list(FEATURE_ORDER),
        "output_order": list(CONTROLLERS),
        "trainable_linear_layers": 1,
        "hidden_layers": 0,
        "learned_parameters": 72,
        "preprocessing": preprocessing,
        "fit": {
            "algorithm": "Nine independent CPU LBFGS binary logistic regressions",
            "l2": args.l2,
            "loss": "Mean unweighted binary cross-entropy plus L2 without intercept penalty",
        },
        "W": selector.weights.tolist(),
        "b": selector.biases.tolist(),
        "acceptance_threshold": selector.threshold,
        "fallback": "Integration pending; argmax remains active below the advisory threshold",
    }
    write_json(args.output, artifact)
    write_json(report_path, report)
    print(f"Fitted nine 8-parameter logistic regressions at {args.output}")
    print(f"Validation-selected advisory threshold {selector.threshold:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
