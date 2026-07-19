from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


Rating = Tuple[int, int, float]
TestGroup = Tuple[int, List[int]]


# =========================
# 1. Data loading / writing
# =========================

def read_train(path: str) -> List[Rating]:
    """Read train.txt in the required format."""
    ratings: List[Rating] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        line_no = 0
        while True:
            header = f.readline()
            line_no += 1
            if not header:
                break
            header = header.strip()
            if not header:
                continue
            if "|" not in header:
                raise ValueError(f"Bad header at line {line_no}: {header!r}")

            user_str, n_str = header.split("|", 1)
            user = int(user_str)
            n = int(n_str)

            for _ in range(n):
                line = f.readline()
                line_no += 1
                if not line:
                    raise ValueError(f"Unexpected EOF after header {header!r}")
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(f"Bad rating line at line {line_no}: {line!r}")
                item = int(parts[0])
                score = float(parts[1])
                ratings.append((user, item, score))
    return ratings


def read_test_groups(path: str) -> List[TestGroup]:
    """Read test.txt and preserve the original user grouping/order."""
    groups: List[TestGroup] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        line_no = 0
        while True:
            header = f.readline()
            line_no += 1
            if not header:
                break
            header = header.strip()
            if not header:
                continue
            if "|" not in header:
                raise ValueError(f"Bad header at line {line_no}: {header!r}")

            user_str, n_str = header.split("|", 1)
            user = int(user_str)
            n = int(n_str)

            items: List[int] = []
            for _ in range(n):
                line = f.readline()
                line_no += 1
                if not line:
                    raise ValueError(f"Unexpected EOF after header {header!r}")
                parts = line.split()
                if len(parts) < 1:
                    raise ValueError(f"Bad test line at line {line_no}: {line!r}")
                items.append(int(parts[0]))

            groups.append((user, items))
    return groups


def write_result(path: str, test_groups: Sequence[TestGroup], predictor, round_int: bool = True) -> None:
    """Write predictions using the same grouping format as test.txt / ResultForm.txt."""
    with open(path, "w", encoding="utf-8") as f:
        for user, items in test_groups:
            f.write(f"{user}|{len(items)}\n")
            for item in items:
                pred = predictor.predict(user, item)
                pred = clip_score(pred)
                if round_int:
                    f.write(f"{item} {int(round(pred))}\n")
                else:
                    f.write(f"{item} {pred:.6f}\n")


# =================
# 2. Basic utilities
# =================

def clip_score(x: float, low: float = 10.0, high: float = 100.0) -> float:
    """Clip predicted rating to the observed rating range."""
    if math.isnan(x) or math.isinf(x):
        return 70.0
    return min(high, max(low, x))


def rmse(predictor, ratings: Sequence[Rating]) -> float:
    """Root Mean Square Error."""
    if not ratings:
        return float("nan")
    se = 0.0
    for u, i, r in ratings:
        p = clip_score(predictor.predict(u, i))
        se += (p - r) ** 2
    return math.sqrt(se / len(ratings))


def basic_stats(ratings: Sequence[Rating], test_groups: Sequence[TestGroup]) -> Dict[str, float]:
    users = {u for u, _, _ in ratings}
    items = {i for _, i, _ in ratings}
    scores = [r for _, _, r in ratings]

    test_pairs = [(u, i) for u, items_ in test_groups for i in items_]
    test_users = {u for u, _ in test_pairs}
    test_items = {i for _, i in test_pairs}

    stats = {
        "train_ratings": len(ratings),
        "train_users": len(users),
        "train_items": len(items),
        "test_pairs": len(test_pairs),
        "test_users": len(test_users),
        "test_items": len(test_items),
        "cold_test_users": len(test_users - users),
        "cold_test_items": len(test_items - items),
        "rating_min": min(scores),
        "rating_max": max(scores),
        "rating_mean": sum(scores) / len(scores),
        "rating_sparsity": 1.0 - len(ratings) / (max(1, len(users) * len(items))),
    }
    return stats


def make_validation_split(
    ratings: Sequence[Rating],
    valid_ratio: float = 0.2,
    seed: int = 42,
    min_user_ratings: int = 5,
) -> Tuple[List[Rating], List[Rating]]:

    rng = random.Random(seed)
    by_user: Dict[int, List[Rating]] = defaultdict(list)
    for row in ratings:
        by_user[row[0]].append(row)

    train: List[Rating] = []
    valid: List[Rating] = []

    for user, rows in by_user.items():
        rows = rows[:]
        rng.shuffle(rows)

        if len(rows) >= min_user_ratings:
            k = max(1, int(round(len(rows) * valid_ratio)))
            valid.extend(rows[:k])
            train.extend(rows[k:])
        else:
            train.extend(rows)

    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


# ======================================
# 3. Model A: mean fallback recommender
# ======================================

@dataclass
class MeanModel:
    global_mean: float
    user_mean: Dict[int, float]
    item_mean: Dict[int, float]
    alpha_user: float = 10.0
    alpha_item: float = 20.0

    @classmethod
    def fit(cls, ratings: Sequence[Rating]) -> "MeanModel":
        user_sum = defaultdict(float)
        user_cnt = defaultdict(int)
        item_sum = defaultdict(float)
        item_cnt = defaultdict(int)

        total = 0.0
        for u, i, r in ratings:
            total += r
            user_sum[u] += r
            user_cnt[u] += 1
            item_sum[i] += r
            item_cnt[i] += 1

        global_mean = total / max(1, len(ratings))

        # Shrink small-count means toward global mean.
        user_mean = {
            u: (user_sum[u] + cls.alpha_user * global_mean) / (user_cnt[u] + cls.alpha_user)
            for u in user_sum
        }
        item_mean = {
            i: (item_sum[i] + cls.alpha_item * global_mean) / (item_cnt[i] + cls.alpha_item)
            for i in item_sum
        }
        return cls(global_mean=global_mean, user_mean=user_mean, item_mean=item_mean)

    def predict(self, user: int, item: int) -> float:
        u_known = user in self.user_mean
        i_known = item in self.item_mean

        if u_known and i_known:
            # Conservative ensemble: item mean usually helps, user mean personalizes.
            return 0.50 * self.global_mean + 0.25 * self.user_mean[user] + 0.25 * self.item_mean[item]
        if i_known:
            return 0.35 * self.global_mean + 0.65 * self.item_mean[item]
        if u_known:
            return 0.35 * self.global_mean + 0.65 * self.user_mean[user]
        return self.global_mean


# ====================================================
# 4. Model B: baseline bias model, mu + b_u + b_i
# ====================================================

@dataclass
class BaselineBiasModel:
    mu: float
    bu: Dict[int, float]
    bi: Dict[int, float]
    fallback: MeanModel

    @classmethod
    def fit(
        cls,
        ratings: Sequence[Rating],
        epochs: int = 25,
        lr: float = 0.006,
        reg: float = 0.08,
        seed: int = 42,
        verbose: bool = False,
    ) -> "BaselineBiasModel":
        rng = random.Random(seed)
        fallback = MeanModel.fit(ratings)
        mu = fallback.global_mean
        bu: Dict[int, float] = defaultdict(float)
        bi: Dict[int, float] = defaultdict(float)

        data = list(ratings)
        for ep in range(1, epochs + 1):
            rng.shuffle(data)
            for u, i, r in data:
                pred = mu + bu[u] + bi[i]
                err = r - pred

                bu[u] += lr * (err - reg * bu[u])
                bi[i] += lr * (err - reg * bi[i])

            if verbose and (ep == 1 or ep % 5 == 0 or ep == epochs):
                print(f"[baseline] epoch={ep:02d}")

        return cls(mu=mu, bu=dict(bu), bi=dict(bi), fallback=fallback)

    def predict(self, user: int, item: int) -> float:
        if user in self.bu or item in self.bi:
            return self.mu + self.bu.get(user, 0.0) + self.bi.get(item, 0.0)
        return self.fallback.predict(user, item)


# =======================================================
# 5. Model C: biased matrix factorization, SGD optimizer
# =======================================================

@dataclass
class MatrixFactorizationModel:
    mu: float
    user_to_idx: Dict[int, int]
    item_to_idx: Dict[int, int]
    P: np.ndarray
    Q: np.ndarray
    bu: np.ndarray
    bi: np.ndarray
    fallback: BaselineBiasModel

    @classmethod
    def fit(
        cls,
        ratings: Sequence[Rating],
        factors: int = 32,
        epochs: int = 30,
        lr: float = 0.004,
        reg: float = 0.06,
        reg_bias: float = 0.04,
        seed: int = 42,
        verbose: bool = False,
    ) -> "MatrixFactorizationModel":
        # Baseline fallback is important for cold-start users/items.
        fallback = BaselineBiasModel.fit(
            ratings,
            epochs=max(8, epochs // 3),
            lr=0.006,
            reg=0.08,
            seed=seed,
            verbose=False,
        )
        mu = fallback.mu

        users = sorted({u for u, _, _ in ratings})
        items = sorted({i for _, i, _ in ratings})
        user_to_idx = {u: idx for idx, u in enumerate(users)}
        item_to_idx = {i: idx for idx, i in enumerate(items)}

        rng_np = np.random.default_rng(seed)
        rng_py = random.Random(seed)

        # Small random values avoid exploding dot products.
        P = rng_np.normal(0.0, 0.08, size=(len(users), factors)).astype(np.float64)
        Q = rng_np.normal(0.0, 0.08, size=(len(items), factors)).astype(np.float64)

        bu = np.zeros(len(users), dtype=np.float64)
        bi = np.zeros(len(items), dtype=np.float64)

        # Initialize biases from the baseline model when available.
        for u, idx in user_to_idx.items():
            bu[idx] = fallback.bu.get(u, 0.0)
        for i, idx in item_to_idx.items():
            bi[idx] = fallback.bi.get(i, 0.0)

        data = [(user_to_idx[u], item_to_idx[i], r) for u, i, r in ratings]

        for ep in range(1, epochs + 1):
            rng_py.shuffle(data)

            # Mild learning rate decay improves stability.
            cur_lr = lr / (1.0 + 0.03 * (ep - 1))

            for uidx, iidx, r in data:
                pu = P[uidx]
                qi = Q[iidx]

                pred = mu + bu[uidx] + bi[iidx] + float(np.dot(pu, qi))
                err = r - pred

                bu[uidx] += cur_lr * (err - reg_bias * bu[uidx])
                bi[iidx] += cur_lr * (err - reg_bias * bi[iidx])

                # Copy pu because Q update must use old P value.
                old_pu = pu.copy()
                P[uidx] += cur_lr * (err * qi - reg * pu)
                Q[iidx] += cur_lr * (err * old_pu - reg * qi)

            if verbose and (ep == 1 or ep % 5 == 0 or ep == epochs):
                print(f"[mf] epoch={ep:02d}")

        return cls(
            mu=mu,
            user_to_idx=user_to_idx,
            item_to_idx=item_to_idx,
            P=P,
            Q=Q,
            bu=bu,
            bi=bi,
            fallback=fallback,
        )

    def predict(self, user: int, item: int) -> float:
        uidx = self.user_to_idx.get(user)
        iidx = self.item_to_idx.get(item)

        if uidx is None or iidx is None:
            return self.fallback.predict(user, item)

        return (
            self.mu
            + self.bu[uidx]
            + self.bi[iidx]
            + float(np.dot(self.P[uidx], self.Q[iidx]))
        )


# ======================
# 6. Training controller
# ======================

def fit_selected_model(
    model_name: str,
    train_ratings: Sequence[Rating],
    args: argparse.Namespace,
    verbose: bool = False,
):
    if model_name == "mean":
        return MeanModel.fit(train_ratings)
    if model_name == "baseline":
        return BaselineBiasModel.fit(
            train_ratings,
            epochs=args.baseline_epochs,
            lr=args.baseline_lr,
            reg=args.baseline_reg,
            seed=args.seed,
            verbose=verbose,
        )
    if model_name == "mf":
        return MatrixFactorizationModel.fit(
            train_ratings,
            factors=args.factors,
            epochs=args.mf_epochs,
            lr=args.mf_lr,
            reg=args.mf_reg,
            reg_bias=args.mf_reg_bias,
            seed=args.seed,
            verbose=verbose,
        )
    raise ValueError(f"Unknown model: {model_name}")


def choose_model_by_validation(ratings: Sequence[Rating], args: argparse.Namespace) -> str:
    train_part, valid_part = make_validation_split(
        ratings,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
        min_user_ratings=5,
    )

    print("\nValidation split")
    print(f"  train ratings: {len(train_part)}")
    print(f"  valid ratings: {len(valid_part)}")

    candidates = ["mean", "baseline", "mf"]
    results = []

    for name in candidates:
        t0 = time.perf_counter()
        model = fit_selected_model(name, train_part, args, verbose=False)
        elapsed = time.perf_counter() - t0
        score = rmse(model, valid_part)
        results.append((score, elapsed, name))
        print(f"  {name:8s} RMSE={score:.6f}, train_time={elapsed:.2f}s")

    results.sort()
    best_score, best_time, best_name = results[0]
    print(f"\nSelected model: {best_name}  validation_RMSE={best_score:.6f}")
    return best_name


def print_stats(stats: Dict[str, float]) -> None:
    print("Dataset statistics")
    for key in [
        "train_ratings",
        "train_users",
        "train_items",
        "test_pairs",
        "test_users",
        "test_items",
        "cold_test_users",
        "cold_test_items",
        "rating_min",
        "rating_max",
        "rating_mean",
        "rating_sparsity",
    ]:
        value = stats[key]
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recommendation rating prediction for train.txt/test.txt"
    )
    parser.add_argument("--train", default="train.txt", help="Path to train.txt")
    parser.add_argument("--test", default="test.txt", help="Path to test.txt")
    parser.add_argument("--output", default="Result.txt", help="Output result file")
    parser.add_argument(
        "--model",
        default="auto",
        choices=["auto", "mean", "baseline", "mf"],
        help="Model to use. auto compares models on validation set.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument(
        "--no_round",
        action="store_true",
        help="Output decimal predictions instead of rounded integer scores.",
    )

    # Baseline hyperparameters
    parser.add_argument("--baseline_epochs", type=int, default=20)
    parser.add_argument("--baseline_lr", type=float, default=0.006)
    parser.add_argument("--baseline_reg", type=float, default=0.08)

    # MF hyperparameters
    parser.add_argument("--factors", type=int, default=16)
    parser.add_argument("--mf_epochs", type=int, default=12)
    parser.add_argument("--mf_lr", type=float, default=0.004)
    parser.add_argument("--mf_reg", type=float, default=0.06)
    parser.add_argument("--mf_reg_bias", type=float, default=0.04)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    total_start = time.perf_counter()

    ratings = read_train(args.train)
    test_groups = read_test_groups(args.test)

    stats = basic_stats(ratings, test_groups)
    print_stats(stats)

    if args.model == "auto":
        selected_model_name = choose_model_by_validation(ratings, args)
    else:
        selected_model_name = args.model

    print(f"\nTraining final model on full train set: {selected_model_name}")
    train_start = time.perf_counter()
    model = fit_selected_model(selected_model_name, ratings, args, verbose=True)
    train_elapsed = time.perf_counter() - train_start

    print(f"Writing result to: {args.output}")
    write_result(args.output, test_groups, model, round_int=(not args.no_round))

    total_elapsed = time.perf_counter() - total_start
    print("\nRun summary")
    print(f"  final_model: {selected_model_name}")
    print(f"  final_training_time: {train_elapsed:.2f}s")
    print(f"  total_time: {total_elapsed:.2f}s")
    try:
        import resource
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux returns KB; macOS returns bytes. This environment is Linux.
        print(f"  peak_memory: {peak_kb / 1024:.2f} MB")
    except Exception:
        print("  peak_memory: unavailable")
    print("  done")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
