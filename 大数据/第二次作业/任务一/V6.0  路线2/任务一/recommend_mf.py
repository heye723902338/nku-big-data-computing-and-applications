from __future__ import annotations

import argparse
import ctypes
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


Rating = Tuple[int, int, float]
TestGroup = Tuple[int, List[int]]


def resolve_input_path(value: str | None, filename: str) -> str:
    if value:
        path = Path(value)
        if path.exists():
            return str(path)
        raise FileNotFoundError(f"Cannot find {filename}: {value}")
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()
    candidates = [cwd / filename, cwd / "data" / filename, script_dir / filename, script_dir / "data" / filename]
    for path in candidates:
        if path.exists():
            return str(path)
    raise FileNotFoundError(f"Cannot auto-detect {filename}. Tried: " + ", ".join(str(p) for p in candidates))


def resolve_output_path(value: str | None) -> str:
    return str(Path(value)) if value else str(Path(__file__).resolve().parent / "Result.txt")


def read_train(path: str) -> List[Rating]:
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
            for _ in range(int(n_str)):
                line = f.readline()
                line_no += 1
                if not line:
                    raise ValueError(f"Unexpected EOF after header {header!r}")
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(f"Bad rating line at line {line_no}: {line!r}")
                ratings.append((user, int(parts[0]), float(parts[1])))
    return ratings


def read_test_groups(path: str) -> List[TestGroup]:
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
            items: List[int] = []
            for _ in range(int(n_str)):
                line = f.readline()
                line_no += 1
                if not line:
                    raise ValueError(f"Unexpected EOF after header {header!r}")
                parts = line.split()
                if not parts:
                    raise ValueError(f"Bad test line at line {line_no}: {line!r}")
                items.append(int(parts[0]))
            groups.append((user, items))
    return groups


def clip_score(x: float, low: float = 10.0, high: float = 100.0) -> float:
    if math.isnan(x) or math.isinf(x):
        return 70.0
    return min(high, max(low, x))


def write_result(path: str, test_groups: Sequence[TestGroup], predictor, round_int: bool = True) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for user, items in test_groups:
            f.write(f"{user}|{len(items)}\n")
            for item in items:
                pred = clip_score(predictor.predict(user, item))
                f.write(f"{item} {int(round(pred)) if round_int else f'{pred:.6f}'}\n")


def rmse(predictor, ratings: Sequence[Rating]) -> float:
    if not ratings:
        return float("nan")
    se = 0.0
    for user, item, score in ratings:
        pred = clip_score(predictor.predict(user, item))
        se += (pred - score) ** 2
    return math.sqrt(se / len(ratings))


def make_validation_split(
    ratings: Sequence[Rating],
    valid_ratio: float = 0.1,
    seed: int = 42,
    min_user_ratings: int = 5,
) -> Tuple[List[Rating], List[Rating]]:
    rng = random.Random(seed)
    by_user: Dict[int, List[Rating]] = defaultdict(list)
    for row in ratings:
        by_user[row[0]].append(row)
    train: List[Rating] = []
    valid: List[Rating] = []
    for rows in by_user.values():
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


def basic_stats(ratings: Sequence[Rating], test_groups: Sequence[TestGroup]) -> Dict[str, float]:
    users = {u for u, _, _ in ratings}
    items = {i for _, i, _ in ratings}
    scores = [r for _, _, r in ratings]
    test_pairs = [(u, i) for u, items_ in test_groups for i in items_]
    test_users = {u for u, _ in test_pairs}
    test_items = {i for _, i in test_pairs}
    return {
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
        "rating_sparsity": 1.0 - len(ratings) / max(1, len(users) * len(items)),
    }


def print_stats(stats: Dict[str, float]) -> None:
    print("Dataset statistics")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")


def get_peak_memory_mb() -> float | None:
    if os.name != "nt":
        return None
    try:
        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]
        process = ctypes.windll.kernel32.GetCurrentProcess()
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), ctypes.c_ulong]
        get_process_memory_info.restype = ctypes.c_int
        if not get_process_memory_info(process, ctypes.byref(counters), counters.cb):
            return None
        return max(counters.PeakWorkingSetSize, counters.PeakPagefileUsage, counters.PrivateUsage) / (1024.0 * 1024.0)
    except Exception:
        return None


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
        for user, item, score in ratings:
            total += score
            user_sum[user] += score
            user_cnt[user] += 1
            item_sum[item] += score
            item_cnt[item] += 1
        global_mean = total / max(1, len(ratings))
        user_mean = {u: (user_sum[u] + cls.alpha_user * global_mean) / (user_cnt[u] + cls.alpha_user) for u in user_sum}
        item_mean = {i: (item_sum[i] + cls.alpha_item * global_mean) / (item_cnt[i] + cls.alpha_item) for i in item_sum}
        return cls(global_mean=global_mean, user_mean=user_mean, item_mean=item_mean)

    def predict(self, user: int, item: int) -> float:
        u_known = user in self.user_mean
        i_known = item in self.item_mean
        if u_known and i_known:
            return 0.50 * self.global_mean + 0.25 * self.user_mean[user] + 0.25 * self.item_mean[item]
        if i_known:
            return 0.35 * self.global_mean + 0.65 * self.item_mean[item]
        if u_known:
            return 0.35 * self.global_mean + 0.65 * self.user_mean[user]
        return self.global_mean


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
            for user, item, score in data:
                pred = mu + bu[user] + bi[item]
                err = score - pred
                bu[user] += lr * (err - reg * bu[user])
                bi[item] += lr * (err - reg * bi[item])
            if verbose and (ep == 1 or ep % 5 == 0 or ep == epochs):
                print(f"[baseline] epoch={ep:02d}")
        return cls(mu=mu, bu=dict(bu), bi=dict(bi), fallback=fallback)

    def predict(self, user: int, item: int) -> float:
        if user in self.bu or item in self.bi:
            return self.mu + self.bu.get(user, 0.0) + self.bi.get(item, 0.0)
        return self.fallback.predict(user, item)


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
        P = rng_np.normal(0.0, 0.08, size=(len(users), factors)).astype(np.float64)
        Q = rng_np.normal(0.0, 0.08, size=(len(items), factors)).astype(np.float64)
        bu = np.zeros(len(users), dtype=np.float64)
        bi = np.zeros(len(items), dtype=np.float64)
        for user, idx in user_to_idx.items():
            bu[idx] = fallback.bu.get(user, 0.0)
        for item, idx in item_to_idx.items():
            bi[idx] = fallback.bi.get(item, 0.0)
        data = [(user_to_idx[u], item_to_idx[i], r) for u, i, r in ratings]
        for ep in range(1, epochs + 1):
            rng_py.shuffle(data)
            cur_lr = lr / (1.0 + 0.03 * (ep - 1))
            for uidx, iidx, score in data:
                pu = P[uidx]
                qi = Q[iidx]
                pred = mu + bu[uidx] + bi[iidx] + float(np.dot(pu, qi))
                err = score - pred
                bu[uidx] += cur_lr * (err - reg_bias * bu[uidx])
                bi[iidx] += cur_lr * (err - reg_bias * bi[iidx])
                old_pu = pu.copy()
                P[uidx] += cur_lr * (err * qi - reg * pu)
                Q[iidx] += cur_lr * (err * old_pu - reg * qi)
            if verbose and (ep == 1 or ep % 5 == 0 or ep == epochs):
                print(f"[mf] epoch={ep:02d}")
        return cls(mu=mu, user_to_idx=user_to_idx, item_to_idx=item_to_idx, P=P, Q=Q, bu=bu, bi=bi, fallback=fallback)

    def predict(self, user: int, item: int) -> float:
        uidx = self.user_to_idx.get(user)
        iidx = self.item_to_idx.get(item)
        if uidx is None or iidx is None:
            return self.fallback.predict(user, item)
        return self.mu + self.bu[uidx] + self.bi[iidx] + float(np.dot(self.P[uidx], self.Q[iidx]))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Biased matrix factorization recommender for train.txt/test.txt")
    parser.add_argument("--train", default=None, help="Path to train.txt. Auto-detected when omitted.")
    parser.add_argument("--test", default=None, help="Path to test.txt. Auto-detected when omitted.")
    parser.add_argument("--output", default=None, help="Output file. Defaults to Result.txt beside this script.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--no_round", action="store_true", help="Output decimal predictions instead of rounded integers.")
    parser.add_argument("--factors", type=int, default=16)
    parser.add_argument("--mf_epochs", type=int, default=12)
    parser.add_argument("--mf_lr", type=float, default=0.004)
    parser.add_argument("--mf_reg", type=float, default=0.06)
    parser.add_argument("--mf_reg_bias", type=float, default=0.04)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    train_path = resolve_input_path(args.train, "train.txt")
    test_path = resolve_input_path(args.test, "test.txt")
    output_path = resolve_output_path(args.output)

    total_start = time.perf_counter()
    ratings = read_train(train_path)
    test_groups = read_test_groups(test_path)
    print(f"Using train: {train_path}")
    print(f"Using test: {test_path}")
    print_stats(basic_stats(ratings, test_groups))

    train_part, valid_part = make_validation_split(ratings, valid_ratio=args.valid_ratio, seed=args.seed)
    valid_start = time.perf_counter()
    valid_model = MatrixFactorizationModel.fit(
        train_part,
        factors=args.factors,
        epochs=args.mf_epochs,
        lr=args.mf_lr,
        reg=args.mf_reg,
        reg_bias=args.mf_reg_bias,
        seed=args.seed,
        verbose=False,
    )
    print("\nValidation split")
    print(f"  train ratings: {len(train_part)}")
    print(f"  valid ratings: {len(valid_part)}")
    print(f"  mf RMSE={rmse(valid_model, valid_part):.6f}, train_time={time.perf_counter() - valid_start:.2f}s")

    print("\nTraining final MF model on full train set")
    train_start = time.perf_counter()
    model = MatrixFactorizationModel.fit(
        ratings,
        factors=args.factors,
        epochs=args.mf_epochs,
        lr=args.mf_lr,
        reg=args.mf_reg,
        reg_bias=args.mf_reg_bias,
        seed=args.seed,
        verbose=True,
    )
    train_elapsed = time.perf_counter() - train_start
    write_result(output_path, test_groups, model, round_int=(not args.no_round))

    print("\nRun summary")
    print("  final_model: mf")
    print(f"  output: {output_path}")
    print(f"  final_training_time: {train_elapsed:.2f}s")
    print(f"  total_time: {time.perf_counter() - total_start:.2f}s")
    peak_memory_mb = get_peak_memory_mb()
    if peak_memory_mb is not None:
        print(f"  peak_memory: {peak_memory_mb:.2f} MB")
    print("  done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
