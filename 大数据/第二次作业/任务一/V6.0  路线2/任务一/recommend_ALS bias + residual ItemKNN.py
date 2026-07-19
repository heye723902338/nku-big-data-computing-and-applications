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
    user_history: Dict[int, List[Tuple[int, float]]]
    item_neighbors: Dict[int, List[Tuple[int, float]]]
    knn_shrink: float = 30.0

    def _base_predict(self, user: int, item: int) -> float:
        if user in self.bu or item in self.bi:
            return self.mu + self.bu.get(user, 0.0) + self.bi.get(item, 0.0)
        return self.fallback.predict(user, item)

    @staticmethod
    def _build_history(ratings: Sequence[Rating]) -> Tuple[Dict[int, List[Tuple[int, float]]], Dict[int, List[Tuple[int, float]]]]:
        user_history: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        item_history: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        for user, item, score in ratings:
            user_history[user].append((item, score))
            item_history[item].append((user, score))
        return user_history, item_history

    @classmethod
    def _fit_als_bias(
        cls,
        ratings: Sequence[Rating],
        epochs: int,
        reg_user: float,
        reg_item: float,
        seed: int,
        verbose: bool,
    ) -> Tuple[float, Dict[int, float], Dict[int, float], MeanModel]:
        fallback = MeanModel.fit(ratings)
        mu = fallback.global_mean
        user_history, item_history = cls._build_history(ratings)
        users = sorted(user_history)
        items = sorted(item_history)
        bu: Dict[int, float] = {user: 0.0 for user in users}
        bi: Dict[int, float] = {item: 0.0 for item in items}
        rng = random.Random(seed)

        for ep in range(1, epochs + 1):
            if len(users) > 1:
                rng.shuffle(users)
            if len(items) > 1:
                rng.shuffle(items)

            for user in users:
                rows = user_history[user]
                denom = reg_user + len(rows)
                numer = 0.0
                for item, score in rows:
                    numer += score - mu - bi.get(item, 0.0)
                bu[user] = numer / denom if denom else 0.0

            for item in items:
                rows = item_history[item]
                denom = reg_item + len(rows)
                numer = 0.0
                for user, score in rows:
                    numer += score - mu - bu.get(user, 0.0)
                bi[item] = numer / denom if denom else 0.0

            if verbose and (ep == 1 or ep % 5 == 0 or ep == epochs):
                print(f"[als-bias] epoch={ep:02d}")

        return mu, bu, bi, fallback

    @classmethod
    def _fit_item_knn(
        cls,
        ratings: Sequence[Rating],
        base_model: "BaselineBiasModel",
        k: int,
        min_overlap: int,
        similarity_shrink: float,
    ) -> Dict[int, List[Tuple[int, float]]]:
        user_history, _ = cls._build_history(ratings)
        item_norm: Dict[int, float] = defaultdict(float)
        pair_dot: Dict[Tuple[int, int], float] = defaultdict(float)
        pair_cnt: Dict[Tuple[int, int], int] = defaultdict(int)

        for user, rows in user_history.items():
            residuals: List[Tuple[int, float]] = []
            for item, score in rows:
                residuals.append((item, score - base_model._base_predict(user, item)))
            for item, residual in residuals:
                item_norm[item] += residual * residual
            for idx in range(len(residuals)):
                item_i, resid_i = residuals[idx]
                for jdx in range(idx + 1, len(residuals)):
                    item_j, resid_j = residuals[jdx]
                    if item_i < item_j:
                        pair = (item_i, item_j)
                    else:
                        pair = (item_j, item_i)
                    pair_dot[pair] += resid_i * resid_j
                    pair_cnt[pair] += 1

        neighbor_scores: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        for (item_i, item_j), dot in pair_dot.items():
            overlap = pair_cnt[(item_i, item_j)]
            if overlap < min_overlap:
                continue
            norm_i = item_norm.get(item_i, 0.0)
            norm_j = item_norm.get(item_j, 0.0)
            if norm_i <= 0.0 or norm_j <= 0.0:
                continue
            sim = dot / math.sqrt(norm_i * norm_j)
            if similarity_shrink > 0.0:
                sim *= overlap / (overlap + similarity_shrink)
            if sim == 0.0 or math.isnan(sim) or math.isinf(sim):
                continue
            neighbor_scores[item_i].append((item_j, sim))
            neighbor_scores[item_j].append((item_i, sim))

        neighbors: Dict[int, List[Tuple[int, float]]] = {}
        for item, scores in neighbor_scores.items():
            scores.sort(key=lambda pair: abs(pair[1]), reverse=True)
            neighbors[item] = scores[:k]
        return neighbors

    @classmethod
    def fit(
        cls,
        ratings: Sequence[Rating],
        epochs: int = 24,
        reg_user: float = 10.0,
        reg_item: float = 15.0,
        knn_k: int = 60,
        knn_min_overlap: int = 2,
        knn_similarity_shrink: float = 25.0,
        knn_predict_shrink: float = 30.0,
        seed: int = 42,
        verbose: bool = False,
    ) -> "BaselineBiasModel":
        mu, bu, bi, fallback = cls._fit_als_bias(
            ratings,
            epochs=epochs,
            reg_user=reg_user,
            reg_item=reg_item,
            seed=seed,
            verbose=verbose,
        )
        base_model = cls(
            mu=mu,
            bu=dict(bu),
            bi=dict(bi),
            fallback=fallback,
            user_history={},
            item_neighbors={},
            knn_shrink=knn_predict_shrink,
        )
        user_history, _ = cls._build_history(ratings)
        item_neighbors = cls._fit_item_knn(
            ratings,
            base_model=base_model,
            k=knn_k,
            min_overlap=knn_min_overlap,
            similarity_shrink=knn_similarity_shrink,
        )
        return cls(
            mu=mu,
            bu=dict(bu),
            bi=dict(bi),
            fallback=fallback,
            user_history={user: rows[:] for user, rows in user_history.items()},
            item_neighbors=item_neighbors,
            knn_shrink=knn_predict_shrink,
        )

    def predict(self, user: int, item: int) -> float:
        base = self._base_predict(user, item)
        history = self.user_history.get(user)
        neighbors = self.item_neighbors.get(item)
        if not history or not neighbors:
            return base

        neighbor_map = {neighbor_item: similarity for neighbor_item, similarity in neighbors}
        numer = 0.0
        denom = 0.0
        matched = 0
        for history_item, score in history:
            similarity = neighbor_map.get(history_item)
            if similarity is None:
                continue
            history_residual = score - self._base_predict(user, history_item)
            numer += similarity * history_residual
            denom += abs(similarity)
            matched += 1

        if matched == 0 or denom <= 0.0:
            return base
        return base + numer / (denom + self.knn_shrink)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baseline bias recommender for train.txt/test.txt")
    parser.add_argument("--train", default=None, help="Path to train.txt. Auto-detected when omitted.")
    parser.add_argument("--test", default=None, help="Path to test.txt. Auto-detected when omitted.")
    parser.add_argument("--output", default=None, help="Output file. Defaults to Result.txt beside this script.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--no_round", action="store_true", help="Output decimal predictions instead of rounded integers.")
    parser.add_argument("--baseline_epochs", type=int, default=24)
    parser.add_argument("--baseline_reg_user", type=float, default=10.0)
    parser.add_argument("--baseline_reg_item", type=float, default=15.0)
    parser.add_argument("--knn_k", type=int, default=60)
    parser.add_argument("--knn_min_overlap", type=int, default=2)
    parser.add_argument("--knn_similarity_shrink", type=float, default=25.0)
    parser.add_argument("--knn_predict_shrink", type=float, default=30.0)
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
    valid_model = BaselineBiasModel.fit(
        train_part,
        epochs=args.baseline_epochs,
        reg_user=args.baseline_reg_user,
        reg_item=args.baseline_reg_item,
        knn_k=args.knn_k,
        knn_min_overlap=args.knn_min_overlap,
        knn_similarity_shrink=args.knn_similarity_shrink,
        knn_predict_shrink=args.knn_predict_shrink,
        seed=args.seed,
        verbose=False,
    )
    print("\nValidation split")
    print(f"  train ratings: {len(train_part)}")
    print(f"  valid ratings: {len(valid_part)}")
    print(f"  baseline RMSE={rmse(valid_model, valid_part):.6f}, train_time={time.perf_counter() - valid_start:.2f}s")

    print("\nTraining final baseline model on full train set")
    train_start = time.perf_counter()
    model = BaselineBiasModel.fit(
        ratings,
        epochs=args.baseline_epochs,
        reg_user=args.baseline_reg_user,
        reg_item=args.baseline_reg_item,
        knn_k=args.knn_k,
        knn_min_overlap=args.knn_min_overlap,
        knn_similarity_shrink=args.knn_similarity_shrink,
        knn_predict_shrink=args.knn_predict_shrink,
        seed=args.seed,
        verbose=True,
    )
    train_elapsed = time.perf_counter() - train_start
    write_result(output_path, test_groups, model, round_int=(not args.no_round))

    print("\nRun summary")
    print("  final_model: baseline")
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
