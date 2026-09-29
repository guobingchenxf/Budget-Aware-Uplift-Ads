"""分时段投放节奏（pacing）模拟。

⚠️ 声明：Hillstrom 数据**没有时间维度**。本模块把用户随机分配到若干时段，
并引入"时段价值系数"，用于研究**投放节奏**（何时花钱）这一研究问题。
时段划分、日内曲线、价值系数全部是**模拟设定**，不是观测事实。

设计要点（与第三讲的 pacing 理论一致）：
- 把预算在时间维度上分配，目标不是"消耗均匀"，而是"把预算花在价值更高的时段"；
- 三种策略：
    no_pacing  : 每时段能花多少花多少 → 预算被早期低价值时段吃掉
    uniform    : 每时段固定额度 → 与真实流量/价值曲线不匹配
    feedback   : 比例反馈控制器（阻尼 η），跟踪累计消耗目标 —— 对应生产实践
 外加一个 oracle_value_aware：已知各时段真实价值系数（**模拟上界，不可实现**），
 用来量化"节奏优化"的理论空间。
- 目标人群选择与节奏解耦：每个时段内部按策略分数从高到低选人，
  因此本模块只回答"何时花"，不回答"花给谁"。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def value_profile(n_slots: int, phase: float = 0.75, width: float = 0.18,
                  peak_boost: float = 1.8) -> np.ndarray:
    """时段价值系数：以 phase 为中心的高斯"黄金时段"。

    phase 为峰值位置占全天的比例（0.75 ≈ 晚 18:00 附近）。
    返回长度 n_slots、均值约 1 的正系数。
    """
    s = np.linspace(0.0, 1.0, n_slots, endpoint=False)
    bump = np.exp(-0.5 * ((s - phase) / width) ** 2)
    v = 1.0 + (peak_boost - 1.0) * bump
    return v / v.mean()


def supply_profile(n_slots: int, phase: float = 0.75, width: float = 0.22,
                   concentration: float = 0.0) -> np.ndarray:
    """时段流量供给形状（归一化到和为 1）。

    concentration ∈ [0, 1]：
      0   → 平滑的日内曲线（供给充足，任何时段都能花钱）
      1   → 高度集中在少数时段（早时段供给稀缺，容易出现"想花也花不出去"）
    这一参数是研究"供给受限时 pacing 是否重要"的关键开关。
    """
    s = np.linspace(0.0, 1.0, n_slots, endpoint=False)
    shape = 1.0 + 0.8 * np.exp(-0.5 * ((s - phase) / width) ** 2)
    if concentration > 0:
        # 把形状抬到高次幂再混合，使流量向峰值时段集中
        p = 1.0 + 9.0 * float(concentration)
        concentrated = shape ** p
        shape = (1 - concentration) * shape + concentration * concentrated
    return shape / shape.sum()


def assign_slots(n_users: int, supply: np.ndarray, seed: int = 0) -> np.ndarray:
    """按供给形状把用户随机分配到各时段（多项分布）。"""
    rng = np.random.default_rng(seed)
    probs = supply / supply.sum()
    return rng.choice(len(supply), size=n_users, p=probs)


def _allowance_units(strategy: str, slot: int, n_slots: int, budget_units: int,
                     spent_units: int, damping: float) -> int:
    """该时段允许投放的人数上限（策略核心逻辑）。"""
    remaining = max(0, budget_units - spent_units)
    base = budget_units / float(n_slots)

    if strategy == "no_pacing":
        return remaining

    if strategy == "uniform":
        return int(min(remaining, np.floor(base)))

    if strategy == "feedback":
        # 目标口径：进入本时段之前应已花掉的量 = budget * slot / n_slots
        # （用 slot 而非 slot+1，否则控制器在第 0 个时段就"自认为落后"，
        #   会系统性地超前消费——这是实现时容易犯的差一错误）
        target_cum = budget_units * slot / float(n_slots)
        error = target_cum - spent_units
        allow = base + damping * error
        return int(min(remaining, max(0.0, np.floor(allow))))

    raise ValueError(f"未知节奏策略: {strategy}")


def simulate_pacing(
    score: np.ndarray,
    value: np.ndarray,
    n_slots: int = 24,
    budget_ratio: float = 0.05,
    cost_per_treatment: float = 1.0,
    strategy: str = "feedback",
    damping: float = 0.3,
    peak_boost: float = 1.8,
    supply_concentration: float = 0.0,
    seed: int = 0,
) -> Dict[str, object]:
    """在分时段设定下模拟一种节奏策略。

    参数
    ----
    score : 每个候选的排序分（时段内按此分数从高到低选人）
    value : 每个候选被投放后的**期望增量结果**（用于模拟核算）。
            在半合成数据上取真实 CATE；在真实数据上取模型预测 uplift，
            此时结果是"模型驱动的模拟"，必须在报告中标注。
    supply_concentration : 供给集中度，见 supply_profile。

    策略说明
    --------
    oracle_value_aware 的实现是**真正的边际最优**：
    把所有 (用户, 时段) 组合按 `value_i × v_slot` 降序排列后取前 k 个。
    早期版本按"时段价值"贪心会输给均匀投放（因为它忽略了同一时段内
    用户质量的边际递减），那是一个错误的"上界"，已修正。
    """
    score = np.asarray(score, dtype=float)
    value = np.asarray(value, dtype=float)
    n = len(score)
    if len(value) != n:
        raise ValueError("score 与 value 长度不一致")

    v_slot = value_profile(n_slots, peak_boost=peak_boost)
    supply = supply_profile(n_slots, concentration=supply_concentration)
    slot_of = assign_slots(n, supply, seed=seed)

    budget_units = int(np.floor(n * budget_ratio))
    rows = []

    if strategy == "oracle_value_aware":
        # 边际最优：value_i * v_slot 的全局 top-k（先按 value 降序，等价于组内取前若干）
        marg = value * v_slot[slot_of]
        order = np.argsort(-marg, kind="mergesort")[:budget_units]
        chosen_flag = np.zeros(n, dtype=bool)
        chosen_flag[order] = True
        spent_units = int(chosen_flag.sum())
        total_gain = float((value[chosen_flag] * v_slot[slot_of[chosen_flag]]).sum())
        for s in range(n_slots):
            m = slot_of == s
            picked = m & chosen_flag
            rows.append({"slot": s, "supply": int(m.sum()), "treated": int(picked.sum()),
                         "spend": float(picked.sum() * cost_per_treatment),
                         "gain": float((value[picked] * v_slot[s]).sum()),
                         "slot_value": float(v_slot[s]),
                         "cum_spend": float(chosen_flag[:].sum() * 0.0)})
        table = pd.DataFrame(rows).sort_values("slot").reset_index(drop=True)
        table["cum_spend"] = (table["treated"] * cost_per_treatment).cumsum()
        cost = spent_units * cost_per_treatment
        return {
            "strategy": strategy, "n_candidates": n, "budget_units": budget_units,
            "treated": spent_units, "spend": float(cost),
            "budget": float(budget_units * cost_per_treatment),
            "unspent": float((budget_units - spent_units) * cost_per_treatment),
            "gain": total_gain,
            "gain_per_cost": float(total_gain / cost) if cost > 0 else float("nan"),
            "slots": table,
        }

    spent_local = 0
    total_gain = 0.0
    for s in range(n_slots):
        users = np.flatnonzero(slot_of == s)
        if users.size == 0:
            rows.append({"slot": s, "supply": 0, "treated": 0, "spend": 0.0,
                         "gain": 0.0, "slot_value": float(v_slot[s]),
                         "cum_spend": float(spent_local * cost_per_treatment)})
            continue

        order_in_slot = users[np.argsort(-score[users], kind="mergesort")]
        allow = _allowance_units(strategy, s, n_slots, budget_units, spent_local, damping)
        take = int(min(users.size, max(0, allow), budget_units - spent_local))
        chosen = order_in_slot[:take]

        gain_s = float((value[chosen] * v_slot[s]).sum())
        total_gain += gain_s
        spent_local += take

        rows.append({"slot": s, "supply": int(users.size), "treated": take,
                     "spend": take * cost_per_treatment, "gain": gain_s,
                     "slot_value": float(v_slot[s]),
                     "cum_spend": float(spent_local * cost_per_treatment)})

    table = pd.DataFrame(rows).sort_values("slot").reset_index(drop=True)
    cost = spent_local * cost_per_treatment
    return {
        "strategy": strategy,
        "n_candidates": n,
        "budget_units": budget_units,
        "treated": int(spent_local),
        "spend": float(cost),
        "budget": float(budget_units * cost_per_treatment),
        "unspent": float((budget_units - spent_local) * cost_per_treatment),
        "gain": float(total_gain),
        "gain_per_cost": float(total_gain / cost) if cost > 0 else float("nan"),
        "slots": table,
    }


def compare_pacing_strategies(
    score: np.ndarray,
    value: np.ndarray,
    strategies,
    n_slots: int = 24,
    budget_ratio: float = 0.05,
    cost_per_treatment: float = 1.0,
    damping: float = 0.3,
    peak_boost: float = 1.8,
    supply_concentration: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    rows = []
    for st in strategies:
        res = simulate_pacing(score, value, n_slots, budget_ratio, cost_per_treatment,
                              st, damping, peak_boost, supply_concentration, seed)
        rows.append({k: v for k, v in res.items() if k != "slots"})
    out = pd.DataFrame(rows)
    if len(out):
        best = float(out["gain"].max())
        out["gain_vs_best"] = out["gain"] / best if abs(best) > 1e-12 else np.nan
    return out
