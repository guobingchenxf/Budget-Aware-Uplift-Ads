"""因果排序评估指标：Qini / AUUC / 校准 / 策略增量收益。

术语与前提（必须与文档一致）：
- 数据来自**随机实验**（treatment 随机分配），因此
  "被选中集合 S 的增量收益" 可以用 S 内部的处理组/对照组结果差来无偏估计：
      G(S) = Y_t(S) - Y_c(S) * N_t(S) / N_c(S)          （Radcliffe Qini 形式）
  其中 Y_t、N_t 分别是 S 中处理组的结果和与人数。
- 排序曲线按分数降序对全体样本做前缀，逐点计算 G(S_k)，
  得到 Qini 曲线；其与随机基线（对角直线）的归一化面积差即 Qini 系数。
- ⚠️ 已知局限（UpliftBench, 2026 指出）：Qini 这类排序指标**丢弃分数量级信息**，
  对"阈值型策略"（要不要投）并不充分；连续 outcome 下 Qini 与效应精度
  可能不相关。因此本项目同时报告校准指标与阈值策略结果，不单独依赖 Qini。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def _prefix_stats(y: np.ndarray, t: np.ndarray) -> Dict[str, np.ndarray]:
    n_t = np.cumsum(t)
    n_c = np.cumsum(1 - t)
    y_t = np.cumsum(y * t)
    y_c = np.cumsum(y * (1 - t))
    return {"n_t": n_t, "n_c": n_c, "y_t": y_t, "y_c": y_c}


def qini_curve(y, t, score) -> Dict[str, np.ndarray]:
    """按分数降序的前缀 Qini 曲线与 uplift 曲线。

    返回 dict:
      fractions : 前缀占比 (0,1]
      qini      : Radcliffe 形式 G(S_k)
      uplift    : (ȳ_t(k) - ȳ_c(k)) * k * N ，即"累计增量收益"的另一种常见尺度
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    y, t = y[order], t[order]

    s = _prefix_stats(y, t)
    n_t, n_c, y_t, y_c = s["n_t"], s["n_c"], s["y_t"], s["y_c"]

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(n_c > 0, n_t / np.maximum(n_c, 1e-12), 0.0)
    qini = y_t - y_c * ratio

    k = np.arange(1, len(y) + 1)
    mean_t = np.where(n_t > 0, y_t / np.maximum(n_t, 1e-12), 0.0)
    mean_c = np.where(n_c > 0, y_c / np.maximum(n_c, 1e-12), 0.0)
    uplift = (mean_t - mean_c) * k

    return {
        "fractions": k / len(y),
        "qini": qini,
        "uplift": uplift,
    }


def _area_vs_random(curve: np.ndarray, fractions: np.ndarray) -> float:
    """归一化面积差：曲线按其终值归一后，与 y=x 直线之间的面积。"""
    total = curve[-1]
    if not np.isfinite(total) or abs(total) < 1e-12:
        return float("nan")
    norm = curve / total
    area = float(np.trapz(norm, fractions))
    return area - 0.5


def qini_coefficient(y, t, score) -> float:
    """归一化 Qini 系数：0 = 与随机等价，越大越好，负值表示比随机更差。

    归一化方式：Qini 曲线除以终值后与对角线的面积差（随机策略 = 0）。
    这是常见做法，但不同库的归一化约定不同，跨库比较数值前需确认口径。
    """
    cur = qini_curve(y, t, score)
    return _area_vs_random(cur["qini"], cur["fractions"])


def auuc(y, t, score, normalized: bool = True) -> float:
    """AUUC：uplift 曲线下面积（相对随机基线）。

    normalized=True 时返回与 qini_coefficient 同量纲的归一化面积差；
    False 时返回原始面积（单位：结果个数），便于报告绝对规模。
    """
    cur = qini_curve(y, t, score)
    if normalized:
        return _area_vs_random(cur["uplift"], cur["fractions"])
    return float(np.trapz(cur["uplift"], cur["fractions"]))


def policy_gain(y, t, selected: np.ndarray) -> Dict[str, float]:
    """被选中集合 S 的增量收益估计（Qini 形式）与规模信息。

    G(S) = Y_t(S) - Y_c(S) * N_t(S) / N_c(S)
    在随机实验下这是 S 上增量结果总和的无偏估计（证明见文档）。
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    sel = np.asarray(selected, dtype=bool)
    if sel.sum() == 0:
        return {"n_selected": 0, "n_treated": 0, "n_control": 0,
                "y_t_sum": 0.0, "y_c_sum": 0.0, "gain": 0.0, "gain_per_user": 0.0}

    ys, ts = y[sel], t[sel]
    n_t = float(ts.sum())
    n_c = float((1 - ts).sum())
    y_t = float((ys * ts).sum())
    y_c = float((ys * (1 - ts)).sum())
    if n_c <= 0:
        gain = float("nan")
    else:
        gain = y_t - y_c * n_t / n_c
    return {
        "n_selected": int(sel.sum()),
        "n_treated": int(n_t),
        "n_control": int(n_c),
        "y_t_sum": y_t,
        "y_c_sum": y_c,
        "gain": gain,
        "gain_per_user": gain / float(sel.sum()),
    }


def select_top_k(score: np.ndarray, k: int) -> np.ndarray:
    """按分数取前 k 个（分数相同按索引稳定排序，保证可复现）。"""
    score = np.asarray(score, dtype=float)
    k = int(max(0, min(k, len(score))))
    mask = np.zeros(len(score), dtype=bool)
    if k == 0:
        return mask
    order = np.argsort(-score, kind="mergesort")[:k]
    mask[order] = True
    return mask


def calibration_report(y_true, p_pred, n_bins: int = 10) -> Dict[str, object]:
    """概率校准报告：ECE、平均绝对偏差、分桶明细。

    只对"概率型"分数（响应模型输出）有意义；uplift 分数可正可负，
    不在概率空间，故不适用（需用半合成数据的真实 CATE 另做校准评估）。
    """
    y_true = np.asarray(y_true, dtype=float)
    p_pred = np.asarray(p_pred, dtype=float)
    if len(y_true) != len(p_pred):
        raise ValueError("长度不一致")
    if len(y_true) == 0:
        return {"ece": float("nan"), "bins": []}

    bins = np.quantile(p_pred, np.linspace(0, 1, n_bins + 1))
    bins = np.unique(bins)
    rows: List[Dict[str, float]] = []
    ece = 0.0
    n = len(y_true)
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        m = (p_pred >= lo) & (p_pred <= hi) if i == len(bins) - 2 else (p_pred >= lo) & (p_pred < hi)
        cnt = int(m.sum())
        if cnt == 0:
            continue
        conf = float(p_pred[m].mean())
        acc = float(y_true[m].mean())
        rows.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": cnt,
                     "mean_pred": conf, "mean_true": acc, "gap": conf - acc})
        ece += abs(conf - acc) * cnt / n

    mae = float(np.mean([abs(r["gap"]) for r in rows])) if rows else float("nan")
    return {"ece": float(ece), "mean_abs_bin_gap": mae, "n_bins_used": len(rows), "bins": rows}


def cate_calibration(pred_cate, true_cate, n_bins: int = 10) -> Dict[str, object]:
    """半合成数据专用：预测 CATE 与真实 CATE 的校准与相关。

    真实 CATE 只在半合成数据上已知，因此这是**验证评估链路**的手段，
    不能用于真实数据。
    """
    pred_cate = np.asarray(pred_cate, dtype=float)
    true_cate = np.asarray(true_cate, dtype=float)
    if len(pred_cate) != len(true_cate):
        raise ValueError("长度不一致")
    order = np.argsort(pred_cate, kind="mergesort")
    chunks = np.array_split(order, n_bins)
    rows = []
    for ch in chunks:
        if len(ch) == 0:
            continue
        rows.append({"n": int(len(ch)),
                     "mean_pred": float(pred_cate[ch].mean()),
                     "mean_true": float(true_cate[ch].mean())})
    mae = float(np.mean([abs(r["mean_pred"] - r["mean_true"]) for r in rows])) if rows else float("nan")
    corr = float(np.corrcoef(pred_cate, true_cate)[0, 1]) if len(pred_cate) > 2 else float("nan")
    bias = float(pred_cate.mean() - true_cate.mean())
    return {"mae_binned": mae, "pearson": corr, "bias": bias, "bins": rows}
