# Budget-Aware Uplift Ads
### 预算约束下的增量转化广告排序与投放节奏优化（广告算法研究类项目）

---

## 项目说明

**研究问题**：广告预算有限时，按"预测转化概率"选人，还是按"广告带来的**增量**转化"选人，
哪种策略能带来更高的增量收益？模型误差、概率校准与预算消耗速度会如何影响结论？

**核心结论（本机实测）**：

| 数据/场景 | 响应模型（按预测转化率） | Uplift 模型（按增量） | 随机 |
|---|---|---|---|
| 半合成·aligned（高基线=高增量） | 109.9（3.52×随机） | 91～102（2.9～3.3×） | 31.2 |
| 半合成·conflicting（高基线=低增量） | **10.5** | **21～29** | −11.3 |
| Hillstrom 真实数据（5% 预算，10 种子均值） | 27.2 | 24.4～31.9 | 18.4 |

1. **当"高转化概率"与"高增量"方向一致时，响应模型几乎不输**（aligned 场景 3.52× vs 3.28×）——
   这是"uplift 建模没用"这类说法流行的原因；
2. **两者方向相反时，响应模型会系统性选错人**（conflicting 场景 10.5 vs 21～29，随机为 −11.3）；
3. **在真实数据上，模型排序确实优于随机，但两种排序方式之间分不出高下**：
   10 种子配对检验显示 s_learner / response / t_learner 显著优于随机
   （配对差 +13.5 / +8.8 / +8.2，t95 均不含 0），
   但 **s_learner − response = +4.63 [−1.55, +10.81]，不显著**。
   **核心研究假设（"按增量排序优于按转化率排序"）在 Hillstrom 上依然得不到支持。**
4. **一次真实的方法论翻车与纠正**：单种子 + 自助法曾让我得出"什么都测不出来"，
   并据此估算"需要 136 万样本"。10 种子配对检验证明那个估算**只对低效的比较设计成立**——
   6.4 万行 × 10 种子就够了。**问题不是数据不够，是比较设计不够有效。**
5. **概率校准要选对方法：isotonic 是"有用的毒药"，Platt 才是正确选择**。
   isotonic 让 ECE 从 0.0298 降到 0.0112（降 62%），但把它 8020 个不同分数
   **压成只剩 30 个取值**，top-k 排序退化（k=1920 时增量 −36%）；
   换成严格单调的 **Platt 校准**后，**ECE 0.01197（几乎同样好）、零并列、
   top-k 重合率 100%、各 k 的增量收益变化为 0**。
   **用 0.0008 的 ECE 代价换回全部排序分辨率——所以用 Platt 一种分数即可同时满足排序与乘钱。**
   （绝对阈值决策下三者仍有差异，但差值落在本项目的噪声量级内，不足以宣称孰优。）

**投放节奏（模拟）**：把预算前置花掉是最大的浪费（仅为均匀投放的 25～28%）；
流量供给受限时，均匀投放会**剩下 17% 的预算花不出去**，而反馈控制器能把预算追回来
（Hillstrom：73.8 → 82.1；半合成：146.0 → 166.8）。

6. **观测数据下的因果估计有边界**：把选择偏差从 0 加到 2.0 后，
   朴素估计量的偏差**单调上升**（3.7 → 88.0，且恒为正——它把自然转化算成广告功劳）；
   IPS/DR 在中等混杂下明显更好，但**强混杂下 IPS 反而比朴素更差**
   （重叠性被破坏，倾向得分截断 4.6%）——这是识别性问题，不是实现问题。
   **另一个实测发现**：教科书 Qini 估计量的口径是 $n_t\cdot\overline{\tau}$ 而不是
   $|\text{S}|\cdot\overline{\tau}$，两者在等量随机化下相差约 2 倍，跨估计量比较必须先统一口径。

7. **固定预算 top-k 在预算宽裕时是错的**：真实正增量用户占 49.75% 时，
   预算放到 80% 会迫使 **38.2% 的预算投给负增量用户**，收益比阈值策略低 33%；
   但预算紧张（2%）时 top-k 的效率反而高 3.6 倍。
   **正确规则：预算 ≲20% 用 top-k 追效率；预算 ≳40% 改用"增量>0 才投"并主动不花完预算。**
   参考：全投的收益几乎为零（正负相消）。

---

## 环境要求

| 项目 | 要求 |
|---|---|
| Python | 3.9+（本机实测 3.10.1） |
| 依赖 | 见 `requirements.txt`（**必须 numpy<2**，见下方"已知坑"） |
| GPU | **不需要**。全部实验在本机 8 核 CPU 上跑完 |
| 内存 | 峰值 < 1GB（6.4 万行数据，无全量加载问题） |
| 磁盘 | 约 50MB（数据 4MB + 产物） |
| 系统 | Windows（实测）/ Linux / WSL |


---

## 安装

```powershell
# Windows PowerShell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

```bash
# Linux / WSL / Git Bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
```

## 数据准备

```powershell
.\.venv\Scripts\python.exe -m baua.cli download      # 下载 Hillstrom 并校验 SHA256
.\.venv\Scripts\python.exe -m baua.cli inspect-data  # 打印数据自检报告
```

- 数据来源：**Hillstrom MineThatData E-Mail Analytics And Data Mining Challenge (2008)**
  （Kevin Hillstrom, MineThatData）
- 下载地址：`http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv`
- 规模：**64,000 行 × 12 列**，实测文件大小 **3,964,977 字节**
- SHA256（实测）：`0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece`
- 许可与使用条件：该数据集由作者公开用于教学与研究。**发布衍生成果时请引用原作者与挑战赛名称**；
  商用前请自行确认授权范围（本项目未核验其明确的书面许可条款，标注为"未核验"）。

---

## 最短可运行命令（smoke test，约 20 秒）

```powershell
.\.venv\Scripts\python.exe -m baua.cli smoke
```

跑的是半合成小样本（4000 行、60 棵树），用于验证"数据 → 建模 → 指标 → 预算模拟 → 节奏模拟 → 产物"
整条链路。

## 完整实验命令

```powershell
# 全部实验（含测试、自检、三个数据集场景），约 5 分钟
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
```

```powershell
# 或分步执行
.\.venv\Scripts\python.exe -m pytest tests -q                                  # 单元测试（40 项）
.\.venv\Scripts\python.exe -m baua.cli run --config configs/default.yaml --tag main
.\.venv\Scripts\python.exe -m baua.cli run --config configs/default.yaml --tag synthetic `
    --set data.name=synthetic data.max_rows=40000
.\.venv\Scripts\python.exe -m baua.cli run --config configs/default.yaml --tag synthetic_conflicting `
    --set data.name=synthetic data.max_rows=40000 data.synthetic_mode=conflicting
# 多种子重训 + 配对方差分解（约 8 分钟，回答"策略差异是否稳定"）
.\.venv\Scripts\python.exe -m baua.cli multiseed --config configs/default.yaml --seeds 10 --tag multiseed_main
# 选择偏差研究：naive vs IPS vs DR（约 2 分钟）
.\.venv\Scripts\python.exe -m baua.cli confounding --n 40000 --levels 0 0.5 1.0 2.0
# 阈值策略 vs 固定预算 top-k（约 1 分钟）
.\.venv\Scripts\python.exe -m baua.cli threshold --n 40000 --mode conflicting
```

## 预期输出

全部产物落在 `artifacts/<数据集>_<tag>/`：

| 文件 | 内容 |
|---|---|
| `summary.json` | 环境、配置快照、数据概况、模型诊断（复现凭据） |
| `ranking_metrics.csv` | 各策略的 Qini / AUUC |
| `budget_allocation.csv` | 预算约束分配结果（增量收益、消耗、单位成本收益） |
| `budget_bootstrap_ci.csv` | **增量收益的 95% 自助法区间**（判断差异是否只是噪声） |
| `sensitivity_budget.csv` | 预算比例 1%/2%/5%/10%/20% 的对比 |
| `sensitivity_noise.csv` | 分数加噪 0～2σ 的鲁棒性 |
| `sensitivity_sample_size.csv` | 训练样本 5k/20k/全部 的影响 |
| `calibration_invariance.csv` | **校准对 top-k 排序的影响**（含引入的并列数） |
| `calibration_thresholds.csv` | 绝对阈值策略：未校准 vs 校准 |
| `pacing_strategies.csv` + `pacing_slots_*.csv` | 节奏策略对比与逐时段明细（含 relaxed/constrained 两个供给场景） |
| `calibration.json` | ECE 前后对比、分桶明细、校准器拟合信息 |
| `qini_curves.png`、`budget_sensitivity.png`、`pacing_cum_spend_*.png` | 图 |

`multiseed` 命令额外产出（写在 `artifacts/` 根目录）：

| 文件 | 内容 |
|---|---|
| `multiseed_*_per_seed.csv` | 每个种子 × 策略的 gain / Qini |
| `multiseed_*_summary.csv` | **跨种子均值±标准差 + 配对差异的 t 区间与显著性** |
| `multiseed_*_paired_diff.csv` | 逐种子的配对差值（同 seed 内减去 random） |
| `multiseed_*_qini_summary.csv` | Qini 的跨种子稳定性 |

`confounding` 与 `threshold` 命令的产物：

| 文件 | 内容 |
|---|---|
| `artifacts/confounding_study.csv` | 各混杂强度 × 策略下 naive_full / IPS / DR 的估计值与偏差 |
| `artifacts/threshold_vs_budget.csv` | 各预算比例下 top-k / 预测阈值 / oracle 阈值 / 全投的收益、效率与负增量占比 |

## 在线服务（可选，研究原型）

```powershell
# 1) 训练并保存策略模型（约 1 分钟，Hillstrom 全量）
.\.venv\Scripts\python.exe -m baua.cli fit --config configs/default.yaml --tag served

# 2) 启动服务
.\.venv\Scripts\python.exe -m baua.cli serve --models artifacts/models/served --port 8000
```

| 接口 | 作用 |
|---|---|
| `GET /health` | 健康检查；返回已加载策略与 `is_production: false` 声明 |
| `GET /strategies` | 策略列表 + 各自的 Qini（与离线实验一致） |
| `POST /score` | 批量打分：`{"rows": [{特征...}]}` → 每个策略的逐行分数 |
| `POST /allocate` | 预算约束分配：`mode=topk`（固定预算取前 k）或 `mode=threshold`（增量>阈値才投） |

示例：

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/allocate -H "Content-Type: application/json" \
  -d '{"rows":[{...}],"strategy":"s_learner","mode":"topk","budget_units":5}'
```



## 项目局限性

- 数据集是**邮件营销**随机实验，不是广告竞价数据，无成本/预算/竞价字段；
- 预算与节奏均为**模拟**，由此得到的"收益"是模拟核算值；
- **观测数据因果估计（IPS/DR）已实现并单独验证**（§4.8），但**未接入主流程**——
  Hillstrom 是随机实验（倾向为常数），接入它不会改变结论；
- **在线服务是研究原型**：无鉴权/限流/在线特征拼接/模型热更新，特征须由调用方提供；
- 未实现延迟反馈建模（Hillstrom 无时间戳，需半合成注入）；
- 时长与硬件限制：全部工作在单机 CPU 上完成，未做超参搜索。

## 文档索引

- `docs/项目设计.md` —— 问题定义、数据字段、方法选择理由、模拟假设、评估协议
- `docs/论文与仓库调研.md` —— 已核验的论文/数据集/仓库，含 URL、许可、借鉴点与未核验项
- `docs/实验报告.md` —— 真实运行命令、种子、实测指标、失败记录与限制
- `docs/面试讲解.md` —— 3 分钟讲稿、技术追问、因果假设、局限
- `docs/后续迭代路线.md` —— 4 小时窗口之外值得做的事（按优先级）

## 许可

本项目代码采用 MIT 许可（见 `LICENSE`）。
数据集版权归原作者，使用前请遵守其条款。
