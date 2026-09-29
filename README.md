# Budget-Aware Uplift Ads
### 预算约束下的增量转化广告排序与投放节奏优化（研究型原型）

---

## 30 秒说明

**研究问题**：广告预算有限时，按"预测转化概率"选人，还是按"广告带来的**增量**转化"选人，
哪种策略能带来更高的增量收益？模型误差、概率校准与预算消耗速度会如何影响结论？

**核心结论（本机实测，非编造）**：

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

## ⚠️ 重要声明（请先读）

1. **这是研究型原型，不是生产系统。** 没有复现任何公司的内部系统，
   也没有做线上 A/B 实验，不宣称任何业务提升。
2. **所有成本、预算、时段、价值系数均为模拟变量。** 公开数据集（Hillstrom）不含广告成本、
   预算或竞价字段；`cost_per_treatment`、`budget_ratio`、`peak_slot_boost` 等全部是模拟假设。
3. **数据集的干预是"营销邮件触达"，不是广告曝光。** 它提供的是"随机干预 + 二值结果"的
   公开基准，用于研究预算约束下的增量排序问题，**不等价于真实广告投放**。
4. **Qini/AUUC 是排序指标，不是收益指标。** 本项目的实测显示：
   conflicting 场景下所有策略的归一化 Qini 均为负值，而预算分配口径下 uplift 模型明显胜出——
   排序指标在"总效应接近零"时会失效（与 UpliftBench 2026 的警告一致）。

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

### 已知坑：不要用全局 Python 环境

本机全局环境是 **numpy 2.2.3，已破坏 scipy/scikit-learn**（ABI 不兼容，
导入 sklearn 直接抛 `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.3`）。
本项目锁定 `numpy==1.26.4`，**必须使用项目自带 `.venv`**。

---

## 安装

```powershell
# Windows PowerShell（在项目根目录）
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

## 数据准备（不会自动下载）

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
整条链路。**结果不用于任何结论。**

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

**明确的非目标**（不要把它说成生产系统）：无鉴权、无限流、无在线特征拼接、
无模型热更新、无 GPU 批量推理。特征必须由调用方提供；缺列直接返回 422。

## 上线前的质量门（本轮补齐）

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"   # ruff + mypy + pytest
.\.venv\Scripts\python.exe -m ruff check src tests      # 期望：All checks passed!
.\.venv\Scripts\python.exe -m mypy src/baua             # 期望：Success: no issues found
.\.venv\Scripts\python.exe -m pytest tests              # 期望：66 passed
```

真实记录（不美化）：ruff 初次扫描 **191 项**、自动修复 198 项、手工修 9 项（含 2 处**真实未使用变量**）；
mypy 初次 **55 项错误**，根因是把异构返回值标注为 `dict[str, object]` 导致无法推断属性，
改为 `dict[str, Any]` 并补 `types-PyYAML` 后归零。详见 `docs/实验报告.md` §6。

## 常见问题

**Q：为什么 `python -m baua.cli` 找不到模块？**
A：没有安装包。执行 `.\.venv\Scripts\python.exe -m pip install -e .`。

**Q：为什么 sklearn 导入报 NumPy 版本错误？**
A：用到了全局 Python。请用 `.venv\Scripts\python.exe`。

**Q：能下载 Criteo Uplift 数据集吗？**
A：官方链接已核验但**本机不可达**（返回 404 / 连接超时），因此不作为默认数据源。
若你已手动下载，可用 `baua.data.load_criteo_manual()` 读取。详见 `docs/论文与仓库调研.md`。

**Q：结果怎么复现？**
A：所有随机性由 `configs/*.yaml` 的 `seed` 控制；`summary.json` 记录了完整的配置快照与运行环境。
已验证：同种子跑两次 smoke，指标完全一致。

## 限制（务必阅读）

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

本项目代码采用 MIT 许可（见 `LICENSE`，**其中版权人姓名仍是 TODO，发布前请替换**）。
数据集版权归原作者，使用前请遵守其条款。

## 发布到 GitHub 前的检查清单（需人工审核）

本仓库**已在本地初始化 Git 并提交，但没有配置任何远端，也不会自动推送**。发布前请人工确认：

```powershell
# 1) 确认没有任何数据/模型/密钥被纳入版本控制
git status
git ls-files | Select-String -Pattern "\.(csv|gz|parquet|pkl|joblib)$"   # 应无输出
git ls-files | Select-String -Pattern "(\.env|secret|credential|\.key)"   # 应无输出

# 2) 替换占位信息
#    - LICENSE 中的版权人姓名
#    - pyproject.toml 中的 authors
#    - README 顶部的仓库地址（如有）

# 3) 逐条核验 docs/论文与仓库调研.md 中标记为 ❓未核验 的引用

# 4) 确认没有把模拟结果表述为线上效果（README/文档/讲稿三处自查）

# 5) 确认远端后手动推送（本仓库不做自动推送）
git remote add origin <你的仓库地址>     # 需人工确认地址正确
git push -u origin main                  # 人工执行
```

**切勿提交**：`data/`、`artifacts/`、`.venv/`、任何密钥或凭据。这些已在 `.gitignore` 中排除，
但请在 `git add` 后再次用 `git status` 确认。
