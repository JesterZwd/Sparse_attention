# 面向端侧 LLM 的 GQA 联合 Top-k/Top-p Page 选择：研究与实验方案

> **版本：V1.1｜更新日期：2026-11-17｜目标：ACM GLSVLSI 2027（以正式 CFP 为准）**  
> **定位：当前有效的算法规范与实验执行计划；尚无实测结论。**  
> **应用场景：** Batch=1 自回归 Decode、Full Attention + GQA、Paged KV Cache、目标 KV4、长上下文、低面积端侧加速器。  
> **暂定英文主题：** *Budget-Constrained Coverage-Aware Joint Page Selection for GQA Sparse Attention*。

## 0. 一页摘要：当前到底研究什么

**核心研究问题：** 在一个 GQA Group 的多个 Query Heads 共享同一套物理 KV Pages 时，能否在**固定 LPDDR 物理 Page 预算（Top-k 型约束）**下，根据**各 Head 的 Attention 概率质量覆盖目标（Top-p 型需求）**动态分配 Pages，使尾部 Head/关键检索质量改善，并用**小候选池、有限前沿、分批更新**把选择器的串行成本降低到可实现的范围？

研究主线：

1. **验证机制**：与强基线 Group-Average Top-k 比较，确认覆盖缺口驱动的动态选择是否有独立价值（Oracle）。
2. **降低选择成本**：分别研究候选池截断、指针式前沿及批量更新的质量损失与成本收益（Oracle）。
3. **落实可部署性**：用实际 KV4 + 低成本 Page 摘要生成估计概率，杜绝使用真实 Oracle Mass 做在线决策，并在模型闭环与 ASIC 上验证（Practical）。

**暂不确定：** 最终 Page 摘要形式、是否增加 Peak/Rank 保护、PE / Scorer 数据通路、Selector 微架构。**不预设此前研究中的 INT8×INT2 Bit-Slice PE。**

**三道停/继续门槛：**

- **Gate 1 — 联合选择有效性（Oracle）**：在相同物理 Page 预算下，动态覆盖选择能否改善尾部覆盖或 Attention 输出误差？`clip_rate` / `R_change` 只解释机制，不作为单独放行条件；低频改变也可能保护关键 Head。
- **Gate 2 — 低开销近似（Oracle）**：有限候选、指针前沿和分批更新能否在降低选择工作量与串行依赖的同时，保留 Gate 1 所见收益？
- **Experiment 2 → Experiment 3 启动条件**：Gate 1/2 通过后，进一步验证可构建的 KV4 Metadata 评分和初步存储/计算成本；若存在可信的质量—成本可行区间，**才启动 RTL/PPA/系统评估**。
- **Gate 3 — 部署净收益（Experiment 3 完成后判定）**：根据实测/经校准的面积、延迟、能量、总 LPDDR 流量与任务质量，判断是否具备 ASIC 价值；**Gate 3 不得作为启动 Experiment 3 的前置条件**。

---

## 1. 问题定义与不可违反的约束

### 1.1 符号与 Oracle 信息

针对**单个 GQA Group**：

| 符号 | 含义 |
|---|---|
| $H$ | 共用一个 KV Head 的 Query Heads 数量 |
| $J$ | 当前 Query 可访问的有效物理 KV Pages 数量 |
| $D$ | 每 Head 维数，主模型为 128 |
| $s_{i,h}$ | Head $h$ 对有效历史 Token $i$ 的 QK Logit（含标准缩放） |
| $P_{g,h}$ | Page $g$ 对 Head $h$ 的**真实** Attention Mass 占比；仅 Oracle 阶段可直接获得 |
| $\widehat P_{g,h}$ | 部署阶段经轻量 Metadata 估计的 Head 内归一化 Page Mass |
| $S_0$ | 强制 Sink + Local 保护 Page 集合，去重后计入预算 |
| $S$ | 最终统一读取的共享物理 Page 集合 |
| $B$ | 总物理读取 Page 上限，要求 $|S_0|\le B\le J$ |
| $p_h$ | Head $h$ 的目标概率质量覆盖率（例如 0.7/0.8/0.9） |

令 $g$ 中有效 Token 集为 $\mathcal I_g$（允许尾 Page 不满）：

$$
P_{g,h}=\frac{\sum_{i\in\mathcal I_g}\exp(s_{i,h})}{\sum_{i\in\mathrm{valid}}\exp(s_{i,h})},\qquad \sum_{g=1}^{J}P_{g,h}=1.
$$

在实现中使用稳定的 `logsumexp` / 减最大值计算概率；每个 Head 独立归一化，**不允许直接跨 Head 混用原始 Logit 的数值尺度**。

定义：

$$
C_h(S)=\sum_{g\in S}P_{g,h},\qquad d_h(S)=\max(0,p_h-C_h(S)).
$$

基础优化目标：

$$
\boxed{F(S)=\sum_{h=1}^{H}\min\bigl(p_h,C_h(S)\bigr)
=\sum_h p_h-\sum_h d_h(S),\quad S_0\subseteq S,\quad |S|\le B.}
$$

**意义边界：** 最大化 $F$ 就是最小化**各 Head 剩余缺口之和**；它**不保证**最弱 Head 覆盖率、达标 Head 数量或任务准确率最优，这些必须独立评价。在 $P_{g,h}\ge0$、等成本 Page 且固定候选池时，$F$ 是单调子模函数；**完整逐页贪心 A3**有标准近似保证，有限前沿和分批版本不自动继承该保证。

### 1.2 统一保护与公平计费

- 全部稀疏算法强制选入同一组 $S_0$：先采用 `Sink=1 Page, Local=128 Tokens`，再做保护区大小消融。
- 同一个 Page 同时属于 Sink、Local 或多 Head 候选时，**物理读取只计一次**；保护 Page **计入 $B$ 和真实 KV 字节数**。
- 物理页可能包含部分无效 Token；只对有效 Token 执行 Causal Mask、Mass 求和与 Attention。
- $B<|S_0|$ 或 $B>J$ 属于无效配置；全模型预算为**各 KV Group 各层真实读页数的合计**，不要将单组页数当成整模型流量。
- 第一轮 Oracle 从**同一 Dense KV4 前向路径**采集 Q、K、V、有效长度；不同算法不得改变量化或参考状态。

---

## 2. 待测算法清单

| ID | 方法 | 核心规则 | 定位 / 阶段 |
|---|---|---|---|
| **A0** | Per-Head Top-k + Budgeted Union | 各 Head 按 Mass 独立提议并去重，限定物理总预算 | 基础对照；Exp 1A |
| **A1** | Group-Average Top-k | $W_g=\sum_h P_{g,h}$，统一按 $W_g$ 排序 | **必须比较的强基线**；Exp 1A |
| **A2** | Max-Deficit Head First | 找最大 $d_h$ 的 Head，选其最高剩余 Page，更新全部 Head | 极简动态对照；Exp 1A 完整版 |
| **A3** | Full Marginal-Gain Greedy | 全部未选页中，每轮取最大 $\Delta_g$ | **完整贪心参考**；Exp 1A |
| **A3-C** | Candidate-Pool Greedy | 固定候选池 $\mathcal C$，池内全扫描 $\Delta_g$ | 归因：候选截断；Exp 1B |
| **A3-B** | Full-Pool Batched Greedy | 全页收益评估，每批取 Top-$b$ 并统一更新 | **最小实验中的纯批量消融**；Exp 1A 探针 |
| **A4** | Head-Frontier + Shared-Gain | 有限池、有限前沿，逐页选最大 $\Delta_g$ | 归因：前沿限制；Exp 1B |
| **A4-B** | Batched Frontier Shared-Gain | A4 前沿内每批取 Top-$b$、批后更新缺口 | **重点低开销候选**；Exp 1B |
| **A5** | Rank-Safeguarded Coverage Selection | 近似 Mass 下加入少量峰值风险保护，影响**最终选择** | Exp 2，可选创新扩展 |

> `A3-B` 与 `A4-B` 必须区分：**前者只隔离批处理损失，后者还同时受候选/前沿限制**。不把二者混在一个消融中。

### 2.1 A1：最强的固定排序基线（含自适应版本）

共享分数及确定性排序：

$$W_g=\sum_hP_{g,h},\qquad L_W=\operatorname{Sort}_{\downarrow}(\{(W_g,g):g\notin S_0\}).$$

排序规则：分数降序，完全相同时 Page ID 升序；从 $S_0$ 开始，沿 $L_W$ 顺次加入尚未选中的 Page。

- **A1-Fixed**：取 $L_W$ 前 $B-|S_0|$ 张，最终 $|S|=B$。
- **A1-Adaptive**：**每加入一页后**更新全部 Head 的 $C_h$；当所有 $d_h\le\varepsilon$ 时立即停止，否则最多取到 $B$。若 $S_0$ 已使全部 Head 达标，则不再加入新页。

在固定物理预算下，A1 对**各 Head 的 Mass 总和**是最优选择。其他算法不能声称普遍提高平均保留 Mass；主要比较**尾部 Head、剩余缺口、Attention 输出误差和任务质量**。同一 $L_W$ 同时作为其他方法**达标后补页**的排序来源（构建成本单独核算）。

### 2.2 A2：最大缺口 Head 优先

每轮取 $h^*=\arg\max_h d_h$，然后取该 Head 尚未选的最高 $P_{g,h}$ Page，更新所有 Heads 的覆盖。控制简单，但无法预先识别“同时服务多个 Head 的中等得分页”。

### 2.3 A3：完整动态共同收益

$$\boxed{\Delta_g(S)=F(S\cup\{g\})-F(S)=\sum_h\min(P_{g,h},d_h(S)).}$$

每轮在全部未选页中选最大 $\Delta_g$，更新 $S$、各 Head 覆盖与缺口。对比 A1 可以定位“已经满足覆盖的 Head 是否继续浪费 Page”的机制价值。

### 2.4 A4：有限候选池与**指针式**前沿

首先在全历史评分后，构建排除 $S_0$ 的有序列表：

$$
L_h=\operatorname{TopR}_{g\notin S_0} P_{g,h},\qquad
L_s=\operatorname{Top}_{B'}\{W_g:g\notin S_0\},\quad B'=B-|S_0|.
$$

候选池为

$$\boxed{\mathcal C=\bigl(\cup_hL_h\bigr)\cup L_s.}$$

**$L_s$ 至少含 $B'$ 个不同未保护 Pages**，用于保证严格定预算模式可以从已建立的排序列表中补足；其生成和存储成本必须单独记账，不能当作免费结构。

**共享列表生成的两种待比较硬件路径（均须完整 Head 内归一化）：**

1. **缓存评分（one-pass + score SRAM）**：扫描 Metadata 后保存全部 $J\times H$ 个 Page–Head 的 Log-Mass/Score（或等价可归一化状态），待得到各 Head 的全历史归一化分母后，计算 $\widehat P_{g,h}$、$W_g$ 并建立 $L_s$。若每个分数宽度为 $w$ bit，仅矩阵存储约为 $JHw/8$ Bytes；例如 $J=1024,H=8,w=16$ 时为 **16 KiB/活动 Group**，未计索引和列表。
2. **双遍评分（low SRAM + metadata reread）**：首遍累积各 Head 的全历史分母与 Head 候选排序所需状态；次遍重读 Metadata、重新算 Page–Head 分数并构造 $W_g$ 和 $L_s$。节省全页 Score SRAM，但额外增加一次 Metadata 读取、评分运算与扫描延迟。

必须分别核算 `score_sram_bytes`、`score_sram_accesses`、`metadata_read_bytes`、`scorer_mac_ops`、`shared_list_build_cycles`。两种路径均不可假定在得到 $W_g$ 前就已免费获得归一化值；前沿阶段可按 Group 分时复用状态。

每轮实际计算收益的**前沿**定义为：

$$
\boxed{\mathcal F_t=\bigcup_h\operatorname{FirstUnselected}_{f}(L_h)
\;\cup\;\operatorname{FirstUnselected}_{f_s}(L_s).}
$$

`FirstUnselected` 表示**按列表次序向前跳过所有已选页，暴露当前最前方指定数量的未选页**；用有序列表+读指针/局部扫描模拟。重复候选 Page 去重后只评分一次；每选一批后更新列表的下一次暴露位置。前沿大小受 $|\mathcal F_t|\le Hf+f_s$ 约束。

**不要把整个 $L_s$ 都放进每轮前沿。** 兜底列表可长达约 $B'$ 页，但每轮只暴露 $f_s$ 项。达到全部覆盖目标后的补页阶段，才可以直接沿 $L_s$ 读取剩余页。

### 2.5 A4-B：前沿内分批选择

- 每批按**当前** $d_h(S)$ 给前沿候选计算 $\Delta_g(S)$。
- 取 Top-$b$ 的至多 $b$ 张**不同**未选页 $Q$，且不得超过剩余物理预算。
- **整批加入后**更新一次 $C_h,d_h$，重建下一轮前沿。
- 实际批大小 $|Q|$ 可能小于名义 $b$；要记录分布。试验 $b=1,2,4,8,16,32$。
- 大 $b$ 降低串行决策轮数，但可能在批内重复照顾同一已经满足的 Head；不预设 $b=16/32$ 一定优秀。

### 2.6 A5：Rank 风险保护（仅作为 Exp 2 候选）

低成本 Page Mass 估计可能漏掉“单个高峰值 Key”。仅往候选池里加入 Peak 页，但继续使用严重低估的 $\widehat P$ 算边际收益，并不保证它被选中。因此 A5 只测试**真正影响最终 Page 选择**的 Rank 保护：小的硬保护配额，或在可校准的评分规则中引入峰值风险。两个方案均须计入新增 Metadata、评分与预算成本，并给 A1 提供**相同保护预算**的对照，防止把额外信息误认为联合选择收益。

---

## 3. 预算模式、边界与可行性

### 3.1 模式 F：严格固定预算（**主实验**）

1. 从 $S_0$ 开始，动态选择直到 $|S|=B$ 或全部 Head 达标。
2. **若全部 $d_h\le\varepsilon$，立即停止动态缺口决策。** 后续从**预先建立的** A1 组级排序列表中顺次补页，确保 $|S|=B$。
3. 记录 `coverage_selection_pages` 与 `a1_fill_pages`；达标后的补页轮次**不得计入动态决策改变率**。
4. A3 可使用全页 $W_g$ 的排序；A4/A4-B 直接沿预存 $L_s$ 补页，不重新扫描完整历史。补页排序结构的存储与构建成本需计入系统开销。

严格定预算是**等 Page 个数的选择质量比较**，不是自适应带宽节省证明。

### 3.2 模式 E：自适应早停（扩展实验）

- **A1-Adaptive**：在固定组级排序 $L_W$ 上**逐页**加入、逐页检查各 Head 覆盖；全部 $d_h\le\varepsilon$ 即停，否则最多到 $B$。与动态方法使用同一保护页、预算与目标。
- **A2/A3/A4**：每选一页即更新缺口并检查达标；满足则停，否则最多到 $B$。
- **A3-B/A4-B**：每批完成后才更新缺口并检查达标；允许批内超额页，记录 `first_satisfied_position`、`batch_end_pages` 与 `batch_overshoot_pages`。若需要准确的首次达标位置，可仅在离线审计时按批内选页顺序逐项重算覆盖，不改变算法实际批后决策规则。

全部算法记录实际 $|S|$ 与总访问字节，**自适应模式图表横轴使用真实页数/字节，不得将上限 $B$ 冒充实际读取量**。

### 3.3 预算可行性诊断

对每个 Head，计算单 Head 的必要上界：

$$U_h(B)=C_h(S_0)+\sum_{g\in\operatorname{Top}_{B'}(P_{g,h};g\notin S_0)}P_{g,h}.$$

若 $U_h(B)<p_h$，则即使把全部余量给这个 Head，它也不可能达到目标。所有 Head 分别通过检查**仍不代表联合预算一定可行**；可对少量困难样本使用离线整数规划/穷举检验联合可行性，供解释结果，**不进入线上选择器**。

---

## 4. 机制指标：不仅比较最终覆盖率

### 4.1 收益截断与动态排序改变

当对当前全部候选 Page 均满足 $P_{g,h}\le d_h(S)$ 时，

$$\Delta_g(S)=\sum_hP_{g,h}=W_g,$$

即 A3 的选择优先级与 A1 完全相同。

- **`clip_rate`**：有效动态选择轮次中出现 $P_{g,h}>d_h$ 的 Head–Page 贡献占比；可另记录“有任何截断的轮次比例”。
- **`R_change`**：仅在 $\exists h:d_h(S_t)>\varepsilon$ 的**有效动态决策轮次**，比较 $\arg\max_{g\notin S_t}\Delta_g$ 与在**相同合法未选集合**上的 $\arg\max_{g\notin S_t}W_g$：

$$
R_{\mathrm{change}}=\frac{\#\{t\in\mathrm{active}:\operatorname{argmax}\Delta_g\ne\operatorname{argmax}W_g\}}{\#\{t\in\mathrm{active}\}}.
$$

统一确定性 Tie-break：先按较高得分，再按较小 Page ID。对 A4/A4-B 的受限前沿，需要同时报告“**同前沿基线**”与“**全体未选页基线**”，避免把候选限制误算为动态目标的效果。**达标后 A1 补页轮次必须排除**。高改变率仅说明选择不同，不自动等于质量更好。**Gate 1 以覆盖缺口、Attention 输出误差及后续任务收益为主；即使改变率低，也可能改善少数关键 Retrieval Heads。**

### 4.2 分批集合改变率、冗余与实际批大小

**批量集合改变率 `R_batch_change`：** 对同一个动态批次起点 $S_t$、同一个**实际合法候选集合** $\mathcal F_t$，设 $Q_\Delta$ 为按动态收益 $\Delta_g(S_t)$ 选出的 $k=|Q_\Delta|$ 张，$Q_W$ 为按固定 $W_g$ 在**相同 $\mathcal F_t$** 中选出的 $k$ 张；两者均按统一 Tie-break：

$$
\boxed{R_{\mathrm{batch\_change},t}=1-\frac{|Q_\Delta\cap Q_W|}{|Q_\Delta|}.}
$$

只在仍存在 $d_h>\varepsilon$、且 $|Q_\Delta|>0$ 的**动态选择批次**统计；排除达标后的 A1 补页。汇总时报告逐批均值及按实际批大小加权的全局比例。对于受限前沿，可另将 $Q_W$ 改成全体合法未选页上的 Top-$k$ 作**候选限制诊断**，与同前沿比较分开报告。单页的 `R_change` 和批量的 `R_batch_change` 不互相替代。

**批内冗余：** 对于每批 $Q$ 在起始状态 $S$ 的预测收益之和：

$$\boxed{R_{\mathrm{batch}}=\sum_{g\in Q}\Delta_g(S)-[F(S\cup Q)-F(S)]\ge0.}$$

归一化冗余：

$$\eta_{\mathrm{batch}}=\frac{R_{\mathrm{batch}}}{\sum_{g\in Q}\Delta_g(S)+\epsilon}.$$

报告 `actual_batch_size` 分布、动态迭代轮数、总收益计算次数、Top-$b$ 排序/比较工作量、候选表访问与缺口更新次数。分批降低的仅是**决策更新频率**，实际选择页对所有 Heads 的覆盖累加仍需完成。

### 4.3 覆盖指标与尾部质量

主要指标：`avg_retained_mass`、`min_head_mass`、`p05_query_head_mass`、`unsatisfied_head_ratio`、`remaining_deficit_sum`、`candidate_recall`（相对于完整贪心的被选页）及关键检索 Page/Token Recall。统计 P5 时在大量不同样本、层、Decode Steps 与 Heads 上汇总，而不是在仅 4/7 个 Heads 的单组中机械计算百分位。

**归因原则：** A3 最大化的是截断后的覆盖总和 $F$，它**不能保证**改善最弱 Head、输出误差或任务结果；即使 A1 总 Mass 更高，A3 可能改善低覆盖 Head，这也需数据证实。

### 4.4 Oracle 阶段就测 Attention 输出误差

所有方法使用**同一套 V 表示**。若 QK/Oracle 来自 KV4 路径，则 Dense 和 Sparse 输出均使用同一套 KV4 解量化 V；另外可独立测 BF16 Oracle，不允许混淆量化误差与稀疏误差。

数学上，在选中有效 Token 非空时：

$$O_h^{\mathrm{dense}}=\sum_i\alpha_{i,h}v_i,\qquad
O_h^{\mathrm{sparse}}(S)=\frac{\sum_{i\in S}\alpha_{i,h}v_i}{C_h(S)}.$$

其中 $\alpha_{i,h}$ 为同一 Query/历史/KV 路径上的 Dense Softmax 权重。**实际数值实现不得直接用已经归一化的 $\alpha$ 再除以 $C_h(S)$**：当被选 Token 的 Dense 概率整体下溢为 0 时会发生 `0/0`，即使选中集合非空。

改为在**选中有效 Token** 上用原始 Logit 重新执行稳定 Softmax：

$$
\boxed{m_S=\max_{i\in S\cap\mathrm{valid}}s_{i,h},\quad
O_h^{\mathrm{sparse}}(S)=
\frac{\sum_{i\in S\cap\mathrm{valid}}e^{s_{i,h}-m_S}v_i}
{\sum_{i\in S\cap\mathrm{valid}}e^{s_{i,h}-m_S}}.}
$$

Dense 输出亦用全部有效 Logits 的稳定 Softmax 计算。只要选中集合至少包含一个合法且 Logit 有限的 Token，分母就为正；应对空选集、全 `-inf`、非有限 Logit 设置显式输入校验/错误路径，而不依赖浮点 `eps` 掩盖不合法输入。**Dense/Sparse 使用完全相同的 V 数值表示**（如均为 KV4 解量化 V），以隔离稀疏误差。

$$E_h=\frac{\|O_h^{\mathrm{sparse}}-O_h^{\mathrm{dense}}\|_2}{\|O_h^{\mathrm{dense}}\|_2+\epsilon}.$$

同时报告绝对 $L_2$ 误差、相对误差及 P95/P99 尾部；有条件时增加输出投影后误差。注意：相同 Mass 不代表相同输出误差，因为遗漏的 Value 内容不同。Oracle Trace 结果**不能替代闭环 Sparse Decode 的 RULER/LongBench 任务评估**。

---

## 5. 实验执行计划：先小规模证伪，再补全完整证据

### 5.1 Minimal Probe：最小证伪实验（第一优先级）

**目的：** 在投入候选前沿、摘要、RTL 之前，确认动态覆盖缺口是否有意义，以及批量更新会损失多少质量。

只先实现：

1. `A1` Group-Average Top-k。
2. `A3` Full Marginal-Gain Greedy。
3. `A3-B` **全页候选** Batched Greedy（先 $b=1,4,8,16$；后续全扫）。
4. 同一 Dense KV4 Oracle Trace 的 `clip_rate`、`R_change`、Bottom-5% Coverage、`remaining_deficit_sum`、$O_h$ 相对/绝对误差与批内冗余。

先采代表性 **16K** Trace，后扩至 **32K**：

- `Qwen/Qwen3-4B-Instruct-2507`：GQA 4:1。
- `Qwen/Qwen2.5-7B-Instruct`：GQA 7:1；注意**输入+输出总 Token 长度不得超出模型实际配置上限**，可用约 28K–30K 提示长度为生成预留空间。
- 第一轮 `Page Size=32`，主预算 5%、10%、20%，$p_h=0.7,0.8,0.9$。保护配置固定，所有方法一致。

**判定：** 若 A3 与 A1 选择几乎相同、截断很少生效、尾部覆盖与 Attention 输出差异不明显，则先排查阈值/层/数据分布，不立即推进复杂 Selector。若有可靠收益，再启动下一节。

### 5.2 Experiment 1A：Oracle 完整对照

补充 A0/A2，完善 A1/A3/A3-B 的更广参数扫描；报告 Strict-Budget 与 Adaptive-Stop 两种模式。所有选择使用真实 Page Mass，**此阶段不把 Oracle 当成可部署算法**。

### 5.3 Experiment 1B：四级损失归因

固定同一 Score 和预算条件，顺序比较：

$$\boxed{\mathrm{A3\ (all\ pages)}\ \rightarrow\ \mathrm{A3\!\text{-}C\ (candidate\ pool)}\ \rightarrow\ \mathrm{A4\ (frontier)}\ \rightarrow\ \mathrm{A4\!\text{-}B\ (batched)}.}$$

参数：$b\in\{1,2,4,8,16,32\}$，每 Head 候选 $R$、前沿 $f\in\{1,2,4\}$，共享前沿 $f_s\in\{4,8,16\}$，Page Size 16/32/64。$L_s$ 按每次预算 $B'=B-|S_0|$ 建立；**不要把 $L_s$ 全量暴露到每轮前沿**。

另进行同一候选全集上的 `A3-B` 与 `A4-B` 横向对比，以拆开“批量收益”和“候选池限制”。如需精确诊断联合预算可行性，可对少量困难组做离线整数规划，不把它当主要运行负载。

### 5.4 Experiment 2：实际可部署的 KV4 + Metadata 评分

固定实际 KV4 Key/Value 量化方案、Scale/Zero-point 布局及 Page 维护方式；比较 **Quest 原始 Min/Max Top-k、Single-Mean、可选 Streaming 2-Centroid/其他紧凑摘要**。摘要编码、构建和评分复杂度计入成本，**不预设双质心是最终方案**。

将真实 $P$ 换成 $\widehat P$ 后，以下路径**全部仅用估计值决策**：

- 各 Head 与组级候选排序；
- 归一化分母（必须来自**完整历史摘要扫描**，而不是候选池内部重新归一化）；
- 缺口、Top-$b$ 增益、达标判断、自适应早停和严格预算补页。

**真实 $P$ 与真实 QK Peak 只能用于离线评价或明确标注的 Oracle 消融，不能泄漏到部署算法。** Quest 上界 $U$ 不是质量估计；保留 Quest 原始 Top-k 作为正式基线，并可另外测试 softmax-of-upper-bound 启发式代理（明确其概率并无覆盖保证）。若测试 A5，确保 Rank 直接影响最终选择，并给 A1 等资源等配的 Peak 保护对照。

建议用交叉表拆分：

| 选择器 / 评分 | Oracle Mass $P$ | 相同实际摘要 $\widehat P$ |
|---|:---:|:---:|
| A1 | ✓ | ✓ |
| A3 / A4-B | ✓ | ✓ |
| A5（若启用） | 仅机制诊断 | ✓（主评价） |

最终任务：16K/32K RULER（NIAH、多键/多值检索）、按 Tokenizer 长度筛选的 LongBench v2 子集、256–1024 Token 长 Decode。先 Dense Prefill + Sparse Decode，再进行闭环生成，区分 Trace 与端到端结果。

### 5.5 Experiment 3：硬件与系统闭环（完成后判定 Gate 3）

**启动前提：** Gate 1 与 Gate 2 已经通过，并且 Experiment 2 证实实际 KV4 + 可构建 Metadata 在初步评分准确度、存储流量和计算成本上存在可信的部署窗口。**不要求 Gate 3 预先通过**；Gate 3 就是 Experiment 3 的终局评价。

针对胜出算法设计硬件，不预设 INT8×INT2 或任何固定 PE。关注 Page Scorer、候选 Top-$R$/Top-$B'$ 构建、前沿指针、Top-$b$、Deficit Register、Bitmap、GQA Shared KV Buffer 与必要的 Per-Head Compute Mask。比较独立/共享 Scorer 的面积和关键路径；可按机制需要引入定点/LUT/混合精度。

要求在相同工艺/可比 SRAM 与算力预算下，对照 `Dense KV4`、`Quest-like Sparse`、`Proposed`；报告面积、时序、动态功耗、SRAM 读写、Metadata+Scale+K/V 的**总 LPDDR 字节与能量**、每 Decode Token 延迟，并明确 Amdahl 上限。**固定 Page 预算不能自动证明 LPDDR 字节减少；它证明的是选择质量/精度折中。** 还需对照 §2.4 两种共享列表构建路径，明确单遍 Score SRAM 与双遍 Metadata 重读对面积、总读取量、能量、关键路径及排序构建周期的影响。**只有在本实验完成并获得上述 PPA/系统结果后，才做 Gate 3 净收益判定。**

---

## 6. 必须通过的正确性测试

实现前先用合成小矩阵（不需要 LLM）做单元测试：

1. **线性退化**：统一 $p_h=1$ 时，在同保护、同预算、同 Tie-break 下，`A3` 与 `A1` 的选择顺序一致；单元测试使用足够严格的数值容差，避免在未真正达到覆盖 1 时由较大的停机 `eps` 触发补页。
1a. **A1 自适应一致性**：A1-Adaptive 与沿同一 $W_g$ 排序逐页更新覆盖并早停的参考实现一致；保护页已满足全部目标时只返回 $S_0$；A1-Fixed 始终返回 $B$ 页。
2. **批量退化**：`A4-B(b=1)` 与 `A4` 一致；`A3-B(b=1)` 与 `A3` 一致。
3. **候选退化**：候选全集、前沿全集时，`A4` 与 `A3` 一致。
4. **物理预算与保护**：$S_0\subseteq S$、物理 Page 去重、$|S|\le B$；严格模式 $|S|=B$；自适应模式能正确提前停。
5. **输出退化**：全部有效页被选时 Sparse Output 与同 KV 数值域下的 Dense Output 一致；批内 $R_{\mathrm{batch}}\ge-\epsilon$。
6. **有效轮次与批次改变率**：`R_change` / `R_batch_change` 仅统计至少一个 Head 未达标时的动态选择步骤/批次，排除 A1 补页；比较双方使用相同候选集合、实际批大小与 Tie-break。
7. **统一 V**：同一次 Attention 输出误差比较中 Dense 与 Sparse 必须共用同一套 V 表示。
8. **估计隔离**：Exp 2 用扰动/打乱真实 Oracle Mass 不应改变估计模式的选择结果；除离线评价外，不允许调用真值。
9. **尾页与数值安全**：不满页合法 Token 数正确；空选集、全保护、预算不可行、Tie、极端 Score、浮点舍入均有确定行为；在高分仅存在于未选页的构造案例中，稳定 Sparse Softmax 不得出现 `0/0` 或 NaN。

### 6.1 参考伪代码：统一选择核心（完整含两个停止分支）

```python
# 以下为 Python 风格逻辑伪代码；可将 mass 换成估计质量。
# frontier_fn: A3 返回全部合法未选页；A3-C 返回候选池未选页；
#              A4/A4-B 返回头列表+共享列表指针式前沿。

def select_pages(mass, protected, budget, p, group_order,
                 frontier_fn, batch_size=1, mode="fixed", eps=1e-8):
    # mass: [J, H], 每个 Head 已按完整历史归一化。
    J, H = mass.shape
    S = set(protected)
    assert len(S) <= budget <= J
    c = mass[list(S)].sum(axis=0) if S else np.zeros(H)
    audit = []

    while len(S) < budget:
        d = np.maximum(p - c, 0.0)
        active = bool(np.any(d > eps))

        if not active:
            if mode == "adaptive":
                break
            # 固定预算：只读取预存 A1 组级有序列表，不再运行缺口贪心。
            for g in group_order:
                if len(S) >= budget:
                    break
                if g not in S:
                    S.add(g)
                    c += mass[g]
                    audit.append(("a1_fill", g))
            assert len(S) == budget
            break

        F = sorted(set(frontier_fn(S)) - S)
        if not F:
            raise RuntimeError("dynamic frontier exhausted before coverage/budget")

        gains = {
            g: float(np.minimum(mass[g], d).sum())
            for g in F
        }
        take = min(batch_size, budget - len(S), len(F))
        # 降序 Gain，Page ID 升序，确保可重现。
        Q = sorted(F, key=lambda g: (-gains[g], g))[:take]
        if not Q:
            raise RuntimeError("no selectable page")

        for g in Q:
            S.add(g)
            audit.append(("dynamic", g))
        c += mass[Q].sum(axis=0)  # 仅批次末刷新缺口

    return sorted(S), c, audit
```

**注意：** 算法 A1 单独按 $W_g$ 排序，且有下面明确的自适应规则；A2 单独实现最大缺口 Head 列表选择。批内冗余、首次达标页和机制改变率应在循环内外另行记录，不能仅靠返回的最终 `c` 推导。`group_order` 必须在 Exp 2 中由估计质量构建，不得由 Oracle 构建。

```python
# A1 固定排序与自适应早停；mass 可在 Exp 2 替换为估计值。
def select_a1(mass, protected, budget, p, mode="fixed", eps=1e-8):
    J, H = mass.shape
    S = set(protected)
    assert len(S) <= budget <= J
    c = mass[list(S)].sum(axis=0) if S else np.zeros(H)
    W = mass.sum(axis=1)
    group_order = sorted((g for g in range(J) if g not in S),
                         key=lambda g: (-W[g], g))
    for g in group_order:
        if len(S) >= budget:
            break
        if mode == "adaptive" and np.all(p - c <= eps):
            break
        S.add(g)
        c += mass[g]
    return sorted(S), c
```

```python
# Attention 输出必须使用原始 Logits 进行稳定的局部 Softmax；
# dense 与 sparse 传入完全相同的 V 数值表示。
def stable_attention_output(logits, values, valid_selected_indices):
    idx = np.asarray(valid_selected_indices, dtype=np.int64)
    if len(idx) == 0:
        raise ValueError("selected valid token set cannot be empty")
    scores = logits[idx].astype(np.float64)
    if not np.all(np.isfinite(scores)):
        raise ValueError("invalid selected logits")
    shifted = scores - scores.max()
    weights = np.exp(shifted)
    weights = weights / weights.sum()
    return weights @ values[idx]
```

**伪代码实现约束：** 候选列表排序、批内 Top-$b$、A1 补页均使用统一 Tie-break。对严格定预算模式，动态阶段达到覆盖目标后统一补足；A1-Adaptive 在每页加入后立即判断，不执行补页。`R_batch_change` 的同集合对照和共享列表构建成本必须通过独立的统计模块记录。

---

## 7. 初始配置建议（可直接转为仓库配置）

```yaml
project: glsvlsi27_gqa_joint_page_selection
phase: oracle_minimal_probe
models:
  primary: Qwen/Qwen3-4B-Instruct-2507
  secondary: Qwen/Qwen2.5-7B-Instruct
  literature_anchor: meta-llama/Llama-3.1-8B-Instruct
context_lengths: [16384, 32768]
page_size_main: 32
page_size_ablation: [16, 64]
kv_target_bits: 4
oracle_source: dense_kv4_qk
strict_same_v_representation: true
protected:
  sink_pages: 1
  local_tokens: 128
budgets_fraction: [0.05, 0.10, 0.20, 0.30]
coverage_targets: [0.7, 0.8, 0.9]
modes: [fixed, adaptive]
fixed_fill: prebuilt_group_average_order
adaptive_stop: all_heads_covered_or_budget_reached
algorithms_probe: [A1, A3, A3-B]
algorithms_full: [A0, A1, A2, A3, A3-C, A3-B, A4, A4-B]
batch_sizes: [1, 2, 4, 8, 16, 32]
candidate_r: [16, 32, 64, 128]
head_frontier_f: [1, 2, 4]
shared_frontier_fs: [4, 8, 16]
shared_fallback_size: budget_minus_protected
score_truth_only_for_evaluation_in_practical: true
metrics:
  - avg_retained_mass
  - min_head_mass
  - p05_query_head_mass
  - unsatisfied_head_ratio
  - total_deficit
  - clip_rate
  - decision_change_rate
  - batch_selection_change_rate
  - attention_output_l2_relative
  - attention_output_l2_absolute
  - batch_redundancy
  - actual_batch_size
  - selector_gain_evaluations
  - selector_iterations
  - score_sram_bytes
  - score_sram_accesses
  - metadata_read_bytes
  - scorer_mac_ops
  - shared_list_build_cycles
  - metadata_plus_kv_bytes
shared_list_build_paths: [cached_scores, two_pass_metadata]
a1_adaptive_stop: per_page
attention_output_method: stable_softmax_over_selected_logits
hardware: open_after_gate1_gate2_and_practical_cost_feasibility
gate3_evaluated_after: experiment3_ppa_system
```

> 以上参数是**待测网格**而非最终最佳配置。对于每个预算，都需按有效 $J$ 和 $|S_0|$ 计算 $B$、$B'$；对不同模型/上下文长度按相同物理页预算比例比较。真实 Page 大小、Page 数、实际 V/K Scale 和工作集必须另行统计。

## 8. 结果判定与论文定位

| 比较 | 问题 | 有价值的结果 |
|---|---|---|
| **A1 vs A3** | 截断覆盖机制本身是否有效？ | 在可接受平均 Mass 损失下改善低覆盖 Head 或输出尾部误差；`clip_rate` / `R_change` 用于解释而非强制门槛 |
| **A3 vs A3-C** | 有限候选导致多少损失？ | 大幅减少候选状态且质量接近 |
| **A3-C vs A4** | 指针前沿导致多少损失？ | 在较小每轮前沿下维持质量，降低收益计算/存储访问 |
| **A4 vs A4-B** | 批内缺口滞后损失有多大？ | 以更少串行轮次保留质量；批内冗余与整批选择改变率可解释 |
| **A1/A4-B（Oracle vs Practical）** | 低成本摘要是否毁掉选择增益？ | 可部署评分下仍有真实质量/成本收益，且比较公平 |
| **A5 vs 等资源 Rank 基线** | Peak 保护是否真的有增量？ | 关键检索改善超过新增 Metadata/预算代价 |
| **RTL vs Dense/Quest** | 能否实现端侧净收益？ | PPA、共享列表构建成本、总真实流量、系统能量/Token 收益闭环；完成后判定 Gate 3 |

**不应提前声称：** A3 必然改善最差 Head、$b=8$ 必然近似逐页贪心、Mass 覆盖足以证明 RULER 精度、两质心一定优于 Quest、固定预算必然减少 LPDDR 字节、Head Mask 必然缩短关键路径。

**最终论文叙事（仅在 Gate 成立后使用）：** *基于 GQA 共享页的预算约束覆盖选择，在低资源硬件上通过有限前沿和批次修正实现质量—选择成本折中，并以真实 KV4/LPDDR 数据流证明 ASIC 价值。*

## 9. 相关工作与数据资源（研究边界）

- Quest — https://proceedings.mlr.press/v235/tang24l.html （Page Min/Max 上界、经典分页筛选）
- SPARTA — https://arxiv.org/abs/2610.34824 （Page Top-k + Token Top-p + 专用硬件，必须比较）
- LOCKS — https://arxiv.org/abs/2607.24555 （Page 摘要 Mass 与 GQA 共享选择，重要近邻）
- Double-P — https://arxiv.org/abs/2602.05191 （概率质量与簇摘要的层次选择）
- COBS — https://arxiv.org/abs/2607.09052 （紧凑二阶摘要与 Mass 估计）
- RULER — https://github.com/NVIDIA/RULER
- LongBench — https://github.com/THUDM/LongBench
- Qwen3-4B-Instruct-2507 — https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- Qwen2.5-7B-Instruct — https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- Llama-3.1-8B-Instruct — https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

> 上述相关工作用来界定创新边界；现有组级选页、覆盖优化与分批贪心并非天然全新。GLSVLSI 贡献需要由**具体低成本联合机制及可综合硬件实测/可信系统模型**支持，不能靠算法命名证明。
