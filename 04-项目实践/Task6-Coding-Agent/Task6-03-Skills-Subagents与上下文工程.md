---
tags: [LLM, Coding-Agent, Skills, Subagents, Context-Engineering]
aliases: [Task6 Skills 与 Subagents, Coding Agent 能力三层栈]
updated: 2026-09-18
---

# Task 6.3：Skills、Subagents 与上下文工程

> [!summary] 必须掌握
> Tool 提供动作，Skill 提供按需加载的过程知识，Subagent 提供独立上下文中的多步执行。它们都可能提升复杂任务能力，也都会引入额外 Token、延迟和协调成本。

返回总览：[[Task6-Coding-Agent学习索引]]

## 1. 三层能力栈

| 层 | 项目中的载体 | 解决的问题 | 不自动保证 |
|---|---|---|---|
| Tools | MCP Schema + RepoTools | 模型能够执行什么动作 | 动作选择正确、执行安全、任务完成 |
| Skills | `SKILL.md` | 一类任务应该怎样做 | 一定匹配、一定提高成功率 |
| Subagents | 独立消息历史 + 工具子集 | 隔离复杂子任务和上下文 | 更便宜、更快、更正确 |

Skill 不是简单注册 Tool。比如 `test-runner` Skill 可以告诉主模型先运行测试、如何根据失败缩小范围；真正执行 `run_tests` 的仍然是 Tool。

Subagent 也不是普通函数：它会单独调用模型、多轮使用工具、形成自己的 Usage 和停止原因，再返回摘要。

## 2. Skill 的文件结构

项目中的每个 Skill 使用：

```text
src/skills/<name>/SKILL.md
```

并包含 YAML Front Matter：

```yaml
---
name: test-runner
description: 当任务要求运行测试或诊断测试失败时加载
---
```

正文才是详细工作流。当前包括：

- `code-review`；
- `pr-description-writer`；
- `test-runner`。

## 3. 渐进式披露怎样实现

```text
扫描所有 SKILL.md
→ 只解析 name / description / path
→ 根据 Issue 匹配名称和描述关键词
→ 只读取最多 3 个命中 Skill 的正文
→ 用 <skill name="..."> 标签加入 System Prompt
```

`list_skills()` 不返回正文，因此仅做能力发现时不会把所有说明塞进上下文。`load(name)` 才读取指定正文，这就是 Progressive Disclosure。

标签的作用是标明内容边界和来源，帮助模型区分不同 Skill；标签本身不会赋予能力。

当前 Loader 会缓存元数据，必要时用 `force_refresh=True` 重新扫描。它还拒绝：

- 缺失或非法 YAML Front Matter；
- 空的 `name` / `description`；
- 重复 Skill 名称；
- 未知 Skill 名称。

## 4. 当前匹配器的工作方式与边界

匹配优先级大致为：

1. 任务文本直接包含 Skill 名称；
2. 任务文本包含完整 description；
3. 从 description 的正向条件提取词项，按命中长度计分；
4. 按得分和稳定目录顺序返回。

优点是实现简单、可解释、不需要额外模型请求。局限是：

- 同义词和复杂语义可能漏匹配；
- 描述写得过宽会误匹配；
- 多个 Skill 可能重复或冲突；
- 当前只截取前三个，没有更完整的冲突消解和 Token 预算器。

设计 Skill 描述时，应明确“何时加载”和“不适用于什么”，而不只是写“处理代码”。

### 4.1 Codex 的 Skill 匹配：模型决策与宿主预选分层

> [!info] 源码边界
> 以 2026-09-18 的 `openai/codex` HEAD `7498521` 为依据。Codex 当前正式路径不是先用 Embedding 或 BM25 唯一决定 Skill；隐式语义匹配主要交给模型，词法选择器仍在 shadow experiment 中。

#### 正式路径的完整数据流

```text
扫描各来源下的 SKILL.md
→ 解析 name / description / short_description / locator / authority
→ 按 Token 预算渲染 Available Skills Catalog
→ 作为 developer context 交给模型
→ 显式点名：宿主精确匹配并自动注入完整 SKILL.md
→ 未点名：模型根据 description 判断，再按需读取 SKILL.md
→ 仅加载被选 Skill 直接引用的 references / scripts / assets
```

Codex 把“显式选择”和“隐式语义匹配”分开：

- UI 的 Skill mention、`skill://...`、`SKILL.md` 路径按 locator 精确匹配；
- 文本中的 `$skill-name` 按已启用的 `name` 精确匹配；
- 重复项用 `authority + package` 去重，不仅看名字；
- 显式命中后，宿主读取完整正文，用 `<skill>...</skill>` 边界注入；
- 没有显式点名时，模型先看元数据 Catalog，自己决定是否通过文件读取或 `skills.read` 加载正文。

这样的核心不是“宿主替模型完成全部语义判断”，而是：

```text
确定性意图（用户点名） → 精确路由
模糊意图（任务语义） → 模型阅读 Catalog 后选择
大文档与资源             → 选中后才渐进加载
```

#### Codex 正在影子评测的廉价选择器

开源代码中已经有多种候选召回算法：

| 方法 | 技巧 | 主要解决的问题 |
|---|---|---|
| Weighted Lexical | Skill 名完整短语命中最高分，`name > short_description > description` | 便宜、可解释的关键词召回 |
| Fielded BM25 | 三个字段权重约为 `8 : 4 : 1`，考虑词频和稀有度 | 限制通用词导致的误匹配 |
| Character N-gram | 对字符片段计分 | 处理词形、拼写和缺少空格的匹配 |
| Multi-query Lexical | 按换行、句子和 `and / then / also` 拆分多个子查询 | 一条用户请求同时对应多个 Skill |
| Routing Card | 使用更明确的路由描述与依赖信息 | 减少只看短描述的歧义 |
| LRU | 提升近期在当前任务中真正使用过的 Skill | 利用多轮任务的连续性 |
| RRF / 融合排名 | 融合词法、字符、LRU 等多路排名 | 降低单个召回器偏差 |

但这些选择器目前的关键限制是：

> [!warning] Shadow 不等于生产决策
> 它们每轮预测最多 50 个候选，却不改变模型实际看到的 Catalog。宿主再观察模型真正读取或执行了哪个 Skill，记录 `hit`、`rank`、候选缩减率和耗时。这是用真实调用作为标签的在线影子评测，不是已接管路由的算法。

#### 可迁移的匹配技巧

1. **把 description 写成 Routing Card**：包含触发条件、输出类型和负向边界，不只写宽泛能力。
2. **显式意图优先确定性路由**：用户点名时不应再靠相似度猜测；重名时优先完整 locator。
3. **分离发现和加载**：长期只放短元数据，正文、脚本和资源必须在选中后才进入工作上下文。
4. **候选召回与最终决策分层**：当 Skill 很多时，先用廉价方法保召回，再让模型在小候选集上做语义取舍。
5. **评估实际调用而不是自报命中**：通过读取 `SKILL.md`、运行 Skill 脚本或 `skills.read` 的真实轨迹确定是否使用。
6. **同时测量质量和成本**：至少记录 Recall@K、MRR/命中排名、误加载率、Catalog Token 和额外延迟。
7. **多 Skill 只取最小覆盖集**：同时命中时要处理职责重叠、顺序与冲突，不是全部无脑注入。

#### 对当前 Task6 Loader 的启发

当前 Task6 是宿主用关键词评分后直接注入 Top 3 正文，在 Skill 少、目标可控时简单有效，但宿主过早承担了最终语义决策。更接近 Codex 的演进路径是：

```text
list_skills() 只返回紧凑 Catalog
→ 显式名称或 locator 精确匹配
→ Skill 很多时，用词法 / BM25 做可选候选召回
→ 将候选元数据给模型
→ 模型调用 load_skill(name) 按需读取正文
→ Trace 记录 matched / loaded / actually_used
→ 用实际调用评估召回器，再决定是否让它接管路由
```

安全边界不能因为匹配成功而放宽：Skill 只能指导如何做，Tool 白名单、路径限制和 Evaluator 仍必须由宿主强制执行。

源码定位：

- [Catalog 和渐进加载](https://github.com/openai/codex/blob/7498521d288b9b3b96ffba4eedf089d8d6e06a84/codex-rs/ext/skills/src/catalog_prompt.rs)
- [显式 Skill mention 精确匹配](https://github.com/openai/codex/blob/7498521d288b9b3b96ffba4eedf089d8d6e06a84/codex-rs/ext/skills/src/selection.rs)
- [选中后读取并注入正文](https://github.com/openai/codex/blob/7498521d288b9b3b96ffba4eedf089d8d6e06a84/codex-rs/ext/skills/src/extension.rs)
- [Shadow Selection Experiment](https://github.com/openai/codex/blob/7498521d288b9b3b96ffba4eedf089d8d6e06a84/codex-rs/ext/skills/src/shadow_selection_experiment/mod.rs)

## 5. Subagent 的独立性体现在哪里

`Subagent.run()` 创建全新的消息列表：

```text
System：角色、允许工具、不得修改仓库
User：聚焦任务 + 最小上下文
```

它不会直接继承主 Agent 的完整消息历史。项目还为它设置：

- 独立 `max_steps`；
- 独立 `max_tool_calls`；
- 受限工具集合；
- 独立 Trace 和 Usage；
- 只向主 Agent返回摘要与结构化测试证据。

当前 `TestRunnerSubagent` 只允许 `read_file` 和 `run_tests`，并要求先跑测试、失败后只查看测试输出或委托上下文指出的文件。它不能写代码，也不能自行遍历仓库。

`CodeSearchSubagent` 的规格允许 `list_files`、`search_text`、`read_file` 和 `git_diff`，但当前主循环实际接入的是测试子代理。

## 6. 强制委托与条件委托

### 强制委托

模型第一次动作必须调用 `delegate_test_runner` 获取初始失败证据，修改后再次委托完成最终验证；主 Agent 看不到直接 `run_tests`。

这能确保委托确实发生，适合验证 Subagent 链路，但不代表最佳策略。

### 条件委托

主 Agent 先直接运行测试。只有当前仓库状态出现真实测试失败，才允许把复杂诊断委托出去；最终验证仍由主 Agent执行。

这更接近生产策略，但如果任务简单，模型可能一次也不委托。此时条件组表现更好不能归因于 Subagent。

## 7. 为什么主上下文变短不等于总成本下降

总 Token 应计算所有模型请求：

$$
T_{total}=T_{main}+\sum_{j=1}^{n}T_{subagent,j}.
$$

还要包括：

- 主 Agent 构造委托的输入；
- Subagent 多轮工具调用；
- 返回摘要后主 Agent 的继续决策；
- 错误重试和协调。

S2 的强制委托组就是例子：主 Agent 步数减少，但加上子代理后总 Token 和时延明显上升。

## 8. S2/S3 给出的实际教训

在 5 个简单 Fixture、单轮实验中：

- Baseline、强制 Subagent、条件 Subagent 都是 5/5；
- 强制委托平均 Token 从 11,281.2 增到 18,874，增加约 67.3%；
- 强制委托平均时延从 9.906 秒增到 18.645 秒，增加约 88.2%；
- 条件组没有发生任何委托，因此不能用其较低成本证明 Subagent 有收益。

Skill 实验中：

- Baseline 与 Skill 都是 5/5；
- Match Rate 为 80%，四个任务命中 `test-runner`；
- Skill 平均 Token 增加 959，约 10.1%；
- 成功率不变。

正确结论是：当前任务太简单，成功率存在天花板，新增上下文和协调只体现为成本。不能推出 Skill/Subagent 永远无用。

## 9. 什么任务更适合使用它们

| 任务特征 | 更适合的机制 |
|---|---|
| 单文件、失败位置明确 | 主 Agent + Tools |
| 有稳定领域流程或团队规范 | Skill |
| 大量独立代码搜索 | Code Search Subagent |
| 测试日志很长、诊断相对独立 | Test Runner Subagent |
| 多个模块可并行调查 | 多 Subagent，但要处理共享状态冲突 |
| 修改和测试强依赖 | 保持顺序，不能盲目并行 |

Subagent 最适合边界清晰、上下文较重、输出可压缩的子任务。写操作通常应集中到主 Agent，减少并发冲突和责任不清。

## 10. 上下文窗口与 Agent 工作记忆

### 10.1 一次请求的上下文包含什么

模型窗口不只包含 Issue，还要容纳：

```text
System Prompt
+ Skill 正文（如果启用）
+ 用户 Issue
+ Tool Schema
+ 历史 Assistant Tool Calls
+ 历史 Tool Observations
+ 本轮预留输出
```

基本约束是：

$$
T_{input}+T_{output\_reserved}\le T_{context}.
$$

其中，$T_{context}$ 是服务端配置的最大模型长度。当前 Agent 每轮还固定预留 4096 个输出 Token，因此输入不能真的占满整个窗口。

模型本身通常不会在窗口满时自动总结历史。服务端可能：

1. 返回 `context_length_exceeded` 一类 HTTP 400；
2. 在显式启用时从左侧或右侧机械截断；
3. 输入尚可接受、但输出达到限制时以 `finish_reason=length` 结束。

vLLM 支持 `truncate_prompt_tokens` 和 `truncation_side`，但机械截断不适合作为 Agent 的主要策略：左截断可能丢失 System Prompt 和 Issue，右截断可能丢失最新 Patch 和测试，任意截断还可能破坏 `assistant(tool_calls) -> tool(tool_call_id)` 配对。

### 10.2 累计 Token 不等于单次上下文

S4 失败样本中：

| 实例 | 报告累计 Input Token | 最后一轮 Input Token | 实际停止原因 |
|---|---:|---:|---|
| `astropy__astropy-14182` | 447,213 | 48,642 | `max_tool_calls` |
| `astropy__astropy-14365` | 412,854 | 36,772 | `max_tool_calls` |

四十多万是各轮请求之和，不是模型一次收到的长度：

$$
T_{total}=\sum_{i=1}^{n}T_i.
$$

但因为当前每轮都重发完整历史，$T_i$ 会持续增长，导致累计成本接近二次增长。这两题没有先撞上服务端的硬上限，而是先用完 24 次工具预算；上下文膨胀仍然造成了成本上升、时延增加和注意力稀释。

### 10.3 当前实现的缺口

当前 `messages` 只追加不压缩，下一轮会再次发送旧的 Tool Call 和 Observation。同时：

- `read_file` 一次最多读取 128,000 Bytes，不支持行范围；
- `search_text` 最多返回 200 条匹配；
- 命令和测试输出上限为 12,000 字符；
- 没有 Token 预估、Working Memory、历史 Compaction 或上下文溢出恢复；
- HTTP 400 不在普通可重试范围，上下文过长会直接结束为 `model_error`。

完整 Trace 适合保存到磁盘供评测与复盘，但不应等同于每一轮都要喂给模型的工作上下文。

### 10.4 上下文管理的优化顺序

#### 1. 局部读取代替整文件读取

将：

```python
read_file(path)
```

扩展为：

```python
read_file(path, start_line=60, end_line=120)
read_file_around(path, line=71, context_lines=30)
```

让 `search_text` 先返回行号，再只读命中位置附近。这是当前项目投入产出比最高的优化。

#### 2. 分离完整 Trace 和模型工作上下文

工具的完整输出可保存为 Artifact，模型只接收：

```json
{
  "artifact_id": "read-017",
  "summary": "RST.__init__ 没有透传 header_rows，父类 FixedWidth 已支持",
  "excerpt": "def __init__(self): ...",
  "path": "astropy/io/ascii/rst.py",
  "line": 73
}
```

需要时再通过 `artifact_id` 或路径继续读取，而不是在后续每轮重发整个文件。

#### 3. 建立结构化 Working Memory

在输入达到窗口的约 65%–75% 时主动 Compaction，保留：

```text
System Prompt
+ 原始 Issue
+ 结构化 Working Memory
+ 最近 3–5 轮完整交互
```

Working Memory 至少应记录：

```text
已知事实 / 相关文件与行号 / 当前假设
已做修改 / 当前 Patch / 最新测试证据
失败尝试 / 环境异常 / 剩余预算 / 下一步
```

压缩时必须按完整交互组删除，不能只删 Assistant Tool Call 或只留 Tool Result，否则会破坏 Tool Calling 消息协议。

#### 4. 为上下文溢出做一次专用恢复

```text
收到 context_length_exceeded
→ 构建 Working Memory
→ 压缩成对的旧交互
→ 重新估算 Token
→ 只重试一次
```

不能当作普通网络重试，因为原样重发过长 Prompt 永远不会成功。

#### 5. 同时管理工具阶段预算

可将 24 次调用软划分为：

```text
定位与读取：最多 8 次
假设与验证：最多 4 次
修改与纠错：至少预留 6 次
测试与收尾：至少预留 6 次
```

连续只读且没有形成 Patch 时，应要求模型总结当前证据、提出最小修复或说明唯一缺失信息，避免像 S4 `14182` 一样把 24 次全部用于搜索。

#### 6. 压缩测试输出并分类失败

模型优先需要：

```text
失败阶段 / 失败测试名 / 根异常
断言失败 vs 导入或配置失败
是否真正执行了业务测试
完整日志的 Artifact 引用
```

这能防止 Agent 把本地依赖缺失误当业务 Bug，像 S4 `14365` 一样去修改无关的版本文件。

### 10.5 常见误区

- **更多 KV Cache 不等于上下文无限**：KV Cache 容量影响可服务的序列长度与并发，但不会自动改变服务配置的 `max_model_len`。
- **Prefix Cache 只优化重复计算**：它可降低相同前缀的延迟，却不减少 Token 数，也不消除无关历史的注意力干扰。
- **扩大 `max_model_len` 只是推迟失败**：如果 Agent 仍无限堆积原始 Observation，64K 换成 128K 可能只是多浪费一倍 Token。
- **不能只按字符数估算**：工具 Schema、Chat Template 和特殊 Token 都会计入。应尽量使用实际模型 Tokenizer 或服务端 Tokenize 接口。

### 10.6 其他能力的后续优化

- Skill Match 加入负向条件、冲突处理和 Token 预算；
- 根据任务复杂度、测试日志长度和跨文件程度决定是否委托；
- 给 Subagent 返回证据引用，而不仅是自然语言摘要；
- 评测“每次成功成本”，把失败和子代理开销全部计入分子。

## 11. 自测

1. Skill 与 Tool 的本质区别是什么？
2. 为什么 `list_skills()` 不应直接返回全部正文？
3. Subagent 的“独立上下文”具体体现在哪里？
4. 条件委托组更快，为什么不能证明 Subagent 提速？
5. 强制委托为什么适合实验链路验证，却不一定适合生产？
6. 什么样的子任务值得委托？
7. 报告中累计 45 万 Input Token，是否表示模型一次收到了 45 万 Token？
8. 为什么不应依赖服务端从左侧自动截断 Agent 历史？
9. 完整 Trace 为什么不应全部保留在模型工作上下文中？
10. Prefix Cache 和更大 KV Cache 能否替代 Context Compaction？
11. Codex 的隐式 Skill 匹配目前由谁完成最终判断？
12. Codex 源码已有 BM25 等选择器，为什么不能说它们已接管生产路由？
13. 如果 Task6 要向 Codex 靠拢，SkillLoader 最重要的结构变化是什么？

### 自测标准答案

> [!success]- 标准答案：1. 过程知识与执行能力
> Tool 是有 Schema 的可执行动作；Skill 是告诉模型何时以及如何组织动作的说明。Skill 最终仍需调用 Tool 或生成结果。

> [!success]- 标准答案：2. 控制上下文并减少干扰
> 先用短元数据发现能力，只加载命中正文，可避免无关 Skill 占用 Token、互相冲突或稀释当前任务指令。

> [!success]- 标准答案：3. 新消息历史、独立预算和工具子集
> 子代理只接收聚焦任务与显式上下文，不继承主 Agent 全部历史；它独立调用模型和工具，再返回摘要、证据与 Usage。

> [!success]- 标准答案：4. 本轮委托次数为零
> 条件组实际上退化为另一轮主 Agent 运行；较低时延可能来自远程服务或轨迹差异，不能归因于未发生的委托。

> [!success]- 标准答案：5. 它改变了多个因素
> 强制策略保证每题调用子代理，便于验证功能和测量开销，但也人为增加协调与模型调用，不代表真实任务中的最优路由。

> [!success]- 标准答案：6. 独立、复杂且可压缩
> 适合需要大量专门上下文、与主修改链路耦合较低、结果能用摘要和证据表达的子任务；简单或强依赖任务通常直接处理更省成本。

> [!success]- 标准答案：7. 不是，这是多轮请求之和
> 每轮都有自己的 Input Token，累计值是它们之和。S4 `14182` 最后一轮约 48,642，累计才是 447,213。

> [!success]- 标准答案：8. 可能删掉任务约束并破坏工具协议
> 左截断可能丢失 System Prompt、Issue 和早期关键证据；不按完整交互组处理还可能留下孤立 Tool Result。应由 Agent 保留结构化记忆后主动压缩。

> [!success]- 标准答案：9. 评测证据与当前决策信息的需求不同
> 完整 Trace 适合审计和复现；模型只需要当前目标、关键文件片段、Patch、最新测试和下一步。分离两者能降低重复 Token 和无关干扰。

> [!success]- 标准答案：10. 不能
> Prefix Cache 只减少重复前缀的计算时间，不减少 Token；更大 KV Cache 不会自动扩大 `max_model_len`。它们都不会替代对旧 Observation 的结构化压缩。

> [!success]- 标准答案：11. 模型根据 Catalog 做语义判断
> 宿主先把 Skill 名称、描述和 locator 放入 developer context。用户未显式点名时，模型判断任务是否匹配某个 description，再读取该 Skill 的正文。

> [!success]- 标准答案：12. 它们只在 Shadow Mode 中预测
> 这些选择器会产生候选排名并记录命中、排名和成本，但不改变模型看到的 Catalog。它们的预测要与模型真正读取或执行的 Skill 比较，暂未成为强制路由。

> [!success]- 标准答案：13. 把“宿主直接注入 Top K 正文”改成“候选召回 + 模型按需加载”
> Loader 先提供紧凑 Catalog，显式点名做精确匹配，Skill 很多时才用廉价选择器召回候选，再由模型通过 `load_skill` 加载正文。同时保留 Tool 权限与 Evaluator 的宿主强制边界。
