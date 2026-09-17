---
tags: [LLM, Coding-Agent, Skills, Subagents, Context-Engineering]
aliases: [Task6 Skills 与 Subagents, Coding Agent 能力三层栈]
updated: 2026-09-17
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

## 10. 后续可做的上下文优化

- 用结构化工作记忆保存目标、假设、已读文件、当前 Patch、失败测试和剩余预算；
- 对早期完整工具输出做 Compaction，但保留路径、行号与结论依据；
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
