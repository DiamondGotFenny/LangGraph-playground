你现在是"**Claude sdk Waiter Agent 运行日志评估员**"。请基于仓库中的实现与运行日志，生成一份结构化评估报告（Markdown）。

---

## 📥 输入文件（必须读取）

| 文件                                             | 用途               | 关注重点                                                                                                             |
| ------------------------------------------------ | ------------------ | -------------------------------------------------------------------------------------------------------------------- |
| `waiter_claude_agent_sdk_demo.py`                | 实现代码           | tools 定义（WHEN TO CALL/DO NOT CALL）、反幻觉协议、State Schema、\_postprocess_node 压缩策略、\_agent_node 注入逻辑 |
| `waiter_claude_agent_sdk.log`                    | 运行日志           | tool calls 序列、LLM input(len=…)、tool output JSON、Conversation summary so far、AI response                        |
| `waiter_agent_accessment_golden_dataset_v2.json` | 测试用例（如存在） | expected_behavior 字段，用于逐 case 对照                                                                             |

---

## 🎯 评估目标（6 大维度）

### 1. 回答质量（Accuracy & Consistency）

- AI 回答是否准确、一致
- 是否有产生**幻觉** （即与资料事实不符，创造不存在的内容）
- 是否遵循**Tool Result Processing Protocol**（尤其是错误处理）

### 2. 工具调用正确性（Tool Call Correctness）

- 该调用的工具是否调用（参照各 tool 的 WHEN TO CALL）
- 不该调用的是否乱用（参照各 tool 的 DO NOT CALL）
- tool result 是否被正确遵循（尤其是 insufficient_stock、partially_fulfilled）

### 3. 工具调用效率（Tool Call Efficiency）

- 是否存在冗余/重复调用（如菜单已在 context 中仍调用）
- 是否存在无必要循环
- 是否利用 summary 减少调用

### 4. Summary 质量（Summary Quality）

- 是否覆盖关键事实（flags: restaurant_info_provided, food_menu_shown, drinks_menu_shown; prices）
- 是否引入重复信息
- summary 是否可复用以减少工具调用

### 5. Memory/State 更新准确性（State Consistency）

- 订单状态是否与 tool 结果一致
- 库存是否正确扣减
- 是否存在"话术说改了但 state 没改"的错配

### 6. 对话压缩效果（Compression Efficiency）

- 是否按 `_postprocess_node` 策略裁剪历史（保留最近 5 轮）
- Tool traces 是否从保留窗口移除
- LLM input(len=…) 是否稳定，压缩收益是否被重复注入抵消

---

## 📋 输出要求（必须遵守）

### 基本格式

- 输出**中文报告**，**Markdown 格式**
- 报告文件**必须新建**，文件名格式：
  ```
  waiter_agent_sdk_log_eval_{YYYYMMDD_HHMMSS}.md
  ```
- 报告内需写明：
  - **评估所用模型名**（即你自己）
  - **报告输出途径** 报告保存到 "evaluation_reports" 文件夹

---

## 📑 必须包含的报告章节（按顺序，表格形式）

### 1️⃣ 📋 测试概览

必须包含如下格式的表格：

```markdown
## 📋 测试概览

| 指标                    | 值                             |
| ----------------------- | ------------------------------ |
| **测试日期**            | YYYY-MM-DD HH:MM:SS - HH:MM:SS |
| **评估所用模型**        | 你的模型名                     |
| **Golden Dataset 条目** | X 条                           |
| **总 Token 使用**       | input=X / output=Y / total=Z   |
| **运行时长**            | ≈ X 分 Y 秒                    |
```

---

### 2️⃣ 📊 总体评估摘要

#### 2.1 通过率统计

```markdown
### ✅ 通过项 (X/Y = Z%)
```

#### 2.2 逐 Case 结果表（必须列出所有 case）

```markdown
| ID  | 测试场景       | 状态 | 说明                 |
| --- | -------------- | ---- | -------------------- |
| 1   | 问候与座位确认 | ✅   | 正确欢迎客户         |
| 2   | ...            | ⚠️   | 部分通过（具体原因） |
| 3   | ...            | ❌   | 失败（具体原因）     |
```

**状态 emoji 规范**：

- ✅ = 完全通过
- ⚠️ = 部分通过（需在"说明"中解释）
- ❌ = 失败

---

### 3️⃣ 🔍 详细分析

#### 3.1 工具调用行为分析

##### 正确的工具调用模式 ✅

```markdown
| 场景         | 工具链                           | 结果                           |
| ------------ | -------------------------------- | ------------------------------ |
| 餐厅信息查询 | `get_restaurant_info`            | 仅调用一次，未自动调用菜单工具 |
| 订单创建     | `create_order` → `process_order` | 正确的两步流程                 |
```

##### 问题工具调用模式 ⚠️/❌（如有）

```markdown
| 场景         | 实际调用           | 问题描述            | 预期行为                  |
| ------------ | ------------------ | ------------------- | ------------------------- |
| 菜单重复调用 | `get_food_menu` x3 | 菜单已在 context 中 | 应引用 summary 或最近消息 |
```

#### 3.2 工具调用效率统计

```markdown
| 工具名称        | 调用次数 | 占比     | 备注         |
| --------------- | -------- | -------- | ------------ |
| get_food_menu   | 8        | 19%      | 可能存在冗余 |
| get_drinks_menu | 3        | 7%       |              |
| create_order    | 12       | 29%      |              |
| process_order   | 10       | 24%      |              |
| ...             | ...      | ...      |              |
| **总计**        | **N**    | **100%** |              |
```

```markdown
| 统计项 | 值          |
| ------ | ----------- |
| 均值   | X.X         |
| 中位数 | X           |
| 最大值 | X (Case #Y) |
| 最小值 | X (Case #Y) |
```

#### 3.3 LLM Input 长度分析

```markdown
| 统计项   | 值（消息数）  |
| -------- | ------------- |
| 均值     | X.X           |
| 中位数   | X             |
| 最大值   | X             |
| 最小值   | X             |
| 最常见值 | X (出现 N 次) |
```

#### 3.4 反幻觉协议遵守情况

```markdown
| 协议条款     | 遵守情况 | 典型案例 |
| ------------ | -------- | -------- |
| 菜单项验证   | ✅/⚠️/❌ | Case #X  |
| 价格确认     | ✅/⚠️/❌ | Case #X  |
| 库存不足处理 | ✅/⚠️/❌ | Case #X  |
| 订单状态信任 | ✅/⚠️/❌ | Case #X  |
| 餐厅信息查询 | ✅/⚠️/❌ | Case #X  |
```

#### 3.5 Memory/State 更新分析

```markdown
| Case ID | State 类型 | 工具返回    | 实际 State  | 是否一致 |
| ------- | ---------- | ----------- | ----------- | -------- |
| #6      | 订单状态   | fulfilled   | fulfilled   | ✅       |
| #13     | 库存扣减   | 5 available | 5 remaining | ✅       |
| #16     | 配菜替换   | ...         | ...         | ⚠️       |
```

#### 3.6 Summary 质量分析

```markdown
| 评估项            | 结果     | 说明                                                         |
| ----------------- | -------- | ------------------------------------------------------------ |
| 关键 flags 覆盖   | ✅/❌    | food_menu_shown, drinks_menu_shown, restaurant_info_provided |
| prices 记录完整性 | ✅/❌    | 已报价 X 项，summary 记录 Y 项                               |
| 重复信息          | 有/无    | 具体说明                                                     |
| 可复用性          | 高/中/低 | 是否有效减少工具调用                                         |
```

#### 3.7 对话压缩分析

```markdown
| 指标                    | 值                  |
| ----------------------- | ------------------- |
| 保留轮数策略            | keep_rounds=5       |
| Tool traces 是否清理    | ✅/❌               |
| 压缩触发次数            | X                   |
| 平均 LLM input 长度变化 | 压缩前 X → 压缩后 Y |
```

---

### 4️⃣ 🎯 关键用例复盘

**要求**：

- 必须至少 **3 个 case** 的详细分析
- 必须包含**高风险 case**（如存在），例如：
  - #13 库存不足检测
  - #16 配菜替换（状态错配风险）
  - #19 账单计算

每个 case 的分析格式：

```markdown
#### Case #X: [场景名称]

**用户输入**：

> "原始用户输入"

**工具调用序列**：

1. `tool_name(args)` → 返回摘要
2. `tool_name(args)` → 返回摘要

**AI 响应**：

> "AI 的实际响应"

**与 expected_behavior 对比**：

- 预期：...
- 实际：...
- 差异：...

**问题分析**（如有）：

- 问题描述
- 根因定位（指向代码）
```

---

### 5️⃣ 🐛 根因定位

```markdown
| 问题类型     | 涉及函数/代码位置              | 具体描述                              |
| ------------ | ------------------------------ | ------------------------------------- |
| 工具冗余调用 | `_agent_node` 第 2233-2248 行  | trim_messages 未考虑 summary 中的信息 |
| State 不一致 | `create_order` 第 1479-1800 行 | 特定条件下未正确更新 items            |
| Summary 遗漏 | `_extract_conversation_facts`  | 未提取某类事实                        |
```

---

### 6️⃣ 📈 评分（10 分制）

```markdown
| 评估维度          | 得分     | 依据         |
| ----------------- | -------- | ------------ |
| 回答质量          | X/10     | 简短依据     |
| 工具调用正确性    | X/10     | 简短依据     |
| 工具调用效率      | X/10     | 简短依据     |
| Summary 质量      | X/10     | 简短依据     |
| Memory/State 更新 | X/10     | 简短依据     |
| 对话压缩          | X/10     | 简短依据     |
| **综合分**        | **X/10** | **综合评价** |
```

## 重要！注意：只要最终账单的项目，数量和总金额于预期的 golden dataset 里面的不符，就意味着有重大缺陷，应该直接判定为不及格，6 分以下。

### 7️⃣ 💡 改进建议

```markdown
| 优先级 | 问题     | 建议方案 | 预期收益     | 涉及代码    |
| ------ | -------- | -------- | ------------ | ----------- |
| 🔴 高  | 问题描述 | 具体方案 | 量化收益预估 | 函数/文件名 |
| 🟡 中  | ...      | ...      | ...          | ...         |
| 🟢 低  | ...      | ...      | ...          | ...         |
```

---

## 📐 分析方法（建议按此执行）

### Step 1: 代码提取

从 `waiter_claude_agent_sdk.log` 提取：

1. **Tools 列表**及各 tool 的：
   - `WHEN TO CALL` 条件
   - `DO NOT CALL` 条件
   - 返回值格式

2. **是否有幻觉**

3. **Tool Result Processing**的错误处理流程

4. **State Schema**：
   - `RestaurantOrderState` 字段
   - `_init_state_node` 初始化逻辑
   - `_postprocess_node` 的 `keep_rounds` 和 tool trace 清理逻辑

5. **Summary 逻辑**：
   - `_extract_conversation_facts` 提取的 facts 类型
   - `_format_conversation_summary` 输出格式
   - `_parse_conversation_summary` 解析逻辑

### Step 2: 日志结构化

从 `waiter_react_agent.log` 提取并结构化：

1. **逐 Case 提取**：
   - `user_query`（HumanMessage）
   - `AI response`（LLM OUTPUT 部分）
   - `tool_call` 序列（名称+参数+返回值）
   - `LLM input (len=…)` 数值

2. **关键 Tool Output 提取**（完整 JSON）：
   - `process_order` 的 `order_status`、`items[].status`、`stock`
   - `cashier_calculate_total` 的金额
   - `check_payment` 的结果

3. **Summary 片段**：
   - `Conversation summary so far:` 后的内容
   - 每次 summary 的变化

4. **State 注入**：
   - `Current active order (structured, authoritative):` JSON

### Step 3: Golden Dataset 对照

如存在 `waiter_agent_accessment_golden_dataset.json`：

1. 遍历每个 entry 的 `expected_behavior`
2. 对比实际：
   - 工具调用是否符合预期
   - 回答是否满足预期
   - 是否有多余调用

### Step 4: 统计汇总

1. **Tool Call 统计**：
   - 按工具类型计数
   - 每 case 的 tool call 数
   - 计算均值/中位数/max/min

2. **LLM Input 统计**：
   - 提取所有 `len=X` 数值
   - 计算分布

3. **通过率**：
   - 完全通过 / 部分通过 / 失败 的数量

---

## ⚠️ 注意事项

1. **不要修改除报告以外的任何文件**
2. 所有表格必须使用标准 Markdown 语法
3. 引用代码时使用函数名或行号范围，不要大段粘贴
4. 对于模糊或边界情况，给出你的判断依据
5. 评分必须有具体依据，避免主观臆断

---

## 📝 报告模板快速参考

```markdown
# LangGraph Waiter Agent 评估报告

## 📋 测试概览

（表格）

---

## 📊 总体评估摘要

### ✅ 通过项 (X/Y = Z%)

（逐 case 表格）

---

## 🔍 详细分析

### 1. 工具调用行为分析

### 2. 工具调用效率统计

### 3. LLM Input 长度分析

### 4. 反幻觉情况

### 5. Memory/State 更新分析

### 6. Summary 质量分析

### 7. 对话压缩分析

---

## 🎯 关键用例复盘

（至少 3 个 case 详细分析）

---

## 🐛 根因定位

（问题 → 代码位置 表格）

---

## 📈 评分（10 分制）

（评分表格）

---

## 💡 改进建议

（按优先级排列的表格）

---

_报告生成时间：YYYY-MM-DD HH:MM:SS_
_评估所用模型：xxx_
```

---

# LangGraph Waiter Agent 运行日志评估 Prompt

你现在是"**LangGraph Waiter Agent 运行日志评估员**"。请基于仓库中的实现与运行日志，生成一份结构化评估报告（Markdown）。

---

## 📥 输入文件（必须读取）

| 文件                                          | 用途               | 关注重点                                                                                                             |
| --------------------------------------------- | ------------------ | -------------------------------------------------------------------------------------------------------------------- |
| `waiter_react_agent.py`                       | 实现代码           | tools 定义（WHEN TO CALL/DO NOT CALL）、反幻觉协议、State Schema、\_postprocess_node 压缩策略、\_agent_node 注入逻辑 |
| `waiter_react_agent.log`                      | 运行日志           | tool calls 序列、LLM input(len=…)、tool output JSON、Conversation summary so far、AI response                        |
| `waiter_agent_accessment_golden_dataset.json` | 测试用例（如存在） | expected_behavior 字段，用于逐 case 对照                                                                             |

---

## 🎯 评估目标（6 大维度）

### 1. 回答质量（Accuracy & Consistency）

- AI 回答是否准确、一致
- 是否违反**反幻觉协议**（ANTI-HALLUCINATION PROTOCOL）
- 是否遵循**Tool Result Processing Protocol**（尤其是错误处理）

### 2. 工具调用正确性（Tool Call Correctness）

- 该调用的工具是否调用（参照各 tool 的 WHEN TO CALL）
- 不该调用的是否乱用（参照各 tool 的 DO NOT CALL）
- tool result 是否被正确遵循（尤其是 insufficient_stock、partially_fulfilled）

### 3. 工具调用效率（Tool Call Efficiency）

- 是否存在冗余/重复调用（如菜单已在 context 中仍调用）
- 是否存在无必要循环
- 是否利用 summary 减少调用

### 4. Summary 质量（Summary Quality）

- 是否覆盖关键事实（flags: restaurant_info_provided, food_menu_shown, drinks_menu_shown; prices）
- 是否引入重复信息
- summary 是否可复用以减少工具调用

### 5. Memory/State 更新准确性（State Consistency）

- 订单状态是否与 tool 结果一致
- 库存是否正确扣减
- 是否存在"话术说改了但 state 没改"的错配

### 6. 对话压缩效果（Compression Efficiency）

- 是否按 `_postprocess_node` 策略裁剪历史（保留最近 5 轮）
- Tool traces 是否从保留窗口移除
- LLM input(len=…) 是否稳定，压缩收益是否被重复注入抵消

---

## 📋 输出要求（必须遵守）

### 基本格式

- 输出**中文报告**，**Markdown 格式**
- 报告文件**必须新建**，文件名格式：
  ```
  waiter_react_agent_log_eval_{YYYYMMDD_HHMMSS}_{日志运行模型}.md
  ```
- 报告内需写明：
  - **日志运行模型**（从 log 中解析，如 `gemini-3-flash-preview`）
  - **评估所用模型名**（即你自己）
  - **报告输出途径** 报告保存到 "evaluation_reports" 文件夹

---

## 📑 必须包含的报告章节（按顺序，表格形式）

### 1️⃣ 📋 测试概览

必须包含如下格式的表格：

```markdown
## 📋 测试概览

| 指标                    | 值                                       |
| ----------------------- | ---------------------------------------- |
| **测试日期**            | YYYY-MM-DD HH:MM:SS - HH:MM:SS           |
| **日志运行模型**        | 从 log 解析（如 gemini-3-flash-preview） |
| **评估所用模型**        | 你的模型名                               |
| **Golden Dataset 条目** | X 条                                     |
| **总 Token 使用**       | input=X / output=Y / total=Z             |
| **运行时长**            | ≈ X 分 Y 秒                              |
```

---

### 2️⃣ 📊 总体评估摘要

#### 2.1 通过率统计

```markdown
### ✅ 通过项 (X/Y = Z%)
```

#### 2.2 逐 Case 结果表（必须列出所有 case）

```markdown
| ID  | 测试场景       | 状态 | 说明                 |
| --- | -------------- | ---- | -------------------- |
| 1   | 问候与座位确认 | ✅   | 正确欢迎客户         |
| 2   | ...            | ⚠️   | 部分通过（具体原因） |
| 3   | ...            | ❌   | 失败（具体原因）     |
```

**状态 emoji 规范**：

- ✅ = 完全通过
- ⚠️ = 部分通过（需在"说明"中解释）
- ❌ = 失败

---

### 3️⃣ 🔍 详细分析

#### 3.1 工具调用行为分析

##### 正确的工具调用模式 ✅

```markdown
| 场景         | 工具链                           | 结果                           |
| ------------ | -------------------------------- | ------------------------------ |
| 餐厅信息查询 | `get_restaurant_info`            | 仅调用一次，未自动调用菜单工具 |
| 订单创建     | `create_order` → `process_order` | 正确的两步流程                 |
```

##### 问题工具调用模式 ⚠️/❌（如有）

```markdown
| 场景         | 实际调用           | 问题描述            | 预期行为                  |
| ------------ | ------------------ | ------------------- | ------------------------- |
| 菜单重复调用 | `get_food_menu` x3 | 菜单已在 context 中 | 应引用 summary 或最近消息 |
```

#### 3.2 工具调用效率统计

```markdown
| 工具名称        | 调用次数 | 占比     | 备注         |
| --------------- | -------- | -------- | ------------ |
| get_food_menu   | 8        | 19%      | 可能存在冗余 |
| get_drinks_menu | 3        | 7%       |              |
| create_order    | 12       | 29%      |              |
| process_order   | 10       | 24%      |              |
| ...             | ...      | ...      |              |
| **总计**        | **N**    | **100%** |              |
```

```markdown
| 统计项 | 值          |
| ------ | ----------- |
| 均值   | X.X         |
| 中位数 | X           |
| 最大值 | X (Case #Y) |
| 最小值 | X (Case #Y) |
```

#### 3.3 LLM Input 长度分析

```markdown
| 统计项   | 值（消息数）  |
| -------- | ------------- |
| 均值     | X.X           |
| 中位数   | X             |
| 最大值   | X             |
| 最小值   | X             |
| 最常见值 | X (出现 N 次) |
```

#### 3.4 反幻觉协议遵守情况

```markdown
| 协议条款     | 遵守情况 | 典型案例 |
| ------------ | -------- | -------- |
| 菜单项验证   | ✅/⚠️/❌ | Case #X  |
| 价格确认     | ✅/⚠️/❌ | Case #X  |
| 库存不足处理 | ✅/⚠️/❌ | Case #X  |
| 订单状态信任 | ✅/⚠️/❌ | Case #X  |
| 餐厅信息查询 | ✅/⚠️/❌ | Case #X  |
```

#### 3.5 Memory/State 更新分析

```markdown
| Case ID | State 类型 | 工具返回    | 实际 State  | 是否一致 |
| ------- | ---------- | ----------- | ----------- | -------- |
| #6      | 订单状态   | fulfilled   | fulfilled   | ✅       |
| #13     | 库存扣减   | 5 available | 5 remaining | ✅       |
| #16     | 配菜替换   | ...         | ...         | ⚠️       |
```

#### 3.6 Summary 质量分析

```markdown
| 评估项            | 结果     | 说明                                                         |
| ----------------- | -------- | ------------------------------------------------------------ |
| 关键 flags 覆盖   | ✅/❌    | food_menu_shown, drinks_menu_shown, restaurant_info_provided |
| prices 记录完整性 | ✅/❌    | 已报价 X 项，summary 记录 Y 项                               |
| 重复信息          | 有/无    | 具体说明                                                     |
| 可复用性          | 高/中/低 | 是否有效减少工具调用                                         |
```

#### 3.7 对话压缩分析

```markdown
| 指标                    | 值                  |
| ----------------------- | ------------------- |
| 保留轮数策略            | keep_rounds=5       |
| Tool traces 是否清理    | ✅/❌               |
| 压缩触发次数            | X                   |
| 平均 LLM input 长度变化 | 压缩前 X → 压缩后 Y |
```

---

### 4️⃣ 🎯 关键用例复盘

**要求**：

- 必须至少 **3 个 case** 的详细分析
- 必须包含**高风险 case**（如存在），例如：
  - #13 库存不足检测
  - #16 配菜替换（状态错配风险）
  - #19 账单计算

每个 case 的分析格式：

```markdown
#### Case #X: [场景名称]

**用户输入**：

> "原始用户输入"

**工具调用序列**：

1. `tool_name(args)` → 返回摘要
2. `tool_name(args)` → 返回摘要

**AI 响应**：

> "AI 的实际响应"

**与 expected_behavior 对比**：

- 预期：...
- 实际：...
- 差异：...

**问题分析**（如有）：

- 问题描述
- 根因定位（指向代码）
```

---

### 5️⃣ 🐛 根因定位

```markdown
| 问题类型     | 涉及函数/代码位置              | 具体描述                              |
| ------------ | ------------------------------ | ------------------------------------- |
| 工具冗余调用 | `_agent_node` 第 2233-2248 行  | trim_messages 未考虑 summary 中的信息 |
| State 不一致 | `create_order` 第 1479-1800 行 | 特定条件下未正确更新 items            |
| Summary 遗漏 | `_extract_conversation_facts`  | 未提取某类事实                        |
```

---

### 6️⃣ 📈 评分（10 分制）

```markdown
| 评估维度          | 得分     | 依据         |
| ----------------- | -------- | ------------ |
| 回答质量          | X/10     | 简短依据     |
| 工具调用正确性    | X/10     | 简短依据     |
| 工具调用效率      | X/10     | 简短依据     |
| Summary 质量      | X/10     | 简短依据     |
| Memory/State 更新 | X/10     | 简短依据     |
| 对话压缩          | X/10     | 简短依据     |
| **综合分**        | **X/10** | **综合评价** |
```

## 重要！注意：只要最终账单的项目，数量和总金额于预期的 golden dataset 里面的不符，就意味着有重大缺陷，应该直接判定为不及格，6 分以下。

### 7️⃣ 💡 改进建议

```markdown
| 优先级 | 问题     | 建议方案 | 预期收益     | 涉及代码    |
| ------ | -------- | -------- | ------------ | ----------- |
| 🔴 高  | 问题描述 | 具体方案 | 量化收益预估 | 函数/文件名 |
| 🟡 中  | ...      | ...      | ...          | ...         |
| 🟢 低  | ...      | ...      | ...          | ...         |
```

---

## 📐 分析方法（建议按此执行）

### Step 1: 代码提取

从 `waiter_react_agent.py` 提取：

1. **Tools 列表**及各 tool 的：
   - `WHEN TO CALL` 条件
   - `DO NOT CALL` 条件
   - 返回值格式

2. **反幻觉协议**（ANTI-HALLUCINATION PROTOCOL）的 5 个条款

3. **Tool Result Processing Protocol**的错误处理流程

4. **State Schema**：
   - `RestaurantOrderState` 字段
   - `_init_state_node` 初始化逻辑
   - `_postprocess_node` 的 `keep_rounds` 和 tool trace 清理逻辑

5. **Summary 逻辑**：
   - `_extract_conversation_facts` 提取的 facts 类型
   - `_format_conversation_summary` 输出格式
   - `_parse_conversation_summary` 解析逻辑

### Step 2: 日志结构化

从 `waiter_react_agent.log` 提取并结构化：

1. **逐 Case 提取**：
   - `user_query`（HumanMessage）
   - `AI response`（LLM OUTPUT 部分）
   - `tool_call` 序列（名称+参数+返回值）
   - `LLM input (len=…)` 数值

2. **关键 Tool Output 提取**（完整 JSON）：
   - `process_order` 的 `order_status`、`items[].status`、`stock`
   - `cashier_calculate_total` 的金额
   - `check_payment` 的结果

3. **Summary 片段**：
   - `Conversation summary so far:` 后的内容
   - 每次 summary 的变化

4. **State 注入**：
   - `Current active order (structured, authoritative):` JSON

### Step 3: Golden Dataset 对照

如存在 `waiter_agent_accessment_golden_dataset.json`：

1. 遍历每个 entry 的 `expected_behavior`
2. 对比实际：
   - 工具调用是否符合预期
   - 回答是否满足预期
   - 是否有多余调用

### Step 4: 统计汇总

1. **Tool Call 统计**：
   - 按工具类型计数
   - 每 case 的 tool call 数
   - 计算均值/中位数/max/min

2. **LLM Input 统计**：
   - 提取所有 `len=X` 数值
   - 计算分布

3. **通过率**：
   - 完全通过 / 部分通过 / 失败 的数量

---

## ⚠️ 注意事项

1. **不要修改除报告以外的任何文件**
2. 所有表格必须使用标准 Markdown 语法
3. 引用代码时使用函数名或行号范围，不要大段粘贴
4. 对于模糊或边界情况，给出你的判断依据
5. 评分必须有具体依据，避免主观臆断

---

## 📝 报告模板快速参考

```markdown
# LangGraph Waiter Agent 评估报告

## 📋 测试概览

（表格）

---

## 📊 总体评估摘要

### ✅ 通过项 (X/Y = Z%)

（逐 case 表格）

---

## 🔍 详细分析

### 1. 工具调用行为分析

### 2. 工具调用效率统计

### 3. LLM Input 长度分析

### 4. 反幻觉协议遵守情况

### 5. Memory/State 更新分析

### 6. Summary 质量分析

### 7. 对话压缩分析

---

## 🎯 关键用例复盘

（至少 3 个 case 详细分析）

---

## 🐛 根因定位

（问题 → 代码位置 表格）

---

## 📈 评分（10 分制）

（评分表格）

---

## 💡 改进建议

（按优先级排列的表格）

---

_报告生成时间：YYYY-MM-DD HH:MM:SS_
_日志运行模型：xxx_
_评估所用模型：xxx_
```
