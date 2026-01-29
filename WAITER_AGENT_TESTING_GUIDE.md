# Waiter Agent Testing Guide

## 测试数据集文件说明

### 主场景测试 (Main Scenario)

**文件**: `waiter_agent_accessment_golden_dataset_v2.json`

**内容**:

- **id:1-20**: 一个完整的连续对话场景
- 客户从进店 → 看菜单 → 点菜 → 修改 → 确认 → 结账 → 支付

**特点**:

- ✅ 连续对话，有完整的上下文
- ✅ 测试正常的服务流程
- ✅ 包含 id:13.5（库存充足场景）
- ✅ 总计21个测试用例

**用法**:

```bash
# 直接运行，不需要重置状态
python waiter_claude_agent_sdk_demo.py --dataset waiter_agent_accessment_golden_dataset_v2.json
```

---

### 边界测试 (Edge Cases)

**文件**: `waiter_agent_edge_cases_dataset.json`

**内容**:

- 4个独立的测试场景
- 每个scenario有独立的上下文

**场景列表**:

1. **edge_case_1**: 取消未确认订单
2. **edge_case_2**: 修改已下订单
3. **edge_case_3**: 分批上菜请求
4. **edge_case_4**: 空note尝试下单

**特点**:

- ✅ 每个scenario独立运行
- ✅ 测试边界和异常情况
- ✅ 不依赖其他scenario的上下文

**用法**:

```python
# 需要为每个scenario重置状态
import json
from pathlib import Path

# 加载边界测试
edge_cases_path = Path("waiter_agent_edge_cases_dataset.json")
edge_cases = json.loads(edge_cases_path.read_text(encoding="utf-8"))

for scenario in edge_cases:
    print(f"\n{'='*60}")
    print(f"Testing: {scenario['scenario_name']}")
    print(f"{'='*60}\n")

    # 重置状态（重要！）
    state_reset()
    conversation_reset()

    # 运行该scenario的所有测试用例
    for test_case in scenario['test_cases']:
        user_query = test_case['user_query']
        response = await _run_turn(user_query, options)
        print(f"User: {user_query}")
        print(f"Waiter: {response}\n")
```

---

## 为什么要分离？

### 问题场景

如果将边界测试混在主场景后（id:21-27），会发生：

```
id:20 - 客户支付完成，准备离开
id:21 - "Hi, we'd like two Caprese Salads..."
        ❌ Agent会认为是同一桌客户又回来点菜了！
```

### 正确的方式

```
主场景文件 (v2.json):
  id:1-20 - 完整的客户A对话

边界测试文件 (edge_cases.json):
  scenario_1 - 客户B的独立对话
  scenario_2 - 客户C的独立对话
  scenario_3 - 客户D的独立对话
  scenario_4 - 客户E的独立对话
```

---

## 评估标准

### 主场景评估重点

1. **工具调用时机**
   - ✅ id:6-15 只操作NOTE，不创建订单
   - ✅ id:16 用户说"that's all"时才创建订单
   - ❌ 不要在NOTE阶段调用 `order_create_from_note`

2. **措辞准确性**
   - ✅ NOTE阶段说"noted"、"added to my note"
   - ❌ 不要说"placed order"、"sent to kitchen"

3. **上下文连贯性**
   - ✅ agent记得之前的对话内容
   - ✅ 正确引用已点的菜品
   - ❌ 不要遗忘之前的订单项

4. **模糊表达理解** (id:14)
   - ✅ 识别"skip dessert"的意图
   - ✅ 同时确认"stick to main courses"
   - 这是测试LLM理解能力，不要求完美匹配example_response

### 边界测试评估重点

1. **状态隔离**
   - ✅ 每个scenario开始时状态是干净的
   - ❌ 不要携带上个scenario的订单

2. **错误处理**
   - ✅ 空note时拒绝创建订单
   - ✅ 库存不足时提供替代方案
   - ✅ 取消订单时礼貌响应

3. **复杂操作**
   - ✅ 修改已下订单使用正确的load→modify→update流程
   - ✅ 处理带notes的item更新

---

## 测试执行示例

### 完整测试流程

```python
import asyncio
import json
from pathlib import Path

async def run_full_test_suite():
    """运行完整的测试套件"""

    # === 1. 主场景测试 ===
    print("="*80)
    print("MAIN SCENARIO TEST")
    print("="*80)

    state_reset()
    conversation_reset()

    main_dataset = json.loads(
        Path("waiter_agent_accessment_golden_dataset_v2.json")
        .read_text(encoding="utf-8")
    )

    for entry in main_dataset:
        user_query = entry["user_query"]
        response = await _run_turn(user_query, options)

        # 评估是否符合expected_behavior
        evaluate_response(entry, response)

    print("\n✅ Main scenario test completed\n")

    # === 2. 边界测试 ===
    print("="*80)
    print("EDGE CASES TEST")
    print("="*80)

    edge_cases = json.loads(
        Path("waiter_agent_edge_cases_dataset.json")
        .read_text(encoding="utf-8")
    )

    for scenario in edge_cases:
        print(f"\n--- {scenario['scenario_name']} ---")

        # 重置状态（关键！）
        state_reset()
        conversation_reset()

        for test_case in scenario['test_cases']:
            user_query = test_case['user_query']
            response = await _run_turn(user_query, options)

            # 评估是否符合expected_behavior
            evaluate_response(test_case, response)

    print("\n✅ Edge cases test completed\n")

# 运行测试
asyncio.run(run_full_test_suite())
```

---

## 常见问题

### Q1: 为什么id:13.5不是整数？

**A**: id:13.5是在id:13和id:14之间插入的测试用例，保持了原有测试的编号不变。

### Q2: 边界测试可以合并到主场景吗？

**A**: 不建议。边界测试是独立场景，合并会导致上下文混淆。除非在主场景逻辑流程中自然出现这些情况。

### Q3: 如何添加新的边界测试？

**A**: 在 `waiter_agent_edge_cases_dataset.json` 中添加新的scenario对象：

```json
{
  "scenario_id": "edge_case_5",
  "scenario_name": "Your Scenario Name",
  "description": "Scenario description",
  "test_cases": [...]
}
```

### Q4: 主场景测试需要重置状态吗？

**A**: 不需要！主场景是连续对话，状态需要保持。只在开始前重置一次。

### Q5: 如何判断agent是否正确理解了id:14？

**A**: 关键检查点：

- ✅ 识别"skip dessert"意图
- ✅ 从note中移除Chocolate Cake
- ✅ 确认还有Lasagna和Ribeye Steak
- 不要求response完全匹配example，理解意图即可

---

## 评分建议

### 主场景 (70分)

- 工具调用正确性: 30分
- 响应措辞准确性: 15分
- 上下文连贯性: 15分
- 模糊理解能力: 10分

### 边界测试 (30分)

- 取消订单处理: 8分
- 修改订单流程: 10分
- 分批确认识别: 7分
- 空状态处理: 5分

---

## 更新日志

详见 `GOLDEN_DATASET_V2_CHANGELOG.md`
