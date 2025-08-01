# 菜谱工具 (Recipe Tools)

菜谱工具是一个综合性的 MCP 美食助手，提供了菜谱查询、分类浏览、智能推荐、搜索等功能，帮助用户解决"今天吃什么"的问题。

### 常见使用场景

**菜谱查询:**
- "我想学做宫保鸡丁"
- "红烧肉怎么做"
- "番茄炒蛋的做法"
- "查一下麻婆豆腐的菜谱"

**分类浏览:**
- "有什么川菜推荐"
- "给我看看家常菜"
- "有哪些素食菜谱"
- "推荐几道汤类菜谱"

**智能推荐:**
- "今天吃什么好呢"
- "推荐几道适合2个人的晚餐"
- "给我推荐几道早餐"
- "4个人的聚餐菜谱"

**搜索功能:**
- "有没有用土豆做的菜"
- "搜索含有鸡肉的菜谱"
- "找一些简单易做的菜"
- "有什么辣的菜推荐"

**随机推荐:**
- "随机推荐一道菜"
- "不知道做什么，随便推荐几个"
- "来个惊喜菜谱"

### 使用提示

1. **明确需求**: 可以指定菜系、食材、难度等偏好
2. **人数考虑**: 可以说明用餐人数，获得更精准的推荐
3. **用餐时间**: 可以指定早餐、午餐、晚餐等时间
4. **食材偏好**: 可以提及喜欢或不喜欢的食材
5. **难度选择**: 可以要求简单易做或挑战性菜谱

AI 助手会根据您的需求自动调用菜谱工具，为您提供详细的烹饪指导。

## 功能概览

### 菜谱查询功能
- **详细菜谱**: 提供完整的制作步骤和食材清单
- **分类浏览**: 按菜系、类型、难度等分类查看
- **智能搜索**: 支持模糊搜索和关键词匹配
- **菜谱详情**: 包含制作时间、难度、营养信息等

### 智能推荐功能
- **个性化推荐**: 根据用餐人数和时间推荐
- **随机推荐**: 解决选择困难症，随机推荐菜品
- **场景推荐**: 针对不同用餐场景的菜谱推荐
- **营养搭配**: 考虑营养均衡的菜品组合

### 分类管理功能
- **菜系分类**: 川菜、粤菜、湘菜等地方菜系
- **类型分类**: 家常菜、素食、汤类等类型
- **难度分类**: 简单、中等、困难等难度级别
- **时间分类**: 早餐、午餐、晚餐、夜宵等

### 搜索功能
- **食材搜索**: 根据食材查找相关菜谱
- **关键词搜索**: 支持菜名、特色等关键词
- **模糊搜索**: 智能匹配相似菜谱
- **组合搜索**: 多条件组合搜索

## 工具列表

### 1. 菜谱查询工具

#### get_all_recipes - 获取所有菜谱
获取菜谱列表，支持分页浏览。

**参数:**
- `page` (可选): 页码，默认1
- `page_size` (可选): 每页数量，默认10，最大50

**使用场景:**
- 浏览菜谱列表
- 了解菜谱总览
- 分页查看菜谱

#### get_recipe_by_id - 获取菜谱详情
根据菜谱ID或名称获取详细信息。

**参数:**
- `query` (必需): 菜谱名称或ID

**使用场景:**
- 查看具体菜谱详情
- 获取制作步骤
- 查询食材清单

### 2. 分类浏览工具

#### get_recipes_by_category - 按分类获取菜谱
根据分类获取菜谱列表。

**参数:**
- `category` (必需): 分类名称
- `page` (可选): 页码，默认1
- `page_size` (可选): 每页数量，默认10，最大50

**使用场景:**
- 浏览特定菜系
- 查看分类菜谱
- 按类型筛选

### 3. 智能推荐工具

#### recommend_meals - 推荐菜品
根据用餐人数和时间推荐合适的菜品。

**参数:**
- `people_count` (可选): 用餐人数，默认2
- `meal_type` (可选): 用餐类型，默认"dinner"
- `page` (可选): 页码，默认1
- `page_size` (可选): 每页数量，默认10，最大50

**使用场景:**
- 根据人数推荐菜品
- 按用餐时间推荐
- 个性化菜谱推荐

#### what_to_eat - 随机推荐菜品
随机推荐菜品，解决选择困难。

**参数:**
- `meal_type` (可选): 用餐类型，默认"any"
- `page` (可选): 页码，默认1
- `page_size` (可选): 每页数量，默认10，最大50

**使用场景:**
- 随机菜品推荐
- 解决选择困难
- 尝试新菜谱

### 4. 搜索工具

#### search_recipes_fuzzy - 模糊搜索菜谱
根据关键词模糊搜索菜谱。

**参数:**
- `query` (必需): 搜索关键词
- `page` (可选): 页码，默认1
- `page_size` (可选): 每页数量，默认10，最大50

**使用场景:**
- 关键词搜索
- 食材搜索
- 模糊匹配搜索

## 使用示例

### 菜谱查询示例

```python
# 获取菜谱列表
result = await mcp_server.call_tool("get_all_recipes", {
    "page": 1,
    "page_size": 10
})

# 获取具体菜谱详情
result = await mcp_server.call_tool("get_recipe_by_id", {
    "query": "宫保鸡丁"
})

# 按分类获取菜谱
result = await mcp_server.call_tool("get_recipes_by_category", {
    "category": "川菜",
    "page": 1,
    "page_size": 10
})
```

### 智能推荐示例

```python
# 根据人数和时间推荐
result = await mcp_server.call_tool("recommend_meals", {
    "people_count": 4,
    "meal_type": "dinner",
    "page": 1,
    "page_size": 5
})

# 随机推荐菜品
result = await mcp_server.call_tool("what_to_eat", {
    "meal_type": "lunch",
    "page": 1,
    "page_size": 3
})
```

### 搜索功能示例

```python
# 模糊搜索菜谱
result = await mcp_server.call_tool("search_recipes_fuzzy", {
    "query": "土豆",
    "page": 1,
    "page_size": 10
})

# 搜索特定菜系
result = await mcp_server.call_tool("search_recipes_fuzzy", {
    "query": "家常菜",
    "page": 1,
    "page_size": 15
})
```

## 数据结构

### 菜谱信息 (Recipe)
```python
{
    "id": "recipe_123",
    "name": "宫保鸡丁",
    "category": "川菜",
    "difficulty": "中等",
    "cooking_time": "30分钟",
    "serving": "2-3人",
    "ingredients": [
        {
            "name": "鸡胸肉",
            "amount": "300g",
            "note": "切丁"
        },
        {
            "name": "花生米",
            "amount": "50g",
            "note": "炸熟"
        }
    ],
    "steps": [
        {
            "step": 1,
            "description": "鸡胸肉切丁，用料酒、生抽、淀粉腌制15分钟"
        },
        {
            "step": 2,
            "description": "热锅下油，爆炒鸡丁至变色盛起"
        }
    ],
    "tips": "炒制时火候要控制好，避免过老",
    "nutrition": {
        "calories": "280kcal",
        "protein": "25g",
        "fat": "12g",
        "carbs": "15g"
    }
}
```

### 分页结果 (PagedResult)
```python
{
    "data": [
        {
            "id": "recipe_123",
            "name": "宫保鸡丁",
            "category": "川菜",
            "difficulty": "中等",
            "cooking_time": "30分钟"
        }
    ],
    "pagination": {
        "page": 1,
        "page_size": 10,
        "total": 156,
        "total_pages": 16,
        "has_next": true,
        "has_prev": false
    }
}
```

### 推荐信息 (RecommendationInfo)
```python
{
    "recommendation_info": {
        "people_count": 4,
        "meal_type": "dinner",
        "message": "为 4 人的dinner推荐菜品"
    }
}
```

## 支持的分类

### 菜系分类
- **川菜**: 麻辣鲜香的四川菜系
- **粤菜**: 清淡鲜美的广东菜系
- **湘菜**: 香辣浓郁的湖南菜系
- **鲁菜**: 咸鲜为主的山东菜系
- **苏菜**: 清淡甜美的江苏菜系
- **浙菜**: 清香爽脆的浙江菜系
- **闽菜**: 清淡甘甜的福建菜系
- **徽菜**: 香鲜适口的安徽菜系

### 类型分类
- **家常菜**: 日常家庭料理
- **素食**: 素食菜谱
- **汤类**: 各种汤品
- **凉菜**: 冷菜开胃菜
- **面食**: 面条、饺子等
- **甜品**: 甜点糕点
- **下酒菜**: 适合配酒的菜品

### 难度分类
- **简单**: 新手友好，步骤简单
- **中等**: 需要一定烹饪技巧
- **困难**: 需要丰富烹饪经验

### 时间分类
- **早餐**: 早餐菜谱
- **午餐**: 午餐菜谱
- **晚餐**: 晚餐菜谱
- **夜宵**: 夜宵小食
- **下午茶**: 茶点小食

## 最佳实践

### 1. 菜谱查询
- 使用准确的菜名获得最佳结果
- 可以通过分类浏览发现新菜谱
- 关注菜谱的难度和时间要求

### 2. 智能推荐
- 准确提供用餐人数获得合适分量
- 根据用餐时间选择合适菜品
- 考虑营养搭配的平衡性

### 3. 搜索技巧
- 使用食材名称搜索相关菜谱
- 尝试不同的关键词组合
- 利用模糊搜索发现意外惊喜

### 4. 分页使用
- 合理设置每页数量
- 逐页浏览避免信息过载
- 注意总页数和当前页位置

## 注意事项

1. **食材新鲜**: 确保使用新鲜食材
2. **过敏提醒**: 注意食物过敏问题
3. **营养搭配**: 考虑营养均衡
4. **烹饪安全**: 注意厨房安全操作
5. **分量调整**: 根据实际人数调整用量

## 故障排除

### 常见问题
1. **搜索无结果**: 尝试不同关键词或分类浏览
2. **菜谱不详细**: 查看菜谱详情页面
3. **推荐不合适**: 调整推荐参数
4. **分页错误**: 检查页码和页面大小

### 调试方法
1. 验证搜索关键词拼写
2. 检查分类名称是否正确
3. 确认页码参数范围
4. 查看返回的错误信息

通过菜谱工具，您可以轻松解决"今天吃什么"的问题，发现新的美食，学习烹饪技巧，享受美食带来的快乐。
