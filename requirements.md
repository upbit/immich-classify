# 需求文档：基于 VLM 的 Immich 图片分类工具（Python CLI 版）

## 概述

一个 Python CLI 工具，通过 Immich API 获取相册图片，调用 VLM（Vision Language Model）进行结构化分类，将分类结果存入本地 SQLite，供用户预览确认后再手动操作。

### 技术栈

| 组件 | 选择 | 理由 |
|------|------|------|
| 语言 | Python 3.11+ | 方便调整迭代 |
| VLM 调用 | OpenAI Chat Completion API（`/v1/chat/completions`） | 兼容 vLLM、Ollama、OpenAI 等所有框架 |
| 图片传入 | base64 编码 | 兼容云端 API，无需 VLM 直连 Immich |
| 数据存储 | SQLite | 简单可靠，方便按分类筛选查询 |
| Prompt/Schema 管理 | Python 内置类（dataclass） | 方便后续 AI 编写和维护 |
| 日志 | loguru | 可读性强，方便调试和突出展示关键信息 |
| 包管理 | uv | 现代化 Python 包管理器，快速可靠 |
| 交互方式 | CLI（子命令） | 先做原型，后续可扩展 Web |

### 核心流程

```
CLI 命令
  │
  ├─ immich-classify albums             # 列出所有相册
  ├─ immich-classify classify           # 对指定相册执行分类
  ├─ immich-classify debug              # 执行分类小批量调试
  ├─ immich-classify status             # 查看任务状态/进度
  ├─ immich-classify results            # 查看/导出分类结果
  ├─ immich-classify pause              # 暂停运行中的任务
  ├─ immich-classify resume             # 恢复已暂停的任务
  └─ immich-classify cancel             # 取消任务（保留已有结果）
```

单个图片的执行流程

```
CLI ──Immich API──▶ Immich Server（获取相册/图片/缩略图）
 │
 ├──base64──▶ VLM API（OpenAI Chat Completion 协议）
 │
 └──写入──▶ 本地 SQLite（分类结果 + 任务状态）
```

---

## 需求 1：配置管理

**目标：** 通过 `.env` 文件和环境变量管理所有连接信息和默认参数。

### 配置项

| 配置项 | 环境变量 | 说明 | 默认值 |
|--------|----------|------|--------|
| Immich 服务地址 | `IMMICH_API_URL` | Immich API 基础 URL | 无，必填 |
| Immich API Key | `IMMICH_API_KEY` | Immich 访问密钥 | 无，必填 |
| VLM API 地址 | `VLM_API_URL` | OpenAI 兼容 API 的基础 URL | `http://localhost:8000/v1` |
| VLM API Key | `VLM_API_KEY` | VLM 服务的 API Key | `no-key` |
| VLM 模型名称 | `VLM_MODEL_NAME` | 模型名称 | 无，不配置默认model填空，此时用框架默认模型 |
| SQLite 数据库路径 | `CLASSIFY_DB_PATH` | 数据库文件路径 | `./classify.db` |
| 并发数 | `CLASSIFY_CONCURRENCY` | 同时处理的图片数 | `1` |
| 请求超时 | `CLASSIFY_TIMEOUT` | 单次 VLM 请求超时（秒） | `60` |
| 图片尺寸 | `CLASSIFY_IMAGE_SIZE` | 使用 `thumbnail` 或 `original` | `thumbnail` |

### 配置加载优先级

`.env` 文件 → 环境变量（后者覆盖前者）

### 验收标准

1. WHEN `.env` 文件存在 THEN 系统 SHALL 读取其中的配置
2. WHEN 环境变量存在 THEN 系统 SHALL 用环境变量覆盖 `.env` 中的同名配置
3. WHEN 必填配置缺失 THEN 系统 SHALL 输出明确的错误信息并退出

---

## 需求 2：Immich API 交互

**目标：** 通过 Immich API 获取相册列表、相册内资产列表、下载图片。

### 需要调用的 Immich API

| 功能 | Immich API 端点 | 说明 |
|------|-----------------|------|
| 获取相册列表 | `GET /api/albums` | 返回所有相册（名称、ID、资产数量） |
| 获取相册资产 | `GET /api/albums/{id}` | 返回相册内所有资产详情 |
| 获取缩略图 | `GET /api/assets/{id}/thumbnail` | 下载缩略图用于 VLM 分析 |
| 获取原图 | `GET /api/assets/{id}/original` | 可选，下载原图用于更精确分析 |

### 验收标准

1. WHEN 用户执行 `albums` 子命令 THEN 系统 SHALL 列出所有相册的 ID、名称、资产数量
2. WHEN Immich API 不可达 THEN 系统 SHALL 输出明确的连接错误信息
3. WHEN 下载图片用于分析 THEN 系统 SHALL 将图片转为 base64 编码字符串
4. WHEN 配置为 `thumbnail` 模式 THEN 系统 SHALL 使用缩略图端点（节省带宽和 token）

---

## 需求 3：VLM 调用（OpenAI Chat Completion 协议）

**目标：** 通过标准 OpenAI Chat Completion API 协议调用 VLM，发送图片和 Prompt，获取结构化 JSON 分类结果。

### API 调用格式

```json
{
  "model": "<model_name>",
  "messages": [
    {
      "role": "system",
      "content": "<system_prompt>"
    },
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,<base64_data>"
          }
        },
        {
          "type": "text",
          "text": "<user_prompt>"
        }
      ]
    }
  ],
  "temperature": 0.1,
  "max_tokens": 1024
}
```

### 验收标准

1. WHEN 系统调用 VLM THEN 系统 SHALL 使用标准 OpenAI `/v1/chat/completions` 端点
2. WHEN 模型返回合法 JSON THEN 系统 SHALL 解析并按 Schema 验证
3. IF 模型返回非法 JSON THEN 系统 SHALL 记录原始响应并标记为"解析失败"，不中断任务
4. WHEN VLM 请求超时或失败 THEN 系统 SHALL 记录错误并继续处理下一张，支持配置重试次数

---

## 需求 4：Prompt 与 Schema 管理（Python 内置类）

**目标：** 使用 Python dataclass 定义 Prompt 和输出 Schema，内置默认配置，支持用户自定义。

### 默认 Prompt 和 Schema

系统内置一个默认的分类配置类，用户可以通过继承或传入配置文件覆盖：

```python
@dataclass
class ClassificationPrompt:
    """Classification prompt and schema definition."""

    system_prompt: str = (
        "You are an image classification assistant. "
        "Analyze the given image and output a JSON object "
        "following the specified schema. Output ONLY valid JSON, no other text."
    )

    user_prompt: str = (
        "Classify this image according to the following schema:\n"
        "{schema_description}\n\n"
        "Output a JSON object with the specified fields."
    )

    schema: dict[str, SchemaField] = field(default_factory=lambda: {
        "category": SchemaField(
            field_type="string",
            description="Primary category of the image",
            enum=["people", "landscape", "food", "animal", "architecture",
                  "document", "screenshot", "meme", "other"],
        ),
        "quality": SchemaField(
            field_type="string",
            description="Image quality assessment",
            enum=["high", "medium", "low", "blurry"],
        ),
        "tags": SchemaField(
            field_type="list[string]",
            description="Descriptive tags for the image content",
        ),
    })


@dataclass
class SchemaField:
    """Definition of a single field in the classification schema."""

    field_type: str          # "string", "int", "float", "bool", "list[string]"
    description: str         # Field description for the prompt
    enum: list[str] | None = None  # Optional: allowed values
    default: Any = None      # Optional: default value
```

### 验收标准

1. WHEN 用户未指定自定义配置 THEN 系统 SHALL 使用内置的默认 `ClassificationPrompt`
2. WHEN 用户通过 `--prompt-config` 指定 Python 文件 THEN 系统 SHALL 动态加载该文件中的 `ClassificationPrompt` 实例并覆盖默认配置（Python 文件方便 AI 编写和验证，且支持复杂逻辑扩展）
3. WHEN 构建 VLM 请求 THEN 系统 SHALL 根据 Schema 自动生成 schema 描述文本并嵌入 user_prompt
4. WHEN 调用 VLM THEN 系统 SHALL 使用 Structured Output（`response_format`）解析返回结果；若服务端不支持 Structured Output THEN 系统 SHALL 报错并提示原因

---

## 需求 5：SQLite 存储

**目标：** 使用 SQLite 存储任务状态和分类结果，结构简单清晰。全程使用 `aiosqlite` 异步接口，与任务执行引擎的 `asyncio` 架构保持一致。

### 表结构

```sql
-- Task table: stores task metadata and execution state
CREATE TABLE tasks (
    task_id         TEXT PRIMARY KEY,           -- UUID
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending/running/paused/completed/cancelled/failed
    album_ids       TEXT NOT NULL,              -- JSON array of album IDs
    prompt_config   TEXT NOT NULL,              -- JSON serialized ClassificationPrompt
    total_count     INTEGER NOT NULL DEFAULT 0, -- Total assets to process
    completed_count INTEGER NOT NULL DEFAULT 0, -- Successfully processed count
    failed_count    INTEGER NOT NULL DEFAULT 0, -- Failed count
    created_at      TEXT NOT NULL,              -- ISO 8601 timestamp
    updated_at      TEXT NOT NULL               -- ISO 8601 timestamp
);

-- Classification results table: stores per-image results
CREATE TABLE classification_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id         TEXT NOT NULL REFERENCES tasks(task_id),
    asset_id        TEXT NOT NULL,              -- Immich asset ID
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending/success/failed
    result_json     TEXT,                       -- Full JSON result from VLM
    error_message   TEXT,                       -- Error message if failed
    raw_response    TEXT,                       -- Raw VLM response (for debugging)
    created_at      TEXT NOT NULL,
    UNIQUE(task_id, asset_id)
);

-- Index for efficient querying by classification values
CREATE INDEX idx_results_task ON classification_results(task_id);
CREATE INDEX idx_results_status ON classification_results(status);
```

### 分类结果查询

由于分类字段是动态的（用户自定义 Schema），`result_json` 存储完整 JSON，查询时使用 SQLite 的 `json_extract()` 函数：

```sql
-- Example: find all "people" images in a task
SELECT asset_id, result_json
FROM classification_results
WHERE task_id = ? AND json_extract(result_json, '$.category') = 'people';
```

### 验收标准

1. WHEN 任务创建 THEN 系统 SHALL 在 `tasks` 表中插入一条记录
2. WHEN 每张图片处理完成 THEN 系统 SHALL 通过 `aiosqlite` 立即异步写入 `classification_results` 并更新 `tasks` 的计数
3. WHEN 任务恢复 THEN 系统 SHALL 跳过 `classification_results` 中已有 `status='success'` 的资产
4. WHEN 用户查询分类结果 THEN 系统 SHALL 使用 `json_extract()` 按动态字段筛选
5. WHEN 执行数据库操作 THEN 系统 SHALL 全程使用 `aiosqlite` 异步接口

---

## 需求 6：CLI 子命令设计

**目标：** 提供简洁的 CLI 子命令，覆盖核心工作流。

### 子命令一览

```
immich-classify albums
    List all Immich albums with ID, name, and asset count.

immich-classify classify --album <album_id> [--prompt-config <file>] [--concurrency <n>]
    Create and run a classification task for the specified album.

immich-classify debug --album <album_id> [--count <n>] [--prompt-config <file>]
    Run a small debug batch (default 10 images) and print results to stdout.

immich-classify status [--task <task_id>]
    Show status of all tasks, or a specific task's detailed progress.

immich-classify results --task <task_id> [--filter <key=value>] [--format json|csv|table]
    Query and display classification results. Supports filtering by any schema field.

immich-classify pause --task <task_id>
    Pause a running task.

immich-classify resume --task <task_id>
    Resume a paused task.

immich-classify cancel --task <task_id>
    Cancel a task (keeps existing results).
```

### 典型工作流

```bash
# 1. List albums to find the target
immich-classify albums

# 2. Debug with a small batch first
immich-classify debug --album <album_id> --count 10

# 3. Run full classification
immich-classify classify --album <album_id>

# 4. Check progress
immich-classify status --task <task_id>

# 5. View results, filter by category
immich-classify results --task <task_id> --filter category=people --format table

# 6. Export results for manual review
immich-classify results --task <task_id> --format csv > results.csv
```

### 验收标准

1. WHEN 用户执行 `albums` THEN 系统 SHALL 以表格形式输出相册列表
2. WHEN 用户执行 `classify` THEN 系统 SHALL 创建任务并开始处理，实时输出进度
3. WHEN 用户执行 `results --filter` THEN 系统 SHALL 使用 `json_extract()` 按指定字段筛选
4. WHEN 用户执行 `results --format csv` THEN 系统 SHALL 输出 CSV 格式，动态列名来自 Schema 字段
5. WHEN 用户执行 `debug` THEN 系统 SHALL 随机抽取指定数量图片分析并通过 `loguru` 输出结果到 stdout（使用高亮和结构化格式增强可读性），不写入任何数据库
6. WHEN 用户按 Ctrl+C 中断运行中的任务 THEN 系统 SHALL 优雅退出并将任务状态标记为 `paused`

---

## 需求 7：任务执行引擎

**目标：** 支持异步并发处理、进度持久化、可暂停/恢复。

### 验收标准

1. WHEN 任务开始 THEN 系统 SHALL 使用 `asyncio` + `aiohttp`（或 `httpx`）进行异步并发处理
2. WHEN 并发数配置为 N THEN 系统 SHALL 同时最多处理 N 张图片
3. WHEN 每张图片完成 THEN 系统 SHALL 立即持久化结果到 SQLite（不等待批量提交）
4. WHEN 任务暂停后恢复 THEN 系统 SHALL 从 SQLite 读取已完成列表，只处理剩余图片
5. WHEN 处理过程中 VLM 返回错误 THEN 系统 SHALL 记录错误并继续，不中断整个任务
6. WHEN 任务运行中 THEN 系统 SHALL 每处理完一张图片输出进度信息（`[42/1000] asset_id: xxx - category: people`）

---

## 非功能性需求

### NFR-1：性能

- 支持配置并发数（默认 1，可调高）
- SQLite 查询应在 1 秒内完成（10 万条记录规模）
- 使用缩略图模式时，单张图片的 base64 编码大小应控制在合理范围内

### NFR-2：可靠性

- 每张图片处理结果立即持久化，崩溃后可恢复
- VLM 请求失败不中断整个任务
- Ctrl+C 优雅退出，保存当前进度

### NFR-3：可扩展性

- VLM 调用层使用标准 OpenAI 协议，可无缝切换 vLLM / Ollama / OpenAI / 其他兼容服务
- Prompt 和 Schema 通过 Python 类管理，方便扩展和维护
- 分类字段完全动态，不硬编码任何特定字段名

### NFR-4：代码质量

- 完整的类型注解（兼容 Pyright strict 模式）
- 显式处理 `None` 值，使用 `assert` 或条件判断进行类型窄化
- 变量和函数使用完整、明确的命名，避免歧义缩写

---

## 项目结构

```
immich-classify/
├── requirements.md          # This document
├── pyproject.toml           # Project metadata and dependencies (managed by uv)
├── .env.example             # Example environment configuration
├── src/
│   └── immich_classify/
│       ├── __init__.py
│       ├── __main__.py      # CLI entry point (argparse)
│       ├── config.py        # Configuration loading (.env + environment variables)
│       ├── immich_client.py # Immich API client
│       ├── vlm_client.py    # OpenAI-compatible VLM client
│       ├── prompt.py        # ClassificationPrompt & SchemaField dataclasses
│       ├── database.py      # SQLite operations
│       ├── engine.py        # Task execution engine (async)
│       └── cli.py           # CLI subcommand handlers
└── tests/
    └── ...
```

## 依赖管理

使用 [uv](https://docs.astral.sh/uv/) 作为包管理器，依赖声明在 `pyproject.toml` 中。

### 常用命令

```bash
uv sync              # Install dependencies
uv run immich-classify albums   # Run CLI via uv
```

### 依赖列表

```
httpx>=0.27          # HTTP client (async support)
aiosqlite>=0.20      # Async SQLite
tabulate>=0.9        # CLI table formatting
loguru>=0.7          # Logging with rich formatting
python-dotenv>=1.0   # .env file loading
```
