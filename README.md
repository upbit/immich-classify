# immich-classify

A Python CLI tool that classifies photos in [Immich](https://immich.app/) using Vision Language Models (VLM). It fetches images via the Immich API, sends them to any OpenAI-compatible VLM for structured classification, and stores results in a local SQLite database for querying and export.

```
CLI ──Immich API──> Immich Server (albums / assets / thumbnails)
 |
 |──base64──> VLM API (OpenAI Chat Completion protocol)
 |
 └──write──> Local SQLite (classification results + task state)
```

## Features

- **Universal VLM support** - Works with vLLM, Ollama, OpenAI, and any OpenAI-compatible API via structured output (`response_format`)
- **Customizable schema** - Define your own classification fields (category, tags, quality, NSFW, etc.) through Python dataclasses
- **Async & concurrent** - Built on `asyncio` + `httpx` with configurable concurrency via semaphore
- **Resumable tasks** - Every result is persisted immediately; pause/resume without losing progress
- **Graceful Ctrl+C** - Interrupt a running task to pause it; resume later from where it stopped
- **Flexible export** - Query results with dynamic JSON field filtering; output as table, JSON, or CSV
- **Type-safe** - Full type annotations passing Pyright strict mode with zero errors

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- A running [Immich](https://immich.app/) server
- An OpenAI-compatible VLM API endpoint (vLLM, Ollama, OpenAI, etc.)

## Installation

```bash
git clone https://github.com/your-username/immich-classify.git
cd immich-classify
uv sync
```

## Configuration

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

| Variable | Description | Default |
|----------|-------------|---------|
| `IMMICH_API_URL` | Immich server URL | *required* |
| `IMMICH_API_KEY` | Immich API key | *required* |
| `VLM_API_URL` | OpenAI-compatible API base URL | `http://localhost:8000/v1` |
| `VLM_API_KEY` | VLM API key | `no-key` |
| `VLM_MODEL_NAME` | Model name (empty = server default) | |
| `CLASSIFY_DB_PATH` | SQLite database path | `./classify.db` |
| `CLASSIFY_CONCURRENCY` | Max concurrent image processing | `1` |
| `CLASSIFY_TIMEOUT` | VLM request timeout in seconds | `60` |
| `CLASSIFY_IMAGE_SIZE` | `thumbnail` or `original` | `thumbnail` |

Environment variables override `.env` file values.

## Usage

### Quick start

```bash
# 1. List albums to find the target
immich-classify albums

# 2. Debug with a small batch first
immich-classify debug --album <album_id> --count 5

# 3. Run full classification
immich-classify classify --album <album_id>

# 4. Check progress
immich-classify status --task <task_id>

# 5. View results
immich-classify results --task <task_id> --filter category=people --format table

# 6. Export for review
immich-classify results --task <task_id> --format csv > results.csv
```

### Commands

```
immich-classify albums
    List all Immich albums with ID, name, and asset count.

immich-classify classify --album <id> [--album <id2>] [--prompt-config <file>] [--concurrency <n>]
    Create and run a classification task. Supports multiple albums.

immich-classify debug --album <id> [--count <n>] [--prompt-config <file>]
    Run a small debug batch (default 10) and print results. No database writes.

immich-classify status [--task <task_id>]
    Show all tasks, or detailed progress for a specific task.

immich-classify results --task <id> [--filter <key=value>]... [--format json|csv|table]
    Query classification results with optional field filtering.

immich-classify pause --task <id>       Pause a running task.
immich-classify resume --task <id>      Resume a paused task.
immich-classify cancel --task <id>      Cancel a task (keeps existing results).
```

### Custom classification schema

Create a Python file with a `ClassificationPrompt` instance:

```python
# my_schema.py
from immich_classify.prompt import ClassificationPrompt, SchemaField

prompt = ClassificationPrompt(
    system_prompt=(
        "You are a photo organizer. Classify the image into the given schema. "
        "Output ONLY valid JSON."
    ),
    schema={
        "scene": SchemaField(
            field_type="string",
            description="Scene type",
            enum=["indoor", "outdoor", "studio", "unknown"],
        ),
        "people_count": SchemaField(
            field_type="int",
            description="Number of people visible",
        ),
        "is_screenshot": SchemaField(
            field_type="bool",
            description="Whether the image is a screenshot",
        ),
        "tags": SchemaField(
            field_type="list[string]",
            description="Descriptive tags",
        ),
    },
)
```

Then use it:

```bash
immich-classify classify --album <id> --prompt-config my_schema.py
```

## Architecture

```
src/immich_classify/
├── config.py          # Config dataclass, .env loading, validation
├── prompt.py          # ClassificationPrompt & SchemaField dataclasses
├── immich_client.py   # Async Immich API client (httpx)
├── vlm_client.py      # Async OpenAI-compatible VLM client (httpx)
├── database.py        # Async SQLite layer (aiosqlite)
├── engine.py          # Task execution engine (asyncio + semaphore)
├── cli.py             # CLI entry point and subcommand handlers
└── __main__.py        # python -m entry point
```

**Key design decisions:**

- **SQLite with `json_extract()`** - Classification fields are fully dynamic. Results are stored as JSON and queried with SQLite's JSON functions, so no schema migration is needed when fields change.
- **Structured Output** - Uses `response_format: { type: "json_schema" }` to enforce valid JSON output from the VLM, rather than fragile regex parsing.
- **Per-asset persistence** - Each image result is committed immediately. A crash or interrupt loses at most the in-flight images, not the entire batch.
- **Asset deduplication** - When classifying multiple albums, assets appearing in more than one album are automatically deduplicated.

## Development

```bash
# Install dev dependencies
uv sync

# Run tests
uv run pytest

# Run type checker (strict mode)
uv run pyright src/immich_classify/
```

### Test suite

72 tests covering all modules:

| Module | Tests | Coverage |
|--------|-------|----------|
| `config.py` | 8 | Validation, env loading, defaults, missing fields |
| `prompt.py` | 13 | Schema generation, JSON schema, serialization roundtrip |
| `database.py` | 11 | CRUD, filtering with `json_extract`, deduplication |
| `immich_client.py` | 5 | Album listing, asset filtering, image download |
| `vlm_client.py` | 7 | Success, API errors, invalid JSON, structured output |
| `engine.py` | 9 | Concurrency, error continuation, pause/resume, dedup |
| `cli.py` | 19 | Argument parsing, filter parsing, multi-album |

## Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python 3.11+ | Rapid iteration, rich async ecosystem |
| HTTP | [httpx](https://www.python-httpx.org/) | Native async, connection pooling |
| Database | [aiosqlite](https://github.com/omnilib/aiosqlite) | Async SQLite, zero setup |
| Logging | [loguru](https://github.com/Delgan/loguru) | Structured, colorful, zero config |
| CLI | argparse | Standard library, no extra dependency |
| Formatting | [tabulate](https://github.com/astanin/python-tabulate) | Clean table output |
| Type checking | [Pyright](https://github.com/microsoft/pyright) | Strict mode, zero errors |
| Package manager | [uv](https://docs.astral.sh/uv/) | Fast, reliable, modern |

## License

MIT
