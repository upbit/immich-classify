"""CLI subcommand handlers and entry point."""

from __future__ import annotations

import argparse
import asyncio
import csv
import importlib.util
import io
import json
import signal
import sys
from typing import Any, cast

from loguru import logger
from tabulate import tabulate

from immich_classify.config import Config, load_config
from immich_classify.database import Database, FilterParseError, parse_filter_strings
from immich_classify.engine import TaskEngine, debug_classify
from immich_classify.immich_client import ImmichClient
from immich_classify.prompt_base import BasePrompt
from immich_classify.prompt_generator import PromptGenerator, PromptGeneratorError, export_as_python
from immich_classify.vlm_client import VLMClient


def _resolve_prompt_path(path: str) -> str:
    """Resolve a prompt config path with fallback search.

    Search order:
    1. *path* as-is (absolute or relative to cwd).
    2. ``<package>/prompts/<path>`` — the built-in prompts directory.

    Args:
        path: A file path (absolute or relative) or bare filename.

    Returns:
        The first existing path found.

    Raises:
        SystemExit: If the file cannot be found in any search location.
    """
    import pathlib

    # 1. Current directory (or absolute path)
    candidate = pathlib.Path(path)
    if candidate.is_file():
        return str(candidate)

    # 2. Built-in prompts directory
    builtin_dir = pathlib.Path(__file__).parent / "prompts"
    candidate = builtin_dir / path
    if candidate.is_file():
        logger.debug("Resolved prompt config '{}' → {}", path, candidate)
        return str(candidate)

    logger.error(
        "Prompt config '{}' not found. Searched:\n"
        "  1. Current directory: {}\n"
        "  2. Built-in prompts:  {}",
        path,
        pathlib.Path.cwd() / path,
        builtin_dir / path,
    )
    sys.exit(1)


def _load_prompt_from_file(path: str) -> BasePrompt:
    """Load a BasePrompt from a Python file.

    The *path* is first resolved via :func:`_resolve_prompt_path` (cwd →
    built-in prompts directory) before loading.

    Discovery order inside the file:
    1. A module-level variable named ``prompt`` or ``PROMPT``.
    2. Any ``BasePrompt`` *instance* found in the module.
    3. Any ``BasePrompt`` *subclass* (not ``BasePrompt`` itself) — instantiated automatically.

    Args:
        path: Path to a Python file containing a BasePrompt definition.

    Returns:
        BasePrompt instance.
    """
    resolved = _resolve_prompt_path(path)
    spec = importlib.util.spec_from_file_location("prompt_config", resolved)
    if spec is None or spec.loader is None:
        logger.error("Cannot load prompt config from {}", path)
        sys.exit(1)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 1. Look for named variables
    prompt = getattr(module, "prompt", None)
    if prompt is None:
        prompt = getattr(module, "PROMPT", None)

    # 2. Look for any BasePrompt instance
    if prompt is None:
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, BasePrompt):
                prompt = attr
                break

    # 3. Look for any BasePrompt subclass and instantiate it
    if prompt is None:
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BasePrompt)
                and attr is not BasePrompt
            ):
                prompt = attr()
                break

    if not isinstance(prompt, BasePrompt):
        logger.error(
            "Could not find a BasePrompt subclass or instance in {}. "
            "Define a class that subclasses BasePrompt, or a variable named 'prompt'.",
            path,
        )
        sys.exit(1)

    return prompt


def _get_default_prompt(config: Config) -> BasePrompt:
    """Return the default prompt, respecting CLASSIFY_DEFAULT_PROMPT.

    If ``config.default_prompt`` is set, it is treated as a path to a
    Python file containing a prompt definition.  Otherwise, the built-in
    ``ClassificationPrompt`` is used.

    Args:
        config: Application configuration.

    Returns:
        BasePrompt instance.
    """
    if config.default_prompt:
        logger.debug("Loading default prompt from {}", config.default_prompt)
        return _load_prompt_from_file(config.default_prompt)

    # Lazy import to avoid circular dependency and allow the default to be
    # overridden without touching this module.
    from immich_classify.prompts.classification import ClassificationPrompt

    return ClassificationPrompt()


def _load_prompt_config(path: str | None, config: Config) -> BasePrompt:
    """Load a BasePrompt from a Python file or return the configured default.

    Args:
        path: Optional path to a Python file containing a BasePrompt definition.
            When *None*, the default prompt (from ``CLASSIFY_DEFAULT_PROMPT``
            or the built-in ``ClassificationPrompt``) is used.
        config: Application configuration.

    Returns:
        BasePrompt instance.
    """
    if path is None:
        return _get_default_prompt(config)
    return _load_prompt_from_file(path)


async def cmd_albums(config: Config) -> None:
    """List all Immich albums."""
    immich = ImmichClient(config.immich_api_url, config.immich_api_key, config.timeout)
    try:
        albums = await immich.list_albums()
        if not albums:
            logger.info("No albums found")
            return

        table_data = [
            [album.album_id, album.album_name, album.asset_count]
            for album in albums
        ]
        print(tabulate(
            table_data,
            headers=["Album ID", "Name", "Asset Count"],
            tablefmt="simple",
        ))
    except Exception as exc:
        logger.error("Failed to list albums: {}", exc)
        sys.exit(1)
    finally:
        await immich.close()


async def cmd_classify(
    config: Config,
    album_ids: list[str],
    prompt_config_path: str | None,
    concurrency: int | None,
) -> None:
    """Create and run a classification task."""
    prompt_config = _load_prompt_config(prompt_config_path, config)
    database = Database(config.database_path)
    immich = ImmichClient(config.immich_api_url, config.immich_api_key, config.timeout)
    vlm = VLMClient(config.vlm_api_url, config.vlm_api_key, config.vlm_model_name, config.timeout)

    try:
        await database.connect()
        engine = TaskEngine(config, database, immich, vlm)

        # Set up Ctrl+C handler
        loop = asyncio.get_running_loop()

        def signal_handler() -> None:
            logger.info("Received interrupt signal, pausing task...")
            engine.request_cancel()

        loop.add_signal_handler(signal.SIGINT, signal_handler)

        task_id = await engine.create_and_run_task(
            album_ids=album_ids,
            prompt_config=prompt_config,
            concurrency=concurrency,
        )
        logger.info("Task ID: {}", task_id)
    except Exception as exc:
        logger.error("Classification failed: {}", exc)
        sys.exit(1)
    finally:
        await vlm.close()
        await immich.close()
        await database.close()


async def cmd_debug(
    config: Config,
    album_id: str,
    count: int,
    prompt_config_path: str | None,
) -> None:
    """Run a debug batch of classifications."""
    prompt_config = _load_prompt_config(prompt_config_path, config)
    immich = ImmichClient(config.immich_api_url, config.immich_api_key, config.timeout)
    vlm = VLMClient(config.vlm_api_url, config.vlm_api_key, config.vlm_model_name, config.timeout)

    try:
        results = await debug_classify(
            config, immich, vlm,
            album_id=album_id,
            prompt_config=prompt_config,
            count=count,
        )

        if results:
            # Print results as a formatted table
            table_data: list[list[str]] = []
            for r in results:
                if r["status"] == "success":
                    result_str = json.dumps(r["result"], ensure_ascii=False)
                else:
                    result_str = f"ERROR: {r.get('error', 'unknown')}"
                table_data.append([
                    r["asset_id"],
                    r["file_name"],
                    r["status"],
                    result_str,
                ])
            print()
            print(tabulate(
                table_data,
                headers=["Asset ID", "File Name", "Status", "Result"],
                tablefmt="simple",
            ))

            # Print summary and diagnostic hints for failures
            success_count = sum(1 for r in results if r["status"] == "success")
            failed_count = sum(1 for r in results if r["status"] != "success")
            print(f"\n  Total: {len(results)} | Success: {success_count} | Failed: {failed_count}")

            if failed_count > 0:
                print("\n── Diagnostic Details ──")
                for r in results:
                    if r["status"] != "success":
                        print(f"\n  Asset: {r['asset_id']}")
                        print(f"  File:  {r['file_name']}")
                        print(f"  Error: {r.get('error', 'unknown')}")
                        raw = r.get("raw_response")
                        if raw:
                            # Truncate very long responses
                            display_raw = raw[:2000]
                            if len(raw) > 2000:
                                display_raw += f"\n  ... ({len(raw) - 2000} more chars)"
                            print(f"  Raw response:\n    {display_raw}")
                        else:
                            print("  Raw response: (none captured)")

                print("\n── Troubleshooting Tips ──")
                print("  1. Check VLM_API_URL and VLM_MODEL_NAME in .env")
                print("  2. Verify the model supports vision/image input")
                print("  3. Verify the model supports structured output (response_format)")
                print("  4. Try a different model or increase CLASSIFY_TIMEOUT")
                print("  5. Run with LOGURU_LEVEL=DEBUG for full request/response logs")
    except Exception as exc:
        logger.error("Debug classification failed: {}", exc)
        sys.exit(1)
    finally:
        await vlm.close()
        await immich.close()


async def cmd_generate(
    config: Config,
    goal: str,
    output_path: str | None,
    api_url: str | None,
    api_key: str | None,
    model_name: str | None,
) -> None:
    """Generate a prompt config from a natural language description."""
    effective_url = api_url or config.vlm_api_url
    effective_key = api_key or config.vlm_api_key
    effective_model = model_name or config.vlm_model_name

    generator = PromptGenerator(
        api_url=effective_url,
        api_key=effective_key,
        model_name=effective_model,
        timeout=config.timeout,
    )

    try:
        logger.info("Generating prompt config for: {}", goal)
        prompt = await generator.generate(goal)

        # Display the generated config
        print("\n── Generated Prompt Config ──")
        print(f"  Type:          {prompt.name}")
        print(f"  System prompt: {prompt.system_prompt[:120]}...")
        print(f"  User prompt:   {prompt.user_prompt[:120]}...")
        print(f"  Schema fields:")
        for name, sf in prompt.schema.items():
            enum_str = f" (enum: {sf.enum})" if sf.enum else ""
            default_str = f" (default: {sf.default})" if sf.default is not None else ""
            print(f"    - {name} ({sf.field_type}): {sf.description}{enum_str}{default_str}")

        # Show serialized JSON for inspection
        print("\n── Serialized JSON ──")
        print(json.dumps(prompt.to_dict(), indent=2, ensure_ascii=False))

        if output_path:
            export_as_python(prompt, output_path)
            print(f"\n  Exported to: {output_path}")
            print(f"  Test with:   immich-classify debug --album <ALBUM_ID> --prompt-config {output_path}")
        else:
            print("\n  Tip: Add --output <file.py> to export as a reusable config file.")

    except PromptGeneratorError as exc:
        logger.error("Prompt generation failed: {}", exc)
        if exc.raw_response:
            logger.debug("Raw response: {}", exc.raw_response[:2000])
        sys.exit(1)
    except Exception as exc:
        logger.error("Prompt generation failed: {}", exc)
        sys.exit(1)
    finally:
        await generator.close()


async def cmd_status(config: Config, task_id: str | None) -> None:
    """Show task status.

    Per-asset rows in ``classification_results`` are the ground truth; the
    denormalized ``completed_count`` / ``failed_count`` on the ``tasks`` row
    is maintained incrementally and can drift (e.g. legacy data from the
    old double-counting bug, or a crash between ``save_result`` and
    ``increment_task_completed``). Reconcile from rows before displaying so
    the status view always matches reality and always satisfies
    ``completed + failed + pending == total``.
    """
    database = Database(config.database_path)
    try:
        await database.connect()

        if task_id:
            task = await database.get_task(task_id)
            if task is None:
                logger.error("Task {} not found", task_id)
                sys.exit(1)

            # Reconcile counters from per-asset rows (ground truth).
            completed, failed = await database.reset_counts_from_results(task_id)
            summary = await database.get_result_summary(task_id)
            total: int = task["total_count"]
            pending = max(total - completed - failed, 0)

            print(f"Task ID:    {task['task_id']}")
            print(f"Status:     {task['status']}")
            print(f"Albums:     {task['album_ids']}")
            print(f"Total:      {total}")
            print(f"Completed:  {completed}")
            print(f"Failed:     {failed}")
            print(f"Pending:    {pending}")
            print(f"Created:    {task['created_at']}")
            print(f"Updated:    {task['updated_at']}")
            if summary:
                print(f"Results:    {dict(summary)}")
        else:
            tasks = await database.get_all_tasks()
            # Hide cancelled tasks from the summary list — they add noise and
            # their results are already preserved. Users can still inspect
            # a specific cancelled task by passing ``--task <id>``.
            tasks = [t for t in tasks if t["status"] != "cancelled"]
            if not tasks:
                logger.info("No tasks found")
                return

            # Reconcile every task's counters from rows. A ``status`` listing
            # should never lie; the small extra write is cheap.
            table_data: list[list[Any]] = []
            for t in tasks:
                completed, failed = await database.reset_counts_from_results(
                    t["task_id"]
                )
                total = t["total_count"]
                pending = max(total - completed - failed, 0)

                # Extract the prompt type (``BasePrompt.name``) from the
                # stored config so two tasks classifying the same album with
                # different prompts are distinguishable at a glance.
                prompt_name = ""
                try:
                    cfg_raw = t.get("prompt_config") or "{}"
                    cfg = json.loads(cfg_raw)
                    prompt_name = str(cfg.get("name") or "")
                except (ValueError, TypeError):
                    prompt_name = ""

                # Abbreviated album IDs — first 8 chars per album, joined by
                # commas. Same reason: disambiguate multiple runs on the same
                # album at a glance.
                album_ids_list: list[str] = []
                try:
                    parsed_albums: Any = json.loads(t.get("album_ids") or "[]")
                    if isinstance(parsed_albums, list):
                        album_ids_list = [
                            str(a) for a in cast(list[Any], parsed_albums)
                        ]
                except (ValueError, TypeError):
                    album_ids_list = []
                album_abbrev = ",".join(aid[:8] for aid in album_ids_list)

                table_data.append([
                    t["task_id"],
                    t["status"],
                    prompt_name,
                    album_abbrev,
                    f"{completed}/{total}",
                    failed,
                    pending,
                    t["created_at"][:19],
                ])
            print(tabulate(
                table_data,
                headers=[
                    "Task ID",
                    "Status",
                    "Prompt",
                    "Albums",
                    "Progress",
                    "Failed",
                    "Pending",
                    "Created",
                ],
                tablefmt="simple",
            ))
    finally:
        await database.close()


async def cmd_results(
    config: Config,
    task_id: str,
    filters: dict[str, str] | None,
    output_format: str,
) -> None:
    """Query and display classification results."""
    database = Database(config.database_path)
    try:
        await database.connect()

        summary = await database.get_result_summary(task_id)
        logger.info(
            "Task {} result summary: {} (filters={})",
            task_id,
            summary,
            filters,
        )

        results = await database.get_results(task_id, filters)
        logger.info(
            "Query matched {} row(s) out of {} successful result(s)",
            len(results),
            summary.get("success", 0),
        )
        if not results:
            if filters:
                logger.info(
                    "No results matched filters {}. Hints:\n"
                    "  • Filter values are auto-parsed as JSON (e.g. 0 → int, "
                    "true → bool). Check the 'Observed values' log entry "
                    "above to confirm the actual field values and their types.\n"
                    "  • Verify the filter key spelling matches the JSON schema "
                    "produced by your prompt.\n"
                    "  • Enable debug logging (LOGURU_LEVEL=DEBUG) to see the "
                    "full SQL and parameters used for the query.",
                    filters,
                )
            else:
                logger.info("No results found")
            return

        if output_format == "json":
            output: list[dict[str, Any]] = []
            for r in results:
                entry: dict[str, Any] = {
                    "asset_id": r["asset_id"],
                }
                if r["result_json"]:
                    entry.update(json.loads(r["result_json"]))
                output.append(entry)
            print(json.dumps(output, indent=2, ensure_ascii=False))

        elif output_format == "csv":
            # Determine columns dynamically from all results
            all_field_names: set[str] = set()
            for r in results:
                if r["result_json"]:
                    all_field_names.update(json.loads(r["result_json"]).keys())
            sorted_fields = sorted(all_field_names)
            fieldnames = ["asset_id"] + sorted_fields

            string_buffer = io.StringIO()
            writer = csv.DictWriter(string_buffer, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                row: dict[str, Any] = {"asset_id": r["asset_id"]}
                if r["result_json"]:
                    parsed: dict[str, Any] = json.loads(r["result_json"])
                    for key in sorted_fields:
                        value: Any = parsed.get(key, "")
                        if isinstance(value, list):
                            row[key] = "; ".join(str(item) for item in cast(list[Any], value))
                        else:
                            row[key] = value
                writer.writerow(row)
            print(string_buffer.getvalue(), end="")

        else:  # table
            table_data: list[list[str]] = []
            all_keys: list[str] = []
            if results and results[0]["result_json"]:
                first: dict[str, Any] = json.loads(results[0]["result_json"])
                all_keys = list(first.keys())

            for r in results:
                row_data: list[str] = [r["asset_id"]]
                if r["result_json"]:
                    parsed_row: dict[str, Any] = json.loads(r["result_json"])
                    for key in all_keys:
                        val: Any = parsed_row.get(key, "")
                        if isinstance(val, list):
                            row_data.append("; ".join(str(item) for item in cast(list[Any], val)))
                        else:
                            row_data.append(str(val))
                table_data.append(row_data)

            headers = ["Asset ID"] + all_keys
            print(tabulate(table_data, headers=headers, tablefmt="simple"))
    finally:
        await database.close()


async def cmd_pause(config: Config, task_id: str) -> None:
    """Pause a running task."""
    database = Database(config.database_path)
    try:
        await database.connect()
        task = await database.get_task(task_id)
        if task is None:
            logger.error("Task {} not found", task_id)
            sys.exit(1)
        if task["status"] != "running":
            logger.error("Task {} is not running (status: {})", task_id, task["status"])
            sys.exit(1)
        await database.update_task_status(task_id, "paused")
        logger.info("Task {} paused", task_id)
    finally:
        await database.close()


async def cmd_resume(
    config: Config,
    task_id: str,
    concurrency: int | None,
) -> None:
    """Resume a paused, failed, or completed task (retries any non-success assets)."""
    database = Database(config.database_path)
    immich = ImmichClient(config.immich_api_url, config.immich_api_key, config.timeout)
    vlm = VLMClient(config.vlm_api_url, config.vlm_api_key, config.vlm_model_name, config.timeout)

    try:
        await database.connect()
        engine = TaskEngine(config, database, immich, vlm)

        loop = asyncio.get_running_loop()

        def signal_handler() -> None:
            logger.info("Received interrupt signal, pausing task...")
            engine.request_cancel()

        loop.add_signal_handler(signal.SIGINT, signal_handler)

        await engine.resume_task(task_id, concurrency)
    except Exception as exc:
        logger.error("Resume failed: {}", exc)
        sys.exit(1)
    finally:
        await vlm.close()
        await immich.close()
        await database.close()


async def cmd_cancel(config: Config, task_id: str) -> None:
    """Cancel a task (keeps existing results)."""
    database = Database(config.database_path)
    try:
        await database.connect()
        task = await database.get_task(task_id)
        if task is None:
            logger.error("Task {} not found", task_id)
            sys.exit(1)
        await database.update_task_status(task_id, "cancelled")
        logger.info("Task {} cancelled (existing results preserved)", task_id)
    finally:
        await database.close()


async def cmd_app(
    config: Config,
    host: str,
    port: int,
    open_browser: bool,
) -> None:
    """Launch the interactive WebUI (FastAPI + uvicorn).

    The server opens its own :class:`Database` and :class:`ImmichClient` via
    the FastAPI lifespan hook in ``webapp.create_app``; this function just
    builds the app and runs uvicorn. Ctrl+C is handled by uvicorn itself.

    When ``open_browser`` is true, a background task waits for uvicorn to
    report ``started`` and then opens the UI in the user's default browser.
    """
    import webbrowser

    import uvicorn

    from immich_classify.webapp import create_app

    app = create_app(config)
    server_config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
    )
    server = uvicorn.Server(server_config)

    # 0.0.0.0 isn't a browser-reachable address — use localhost in the URL.
    display_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    url = f"http://{display_host}:{port}"
    logger.info("Starting WebUI at {}", url)

    async def _open_when_ready() -> None:
        # Poll the server's started flag instead of using a fixed sleep so we
        # open the browser as soon as possible, but give up after 10s if the
        # server somehow never reports ready (e.g. port bind failure).
        for _ in range(100):
            if server.started:
                break
            await asyncio.sleep(0.1)
        else:
            logger.warning("Server did not report ready within 10s; skipping browser open")
            return
        try:
            opened = webbrowser.open(url, new=2)
        except Exception as exc:  # noqa: BLE001 — webbrowser can raise OS-dependent errors
            logger.warning("Could not open browser automatically: {}", exc)
            return
        if opened:
            logger.info("Opened {} in your default browser", url)
        else:
            logger.info("Browser could not be opened automatically. Visit {} manually.", url)

    if open_browser:
        asyncio.create_task(_open_when_ready())

    await server.serve()


def _parse_filters(filter_args: list[str] | None) -> dict[str, str] | None:
    """Parse filter arguments like 'key=value' into a dict.

    Thin CLI-side wrapper around :func:`parse_filter_strings` that converts
    a raised ``FilterParseError`` into a logged message + ``sys.exit(1)`` so
    the CLI keeps its existing exit-on-bad-input behavior.
    """
    try:
        return parse_filter_strings(filter_args)
    except FilterParseError as exc:
        logger.error("{}", exc)
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="immich-classify",
        description="Classify Immich photos using Vision Language Models",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # albums
    subparsers.add_parser("albums", help="List all Immich albums")

    # classify
    classify_parser = subparsers.add_parser("classify", help="Run classification on an album")
    classify_parser.add_argument("--album", required=True, action="append", dest="albums", help="Album ID to classify (can be specified multiple times)")
    classify_parser.add_argument("--prompt-config", help="Path to custom prompt config Python file")
    classify_parser.add_argument("--concurrency", type=int, help="Override concurrency setting")

    # debug
    debug_parser = subparsers.add_parser("debug", help="Run debug classification batch")
    debug_parser.add_argument("--album", required=True, help="Album ID to sample from")
    debug_parser.add_argument("--count", type=int, default=10, help="Number of images to classify (default: 10)")
    debug_parser.add_argument("--prompt-config", help="Path to custom prompt config Python file")

    # generate
    gen_parser = subparsers.add_parser("generate", help="AI-generate a prompt config from a task description")
    gen_parser.add_argument("--goal", required=True, help="Natural language description of what to classify or detect")
    gen_parser.add_argument("--output", help="Export generated config as a Python file")
    gen_parser.add_argument("--api-url", help="Override VLM_API_URL for the generator model")
    gen_parser.add_argument("--api-key", help="Override VLM_API_KEY for the generator model")
    gen_parser.add_argument("--model", help="Override VLM_MODEL_NAME for the generator model")

    # status
    status_parser = subparsers.add_parser("status", help="Show task status")
    status_parser.add_argument("--task", help="Specific task ID (shows all if omitted)")

    # results
    results_parser = subparsers.add_parser("results", help="Query classification results")
    results_parser.add_argument("--task", required=True, help="Task ID")
    results_parser.add_argument("--filter", action="append", help="Filter by field (e.g. category=people)")
    results_parser.add_argument("--format", choices=["json", "csv", "table"], default="table", help="Output format")

    # pause
    pause_parser = subparsers.add_parser("pause", help="Pause a running task")
    pause_parser.add_argument("--task", required=True, help="Task ID to pause")

    # resume
    resume_parser = subparsers.add_parser("resume", help="Resume a paused task")
    resume_parser.add_argument("--task", required=True, help="Task ID to resume")
    resume_parser.add_argument("--concurrency", type=int, help="Override concurrency setting")

    # cancel
    cancel_parser = subparsers.add_parser("cancel", help="Cancel a task")
    cancel_parser.add_argument("--task", required=True, help="Task ID to cancel")

    # app (WebUI)
    app_parser = subparsers.add_parser(
        "app",
        help="Launch interactive WebUI for filter & browse",
    )
    app_parser.add_argument(
        "--port", type=int, default=8765, help="Port to bind (default: 8765)"
    )
    app_parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)"
    )
    app_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open the WebUI in a browser on startup",
    )

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    config = load_config()

    if args.command == "albums":
        asyncio.run(cmd_albums(config))

    elif args.command == "classify":
        asyncio.run(cmd_classify(
            config,
            album_ids=args.albums,
            prompt_config_path=args.prompt_config,
            concurrency=args.concurrency,
        ))

    elif args.command == "debug":
        asyncio.run(cmd_debug(
            config,
            album_id=args.album,
            count=args.count,
            prompt_config_path=args.prompt_config,
        ))

    elif args.command == "generate":
        asyncio.run(cmd_generate(
            config,
            goal=args.goal,
            output_path=args.output,
            api_url=args.api_url,
            api_key=args.api_key,
            model_name=args.model,
        ))

    elif args.command == "status":
        asyncio.run(cmd_status(config, task_id=args.task))

    elif args.command == "results":
        filters = _parse_filters(args.filter)
        asyncio.run(cmd_results(
            config,
            task_id=args.task,
            filters=filters,
            output_format=args.format,
        ))

    elif args.command == "pause":
        asyncio.run(cmd_pause(config, task_id=args.task))

    elif args.command == "resume":
        asyncio.run(cmd_resume(
            config,
            task_id=args.task,
            concurrency=args.concurrency,
        ))

    elif args.command == "cancel":
        asyncio.run(cmd_cancel(config, task_id=args.task))

    elif args.command == "app":
        asyncio.run(cmd_app(
            config,
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
        ))


if __name__ == "__main__":
    main()
