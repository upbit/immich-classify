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
from immich_classify.database import Database
from immich_classify.engine import TaskEngine, debug_classify
from immich_classify.immich_client import ImmichClient
from immich_classify.prompt_base import BasePrompt
from immich_classify.prompts.classification import ClassificationPrompt
from immich_classify.prompt_generator import PromptGenerator, PromptGeneratorError, export_as_python
from immich_classify.vlm_client import VLMClient


def _load_prompt_config(path: str | None) -> BasePrompt:
    """Load a BasePrompt from a Python file or return default.

    Args:
        path: Optional path to a Python file containing a BasePrompt instance.
            When *None*, the built-in ``ClassificationPrompt`` is used.

    Returns:
        BasePrompt instance.
    """
    if path is None:
        return ClassificationPrompt()

    spec = importlib.util.spec_from_file_location("prompt_config", path)
    if spec is None or spec.loader is None:
        logger.error("Cannot load prompt config from {}", path)
        sys.exit(1)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    prompt = getattr(module, "prompt", None)
    if prompt is None:
        prompt = getattr(module, "PROMPT", None)
    if prompt is None:
        # Try to find any BasePrompt instance
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, BasePrompt):
                prompt = attr
                break

    if not isinstance(prompt, BasePrompt):
        logger.error(
            "Could not find a BasePrompt instance in {}. "
            "Define a variable named 'prompt' or 'PROMPT'.",
            path,
        )
        sys.exit(1)

    return prompt


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
    prompt_config = _load_prompt_config(prompt_config_path)
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
    prompt_config = _load_prompt_config(prompt_config_path)
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
                    r["asset_id"][:12] + "...",
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
        print(f"  Type:          {prompt.prompt_type}")
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
    """Show task status."""
    database = Database(config.database_path)
    try:
        await database.connect()

        if task_id:
            task = await database.get_task(task_id)
            if task is None:
                logger.error("Task {} not found", task_id)
                sys.exit(1)

            summary = await database.get_result_summary(task_id)
            print(f"Task ID:    {task['task_id']}")
            print(f"Status:     {task['status']}")
            print(f"Albums:     {task['album_ids']}")
            print(f"Total:      {task['total_count']}")
            print(f"Completed:  {task['completed_count']}")
            print(f"Failed:     {task['failed_count']}")
            print(f"Created:    {task['created_at']}")
            print(f"Updated:    {task['updated_at']}")
            if summary:
                print(f"Results:    {dict(summary)}")
        else:
            tasks = await database.get_all_tasks()
            if not tasks:
                logger.info("No tasks found")
                return

            table_data = [
                [
                    t["task_id"][:12] + "...",
                    t["status"],
                    f"{t['completed_count']}/{t['total_count']}",
                    t["failed_count"],
                    t["created_at"][:19],
                ]
                for t in tasks
            ]
            print(tabulate(
                table_data,
                headers=["Task ID", "Status", "Progress", "Failed", "Created"],
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

        results = await database.get_results(task_id, filters)
        if not results:
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
                row_data: list[str] = [r["asset_id"][:12] + "..."]
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
    """Resume a paused task."""
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


def _parse_filters(filter_args: list[str] | None) -> dict[str, str] | None:
    """Parse filter arguments like 'key=value' into a dict."""
    if not filter_args:
        return None
    filters: dict[str, str] = {}
    for f in filter_args:
        if "=" not in f:
            logger.error("Invalid filter format: '{}'. Use key=value.", f)
            sys.exit(1)
        key, value = f.split("=", 1)
        filters[key.strip()] = value.strip()
    return filters


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


if __name__ == "__main__":
    main()
