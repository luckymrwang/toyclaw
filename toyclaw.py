"""ToyClaw: a tiny OpenClaw-like CLI assistant.

Everything lives in this single file on purpose:
- interactive TUI-ish REPL
- workspace rooted at /tmp/toyclaw
- context files: USER.md, SOUL.md, IDENTITY.md, AGENT.md
- skills installed as plain markdown files
- minimal shell tool support driven by an OpenAI-compatible chat API
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


WORKSPACE = Path("/tmp/toyclaw")
SKILLS_DIR = WORKSPACE / "skills"
SESSIONS_DIR = WORKSPACE / "sessions"
SESSION_LOG = SESSIONS_DIR / "latest.jsonl"
DEFAULT_CONTEXT_FILES = ("USER.md", "SOUL.md", "IDENTITY.md", "AGENT.md")
SHELL_TIMEOUT_SECONDS = 12
MAX_TOOL_STEPS = 4
MAX_TEXT_CHARS = 18_000
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-5.4"


DEFAULT_FILE_CONTENTS = {
    "USER.md": textwrap.dedent(
        """\
        # USER.md

        Describe the human you are helping here.
        Examples:
        - name / nickname
        - language preference
        - working style
        - constraints to remember
        """
    ),
    "SOUL.md": textwrap.dedent(
        """\
        # SOUL.md

        Define the assistant's values, personality, and tone here.
        """
    ),
    "IDENTITY.md": textwrap.dedent(
        """\
        # IDENTITY.md

        Define the assistant's public identity here.
        Example:
        - name
        - vibe
        - style
        """
    ),
    "AGENT.md": textwrap.dedent(
        """\
        # AGENT.md

        Operating notes:
        - help the user directly
        - keep answers concise
        - use shell only when it materially helps
        - avoid destructive commands
        """
    ),
}


@dataclass(frozen=True)
class ChatConfig:
    api_key: str
    base_url: str
    model: str
    temperature: float
    debug_api: bool


class ChatClient:
    """Small OpenAI-compatible /chat/completions client."""

    def __init__(self, config: ChatConfig):
        self.config = config

    def complete(self, messages: list[dict[str, str]]) -> str:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "ToyClaw/0.1",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=60) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(f"API request failed: {exc.reason}") from exc

        data = json.loads(raw)
        if self.config.debug_api:
            save_text(WORKSPACE / "last_api_response.json", json.dumps(data, ensure_ascii=False, indent=2) + "\n")

        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected API response: {raw}") from exc

        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)
        return ""

    def list_models(self) -> list[str]:
        url = self.config.base_url.rstrip("/") + "/models"
        request = Request(
            url,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "User-Agent": "ToyClaw/0.1",
            },
            method="GET",
        )
        try:
            with urlopen(request, timeout=30) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(f"API request failed: {exc.reason}") from exc

        data = json.loads(raw)
        if self.config.debug_api:
            save_text(WORKSPACE / "last_models_response.json", json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        models = data.get("data", [])
        if not isinstance(models, list):
            raise RuntimeError(f"Unexpected models response: {raw}")
        ids = [item.get("id") for item in models if isinstance(item, dict) and item.get("id")]
        return sorted(str(model_id) for model_id in ids)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny OpenClaw-like CLI assistant.")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI-compatible API key.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible base URL, default: %(default)s",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        help="Chat model name, default: %(default)s",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
        help="Sampling temperature, default: %(default)s",
    )
    parser.add_argument("--no-api", action="store_true", help="Start without a model; local commands still work.")
    parser.add_argument(
        "--debug-api",
        action="store_true",
        help="Write raw API responses to /tmp/toyclaw/last_api_response.json.",
    )
    return parser.parse_args(list(argv))


def build_config(args: argparse.Namespace) -> ChatConfig | None:
    if args.no_api:
        return None
    if not args.api_key:
        return None
    return ChatConfig(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        debug_api=args.debug_api,
    )


def ensure_workspace() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    for name in DEFAULT_CONTEXT_FILES:
        path = resolve_context_path(name)
        if not path.exists():
            save_text(path, DEFAULT_FILE_CONTENTS[name].strip() + "\n")


def resolve_context_path(name: str) -> Path:
    clean_name = Path(name).name
    if clean_name not in DEFAULT_CONTEXT_FILES:
        raise ValueError(f"unknown context file: {name}")
    return WORKSPACE / clean_name


def save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def truncate(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def sanitize_filename(name: str) -> str:
    stem = Path(name).name.strip() or "skill.md"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip(".-")
    if not stem:
        stem = "skill.md"
    if not stem.lower().endswith((".md", ".txt")):
        stem += ".md"
    return stem


def list_skills() -> list[Path]:
    if not SKILLS_DIR.exists():
        return []
    return sorted(path for path in SKILLS_DIR.iterdir() if path.is_file())


def format_installed_skill_names() -> str:
    skills = list_skills()
    if not skills:
        return "(none)"
    return "\n".join(f"- {skill.name}" for skill in skills)


def format_context_block(path: Path) -> str:
    content = load_text(path).strip()
    if not content:
        return ""
    return f"\n## {path.name}\n{truncate(content)}\n"


def install_skill(source: str) -> Path:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        request = Request(source, headers={"User-Agent": "ToyClaw/0.1"})
        with urlopen(request, timeout=20) as response:
            content = response.read().decode("utf-8")
        stem = Path(parsed.path or "skill.md").name
    else:
        local_path = Path(source).expanduser()
        content = local_path.read_text(encoding="utf-8")
        stem = local_path.name

    target = SKILLS_DIR / sanitize_filename(stem)
    save_text(target, content)
    return target


def build_system_prompt() -> str:
    sections = []
    for name in DEFAULT_CONTEXT_FILES:
        block = format_context_block(resolve_context_path(name))
        if block:
            sections.append(block)

    skills = []
    for skill in list_skills():
        content = load_text(skill)
        if content:
            skills.append(f"\n## Skill: {skill.name}\n{truncate(content)}\n")

    skills_text = "".join(skills) if skills else "\n(no installed skills)\n"
    skill_name_list = format_installed_skill_names()
    context_text = "".join(sections) if sections else "\n(no workspace context files)\n"

    return textwrap.dedent(
        f"""\
        You are ToyClaw, a tiny OpenClaw-like assistant running in a CLI.
        Workspace root: {WORKSPACE}

        Respond with exactly one JSON object and no surrounding markdown.

        If you can answer directly:
        {{"type":"answer","content":"your final answer"}}

        If you need a shell command:
        {{"type":"shell","command":"the command to run"}}

        Rules:
        - Use shell only when it clearly helps answer the user.
        - Shell runs inside {WORKSPACE}.
        - Prefer short, non-destructive commands.
        - Never use destructive commands, privilege escalation, background jobs, or interactive programs.
        - After receiving shell output, continue and either ask for another command or provide the final answer.
        - Keep final answers concise and useful.
        - If the user mentions a skill by name, apply that skill's markdown instructions.

        Loaded workspace context:
        {context_text}

        Installed skill file names:
        {skill_name_list}

        Installed skills:
        {skills_text}
        """
    ).strip()


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def is_dangerous_shell(command: str) -> str | None:
    if not command:
        return "empty command"

    lowered = command.lower()
    blocked_fragments = [
        " rm ",
        " rm -",
        "sudo ",
        "su ",
        "chmod ",
        "chown ",
        "mkfs",
        "dd ",
        ":(){",
        "shutdown",
        "reboot",
        "halt",
        "launchctl",
        "systemctl",
        "kill ",
        "killall ",
        "pkill ",
        "curl ",
        "wget ",
        "nc ",
        "netcat ",
        "> /dev/",
        ">/dev/",
    ]
    padded = " " + lowered.strip() + " "
    for fragment in blocked_fragments:
        if fragment in padded:
            return f"blocked fragment: {fragment.strip()}"

    if re.search(r"(^|[;&|]\s*)rm(\s|$)", lowered):
        return "rm is blocked"
    if any(token in command for token in ("&", ";;")):
        return "background jobs and command chaining are blocked"
    if re.search(r"\b(vim|vi|nano|emacs|less|more|top|htop|ssh|ftp|python|python3|node)\b", lowered):
        return "interactive programs are blocked"
    return None


def run_shell(command: str) -> str:
    blocked = is_dangerous_shell(command)
    if blocked:
        return f"COMMAND BLOCKED\nReason: {blocked}\nCommand: {command}"

    try:
        completed = subprocess.run(
            ["/bin/bash", "--noprofile", "--norc", "-c", command],
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=SHELL_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = truncate(exc.stdout or "")
        stderr = truncate(exc.stderr or "")
        return "\n".join(
            [
                f"COMMAND TIMED OUT after {SHELL_TIMEOUT_SECONDS}s",
                f"Command: {command}",
                "Stdout:",
                stdout if stdout else "(empty)",
                "",
                "Stderr:",
                stderr if stderr else "(empty)",
            ]
        )

    stdout = truncate(completed.stdout or "")
    stderr = truncate(completed.stderr or "")
    return "\n".join(
        [
            f"Command: {command}",
            f"Exit code: {completed.returncode}",
            "Stdout:",
            stdout if stdout else "(empty)",
            "",
            "Stderr:",
            stderr if stderr else "(empty)",
        ]
    )


def print_block(title: str, body: str) -> None:
    print(f"\n--- {title} ---")
    print(body)
    print("--- end ---")


def append_session_log(role: str, content: str) -> None:
    entry = {"ts": time.time(), "role": role, "content": content}
    with SESSION_LOG.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def local_help() -> str:
    return textwrap.dedent(
        f"""\
        Commands:
          /help                         show this help
          /quit                         exit
          /workspace                    show workspace path
          /context                      list context files
          /cat USER.md                  print a context file
          /edit USER.md text...         replace a context file
          /append USER.md text...       append to a context file
          /skills                       list installed skills
          /install-skill PATH_OR_URL    install a markdown skill
          /models                       list API models, if the server supports it
          /shell COMMAND                run a non-destructive shell command in {WORKSPACE}

        Natural-language questions require OPENAI_API_KEY or --api-key.
        """
    ).strip()


def maybe_answer_without_model(raw: str) -> str | None:
    normalized = raw.lower()
    cpu_patterns = ("cpu", "进程", "占用", "process")
    if all(token in normalized or token in raw for token in cpu_patterns[:3]) or (
        "cpu" in normalized and "process" in normalized
    ):
        result = run_shell("ps -axo pid,pcpu,pmem,comm | sort -k2 -nr | head -n 8")
        return "我没有可用模型，但可以直接执行一个安全的本地查询：\n\n" + result
    return None


def maybe_handle_local_query(raw: str, client: ChatClient | None = None) -> str | None:
    if not raw:
        return ""

    if raw in {"/q", "/quit", "/exit"}:
        raise EOFError

    if raw in {"/help", "help", "?"}:
        return local_help()

    if raw == "/workspace":
        return str(WORKSPACE)

    if raw == "/context":
        rows = []
        for name in DEFAULT_CONTEXT_FILES:
            path = resolve_context_path(name)
            rows.append(f"- {name} ({path.stat().st_size} bytes)")
        return "\n".join(rows)

    if raw.startswith("/cat "):
        name = raw.split(maxsplit=1)[1]
        try:
            return load_text(resolve_context_path(name)).strip() or "(empty)"
        except ValueError as exc:
            return str(exc)

    if raw.startswith("/edit "):
        parts = raw.split(maxsplit=2)
        if len(parts) < 3:
            return "Usage: /edit USER.md text..."
        try:
            path = resolve_context_path(parts[1])
        except ValueError as exc:
            return str(exc)
        save_text(path, parts[2].rstrip() + "\n")
        return f"Updated {path}"

    if raw.startswith("/append "):
        parts = raw.split(maxsplit=2)
        if len(parts) < 3:
            return "Usage: /append USER.md text..."
        try:
            path = resolve_context_path(parts[1])
        except ValueError as exc:
            return str(exc)
        with path.open("a", encoding="utf-8") as file:
            file.write(parts[2].rstrip() + "\n")
        return f"Appended to {path}"

    if raw == "/skills":
        skills = list_skills()
        if not skills:
            return "No skills installed. Use /install-skill PATH_OR_URL."
        return "\n".join(f"- {skill.name} ({skill.stat().st_size} bytes)" for skill in skills)

    if raw.startswith("/install-skill "):
        source = raw.split(maxsplit=1)[1].strip()
        try:
            target = install_skill(source)
        except Exception as exc:  # noqa: BLE001 - user-facing CLI should report install failures.
            return f"Failed to install skill: {exc}"
        return f"Installed skill: {target}"

    if raw == "/models":
        if client is None:
            return "No model configured. Set OPENAI_API_KEY, pass --api-key, or use /help."
        try:
            models = client.list_models()
        except Exception as exc:  # noqa: BLE001 - user-facing CLI should report API failures.
            return f"Failed to list models: {exc}"
        return "\n".join(f"- {model}" for model in models) if models else "(no models returned)"

    if raw.startswith("/shell "):
        command = raw.split(maxsplit=1)[1].strip()
        return run_shell(command)

    if client is None:
        return maybe_answer_without_model(raw) or "No model configured. Set OPENAI_API_KEY, pass --api-key, or use /help."

    return None


def run_agent_turn(client: ChatClient, history: list[dict[str, str]], user_input: str) -> str:
    system_prompt = build_system_prompt()
    working_messages = list(history)
    working_messages.append({"role": "user", "content": user_input})

    for _ in range(MAX_TOOL_STEPS):
        response_text = client.complete([{"role": "system", "content": system_prompt}] + working_messages)
        if not response_text.strip():
            message = textwrap.dedent(
                f"""\
                API returned an empty assistant message for model '{client.config.model}'.
                Try /models to see supported model IDs, then restart with --model=MODEL_ID.
                You can also restart with --debug-api and inspect {WORKSPACE / 'last_api_response.json'}.
                """
            ).strip()
            history.extend(
                [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": message},
                ]
            )
            return message

        action = extract_first_json_object(response_text)
        if not action:
            history.extend(
                [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": response_text.strip()},
                ]
            )
            return response_text.strip()

        action_type = action.get("type")
        if action_type == "answer":
            content = str(action.get("content", "")).strip()
            history.extend(
                [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": content},
                ]
            )
            return content

        if action_type == "shell":
            command = str(action.get("command", "")).strip()
            shell_result = run_shell(command)
            print_block(f"shell: {command}", shell_result)
            working_messages.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
            working_messages.append(
                {
                    "role": "user",
                    "content": "Shell result:\n" + shell_result,
                }
            )
            continue

        fallback = response_text.strip() or json.dumps(action, ensure_ascii=False)
        history.extend(
            [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": fallback},
            ]
        )
        return fallback

    timeout_message = "I reached the maximum number of shell steps for one turn. Please narrow the request."
    history.extend(
        [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": timeout_message},
        ]
    )
    return timeout_message


def print_banner(config: ChatConfig | None) -> None:
    model_line = f"{config.model} via {config.base_url}" if config else "no model configured"
    print(
        textwrap.dedent(
            f"""\
            ToyClaw
            workspace: {WORKSPACE}
            model: {model_line}
            type /help for commands, /quit to exit
            """
        ).strip()
    )


def repl(client: ChatClient | None) -> int:
    history: list[dict[str, str]] = []
    while True:
        try:
            raw = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return 0

        try:
            local_reply = maybe_handle_local_query(raw, client)
        except EOFError:
            print("\nbye")
            return 0

        if local_reply is not None:
            append_session_log("user", raw)
            append_session_log("assistant", local_reply)
            print(f"\nclaw> {local_reply}")
            continue

        append_session_log("user", raw)
        try:
            reply = run_agent_turn(client, history, raw)
        except Exception as exc:  # noqa: BLE001 - keep the REPL alive after API failures.
            reply = f"Error: {exc}"

        append_session_log("assistant", reply)
        print(f"\nclaw> {reply}")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    ensure_workspace()

    config = build_config(args)
    client = ChatClient(config) if config else None
    print_banner(config)
    return repl(client)


if __name__ == "__main__":
    raise SystemExit(main())
