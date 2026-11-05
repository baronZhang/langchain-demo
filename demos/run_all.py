"""
一键运行 demo（冒烟测试 / 回归验证）
==================================
用法
  uv run demos/run_all.py                # 运行全部 demo（主模型）
  uv run demos/run_all.py 01_core 04_    # 只运行路径包含这些关键字的 demo
  uv run demos/run_all.py --fast         # 全部改用快模型（LLM_FAST_MODEL），几分钟跑完
  uv run demos/run_all.py -j 4           # 4 个 demo 并发执行
  uv run demos/run_all.py --list         # 只列出 demo
约定
  - demos/<章节>/<编号>_<主题>.py 是可独立运行的 demo；以下划线开头的文件是辅助模块（如 MCP Server）
  - 每个 demo 的完整输出写到 .runtime/logs/<章节>__<文件名>.log
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEMOS_DIR = Path(__file__).resolve().parent
ROOT = DEMOS_DIR.parent
LOG_DIR = ROOT / ".runtime" / "logs"


def discover(keywords: list[str]) -> list[Path]:
    demos = sorted(p for p in DEMOS_DIR.glob("*/*.py") if not p.name.startswith("_"))
    if keywords:
        demos = [p for p in demos if any(k in str(p.relative_to(DEMOS_DIR)) for k in keywords)]
    return demos


def run_one(path: Path, env: dict[str, str], timeout: int) -> tuple[Path, bool, float, Path]:
    log = LOG_DIR / f"{path.parent.name}__{path.stem}.log"
    start = time.perf_counter()
    with log.open("w") as fh:
        try:
            proc = subprocess.run([sys.executable, str(path)], cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
            ok = proc.returncode == 0
        except subprocess.TimeoutExpired:
            fh.write(f"\n[run_all] 超时（>{timeout}s）\n")
            ok = False
    return path, ok, time.perf_counter() - start, log


def main() -> None:
    parser = argparse.ArgumentParser(description="批量运行 demo")
    parser.add_argument("keywords", nargs="*", help="按路径关键字过滤")
    parser.add_argument("--fast", action="store_true", help="所有 demo 改用快模型")
    parser.add_argument("-j", "--jobs", type=int, default=1, help="并发数")
    parser.add_argument("--timeout", type=int, default=900, help="单个 demo 超时秒数")
    parser.add_argument("--list", action="store_true", help="只列出 demo")
    args = parser.parse_args()

    demos = discover(args.keywords)
    if args.list:
        for p in demos:
            print(p.relative_to(ROOT))
        return

    env = {**os.environ, "PYTHONUNBUFFERED": "1", "NO_COLOR": "1"}
    if args.fast:
        sys.path.insert(0, str(ROOT / "src"))
        from langchain_demo import get_settings

        env["LLM_MODEL"] = get_settings().llm_fast_model  # 真实环境变量优先级高于 .env
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"共 {len(demos)} 个 demo，并发 {args.jobs}，模型 {env.get('LLM_MODEL', '按 .env')}\n")
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for path, ok, cost, log in pool.map(lambda p: run_one(p, env, args.timeout), demos):
            print(f"{'✅' if ok else '❌'} {cost:6.1f}s  {path.relative_to(ROOT)}" + ("" if ok else f"   日志: {log.relative_to(ROOT)}"))
            results.append(ok)
    print(f"\n通过 {sum(results)}/{len(results)}")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
