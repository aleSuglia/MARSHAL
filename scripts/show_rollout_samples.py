#!/usr/bin/env python3
"""Pretty-print the prompt/response samples the agentic rollout pipeline logs.

agentic_rollout_pipeline.py dumps a JSON array of
{"prompt", "response", "episode_score", "llm_raw_text"} to logger.info every
logging_steps, which ends up buried in a single long, colour-coded line in
log_rank_DRIVER_0_1.log. This pulls those arrays back out and prints them
readably.

Usage:
    python scripts/show_rollout_samples.py <run_dir_or_log_file> [-n N] [--raw] [-o OUT] [--stdout]

    <run_dir_or_log_file>  Either a run directory (e.g.
                           runs/playpen_wordle_rollout/20260705-161341) or a
                           direct path to a log_rank_DRIVER_*.log file.
    -n N                   Only show the first N samples per logged step
                           (default: all).
    --raw                  Also print the raw llm_raw_text (includes special
                           tokens like <|im_start|>/<|im_end|>).
    -o OUT                 Output text file path (default: rollout_samples.txt
                           next to the log file / inside the run directory).
    --stdout               Print to stdout instead of writing a file.
"""
import argparse
import json
import sys
from pathlib import Path


def find_driver_log(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = sorted(path.glob("logs/log_rank_DRIVER_*.log"))
    if not candidates:
        candidates = sorted(path.glob("**/log_rank_DRIVER_*.log"))
    if not candidates:
        sys.exit(f"no log_rank_DRIVER_*.log found under {path}")
    return candidates[0]


def extract_sample_arrays(log_path: Path):
    """Yield each JSON array of samples logged in the file, in order."""
    marker = '[{"prompt":'
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            idx = line.find(marker)
            if idx == -1:
                continue
            samples, _ = json.JSONDecoder().raw_decode(line, idx)
            yield samples


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="run directory or log_rank_DRIVER_*.log file")
    parser.add_argument("-n", type=int, default=None, help="max samples to show per logged step")
    parser.add_argument("--raw", action="store_true", help="also print llm_raw_text")
    parser.add_argument("-o", "--output", type=Path, default=None, help="output text file path")
    parser.add_argument("--stdout", action="store_true", help="print to stdout instead of writing a file")
    args = parser.parse_args()

    log_path = find_driver_log(args.path)

    lines = []
    step_num = 0
    for samples in extract_sample_arrays(log_path):
        step_num += 1
        shown = samples if args.n is None else samples[: args.n]
        lines.append(f"\n{'=' * 80}\nLOGGED STEP {step_num}  ({len(samples)} samples, showing {len(shown)})\n{'=' * 80}")
        for i, sample in enumerate(shown):
            lines.append(f"\n--- sample {i} | episode_score={sample.get('episode_score')} ---")
            lines.append("\n[PROMPT]")
            lines.append(sample.get("prompt", ""))
            lines.append("\n[RESPONSE]")
            lines.append(sample.get("response", ""))
            if args.raw:
                lines.append("\n[LLM_RAW_TEXT]")
                lines.append(sample.get("llm_raw_text", ""))

    if step_num == 0:
        sys.exit(f"no sample dumps found in {log_path} (did the run reach a logging_steps boundary?)")

    text = "\n".join(lines) + "\n"

    if args.stdout:
        print(text)
        return

    out_path = args.output
    if out_path is None:
        out_path = log_path.parent / "rollout_samples.txt" if args.path.is_dir() else log_path.with_name(
            log_path.stem + "_samples.txt"
        )
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote {step_num} logged step(s) to {out_path}")


if __name__ == "__main__":
    main()
