"""
Report generator: reads JSONL benchmark results and produces a summary.

Outputs:
- Markdown summary with tables answering the 4 key engineering questions
- Best configurations per metric
- Bottleneck analysis (routing vs fetch vs H2D vs MaxSim)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List


def load_results(path: Path) -> List[dict]:
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def generate_report(results: List[dict], output_path: Path):
    shard_results = [r for r in results if r.get("pipeline") == "shard_routed"]
    baseline_results = [r for r in results if r.get("pipeline", "").startswith("baseline")]

    lines = [
        "# Shard Late-Interaction Benchmark Report",
        "",
        f"**Generated**: {time.strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Total configurations tested**: {len(results)}",
        f"**Shard-routed configs**: {len(shard_results)}",
        f"**Baselines**: {len(baseline_results)}",
        "",
    ]

    # Baselines table
    if baseline_results:
        lines += [
            "## Baselines",
            "",
            "| Pipeline | Corpus | R@10 | R@100 | MRR@10 | p50 (ms) | p95 (ms) | QPS |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in baseline_results:
            lines.append(
                f"| {r.get('pipeline', '?')} "
                f"| {r.get('corpus_size', '?'):,} "
                f"| {r.get('recall_at_10', 0):.4f} "
                f"| {r.get('recall_at_100', 0):.4f} "
                f"| {r.get('mrr_at_10', 0):.4f} "
                f"| {r.get('p50_total_ms', 0):.1f} "
                f"| {r.get('p95_total_ms', 0):.1f} "
                f"| {r.get('qps', 0):.1f} |"
            )
        lines.append("")

    # Shard-routed results table
    if shard_results:
        lines += [
            "## Shard-Routed Results",
            "",
            "| Compression | Layout | Shards | MaxDocs | Transfer | R@10 | R@100 | p50 (ms) | p95 (ms) | QPS | H2D MB/q |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in sorted(shard_results, key=lambda x: -x.get("recall_at_10", 0)):
            h2d_mb = r.get("mean_h2d_bytes", 0) / 1e6
            lines.append(
                f"| {r.get('compression', '?')} "
                f"| {r.get('layout', '?')} "
                f"| {r.get('top_shards', '?')} "
                f"| {r.get('max_docs_exact', '?'):,} "
                f"| {r.get('transfer_mode', '?')} "
                f"| {r.get('recall_at_10', 0):.4f} "
                f"| {r.get('recall_at_100', 0):.4f} "
                f"| {r.get('p50_total_ms', 0):.1f} "
                f"| {r.get('p95_total_ms', 0):.1f} "
                f"| {r.get('qps', 0):.1f} "
                f"| {h2d_mb:.1f} |"
            )
        lines.append("")

    # Key questions
    lines += _bottleneck_analysis(shard_results)
    lines += _routing_quality_analysis(shard_results)
    lines += _layout_comparison(shard_results)
    lines += _memory_analysis(shard_results)
    lines += _conclusion(shard_results, baseline_results)

    report = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"Report written to {output_path}")
    return report


def _bottleneck_analysis(results: List[dict]) -> List[str]:
    if not results:
        return []

    lines = [
        "## Question 1: Where is the bottleneck?",
        "",
        "Fraction of p50 latency spent in each stage:",
        "",
        "| Config | Routing | Fetch | H2D | MaxSim | Total (ms) |",
        "|---|---|---|---|---|---|",
    ]

    for r in results[:10]:
        total = r.get("p50_total_ms", 1)
        if total <= 0:
            continue
        route_f = r.get("p50_routing_ms", 0) / total * 100
        fetch_f = r.get("p50_fetch_ms", 0) / total * 100
        h2d_f = r.get("p50_h2d_ms", 0) / total * 100
        ms_f = r.get("p50_maxsim_ms", 0) / total * 100
        label = f"{r.get('compression','?')}/{r.get('top_shards','?')}sh/{r.get('max_docs_exact','?')}d"
        lines.append(f"| {label} | {route_f:.0f}% | {fetch_f:.0f}% | {h2d_f:.0f}% | {ms_f:.0f}% | {total:.1f} |")

    lines.append("")
    return lines


def _routing_quality_analysis(results: List[dict]) -> List[str]:
    if not results:
        return []

    lines = [
        "## Question 2: Does routing preserve recall?",
        "",
        "Recall@10 by routing budget (top_shards):",
        "",
        "| top_shards | Best R@10 | Worst R@10 | Mean R@10 |",
        "|---|---|---|---|",
    ]

    by_shards: dict = {}
    for r in results:
        ts = r.get("top_shards")
        if ts is not None:
            by_shards.setdefault(ts, []).append(r.get("recall_at_10", 0))

    for ts in sorted(by_shards.keys()):
        vals = by_shards[ts]
        lines.append(f"| {ts} | {max(vals):.4f} | {min(vals):.4f} | {sum(vals)/len(vals):.4f} |")

    lines.append("")
    return lines


def _layout_comparison(results: List[dict]) -> List[str]:
    if not results:
        return []

    lines = [
        "## Question 3: Does contiguous layout matter?",
        "",
    ]

    grouped = [r for r in results if r.get("layout") == "centroid_grouped"]
    random = [r for r in results if r.get("layout") == "random"]

    if grouped and random:
        g_r10 = sum(r.get("recall_at_10", 0) for r in grouped) / len(grouped)
        r_r10 = sum(r.get("recall_at_10", 0) for r in random) / len(random)
        g_p50 = sum(r.get("p50_total_ms", 0) for r in grouped) / len(grouped)
        r_p50 = sum(r.get("p50_total_ms", 0) for r in random) / len(random)

        lines += [
            "| Layout | Mean R@10 | Mean p50 (ms) | Configs |",
            "|---|---|---|---|",
            f"| centroid_grouped | {g_r10:.4f} | {g_p50:.1f} | {len(grouped)} |",
            f"| random | {r_r10:.4f} | {r_p50:.1f} | {len(random)} |",
            "",
        ]
    else:
        lines.append("Insufficient data for layout comparison.\n")

    return lines


def _memory_analysis(results: List[dict]) -> List[str]:
    if not results:
        return []

    lines = [
        "## Question 4: Does GPU memory stay flat?",
        "",
    ]

    gpu_vals = [(r.get("corpus_size", 0), r.get("gpu_allocated_gb", 0)) for r in results if r.get("gpu_allocated_gb")]
    if gpu_vals:
        by_corpus: dict = {}
        for cs, gb in gpu_vals:
            by_corpus.setdefault(cs, []).append(gb)
        lines += [
            "| Corpus Size | Mean GPU HBM (GB) | Max GPU HBM (GB) |",
            "|---|---|---|",
        ]
        for cs in sorted(by_corpus.keys()):
            vals = by_corpus[cs]
            lines.append(f"| {cs:,} | {sum(vals)/len(vals):.2f} | {max(vals):.2f} |")
        lines.append("")
    else:
        lines.append("No GPU memory data available.\n")

    return lines


def _conclusion(shard_results: List[dict], baseline_results: List[dict]) -> List[str]:
    lines = [
        "## Conclusion",
        "",
    ]

    if not shard_results:
        lines.append("No shard-routed results to analyze.")
        return lines

    best = max(shard_results, key=lambda r: r.get("recall_at_10", 0))
    fastest = min(shard_results, key=lambda r: r.get("p50_total_ms", float("inf")))

    lines += [
        f"**Best recall**: R@10={best.get('recall_at_10', 0):.4f} "
        f"(shards={best.get('top_shards')}, docs={best.get('max_docs_exact')}, "
        f"compression={best.get('compression')}, layout={best.get('layout')})",
        "",
        f"**Lowest latency**: p50={fastest.get('p50_total_ms', 0):.1f}ms "
        f"(shards={fastest.get('top_shards')}, docs={fastest.get('max_docs_exact')}, "
        f"compression={fastest.get('compression')}, transfer={fastest.get('transfer_mode')})",
        "",
    ]

    # Compare vs baselines
    for b in baseline_results:
        b_r10 = b.get("recall_at_10", 0)
        b_p50 = b.get("p50_total_ms", 0)
        lines.append(
            f"**vs {b.get('pipeline', '?')}**: baseline R@10={b_r10:.4f} p50={b_p50:.1f}ms"
        )

    lines += [
        "",
        "### Does this architecture deserve a second iteration?",
        "",
        "Answer based on the data above:",
        "- If R@10 is within 1-3 points of brute-force: YES for quality",
        "- If H2D is <50% of total latency: YES for bandwidth",
        "- If GPU memory stays flat across corpus sizes: YES for scaling",
        "- If contiguous layout beats random: YES for the core thesis",
        "",
        "---",
        "*Generated by `benchmarks/shard_bench/report.py`*",
    ]
    return lines


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m benchmarks.shard_bench.report <results.jsonl> [output.md]")
        sys.exit(1)

    results_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else results_path.with_suffix(".md")

    results = load_results(results_path)
    generate_report(results, output_path)


if __name__ == "__main__":
    main()
