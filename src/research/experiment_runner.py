from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from config.settings import settings
from src.research.metrics import (
    build_coverage_report,
    build_leakage_report,
    summarize_experiment_results,
)
from src.research.validation import ValidationWindow, split_dataframe_by_window


class ExperimentRunner:
    def __init__(self, output_root: Path | None = None):
        self.output_root = Path(output_root or settings.REPORTS_DIR / "experiments")
        self.output_root.mkdir(parents=True, exist_ok=True)

    def run_walk_forward(
        self,
        dataset: dict[str, pd.DataFrame],
        windows: list[ValidationWindow],
        model_factory: Callable[[], Any],
        backtester: Any,
        threshold: float = 0.6,
        ticker_names: dict[str, str] | None = None,
        experiment_config: dict[str, Any] | None = None,
        min_train_rows_per_ticker: int = 60,
        min_test_rows_per_ticker: int = 10,
        run_name: str | None = None,
        market_status_map: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        ticker_names = ticker_names or {}
        all_results: list[dict[str, Any]] = []
        feature_importance_frames: list[pd.DataFrame] = []

        for window in windows:
            train_frames = []
            for df in dataset.values():
                train_df, _ = split_dataframe_by_window(df, window)
                if len(train_df) >= min_train_rows_per_ticker:
                    train_frames.append(train_df)

            if not train_frames:
                continue

            model = model_factory()
            full_train_df = pd.concat(train_frames, ignore_index=True)
            model.train(full_train_df)
            if not getattr(model, "is_trained", False):
                continue

            importance_frame = self._extract_feature_importance(model, window.fold)
            if importance_frame is not None and not importance_frame.empty:
                feature_importance_frames.append(importance_frame)

            for code, df in dataset.items():
                _, test_df = split_dataframe_by_window(df, window)
                if len(test_df) < min_test_rows_per_ticker:
                    continue

                probs = model.predict_batch(test_df)
                result = backtester.run(test_df, probs, threshold=threshold, code=code)
                result["code"] = code
                result["name"] = ticker_names.get(code, code)
                result["fold"] = window.fold
                result["train_start"] = window.train_start
                result["train_end"] = window.train_end
                result["test_start"] = window.test_start
                result["test_end"] = window.test_end
                result["validation_mode"] = window.mode
                if market_status_map is not None:
                    result["regime"] = market_status_map.get(window.test_end, "Unknown Market")
                all_results.append(result)

        summary = summarize_experiment_results(all_results)
        run_dir = self._create_run_dir(run_name)
        config_payload = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_name": run_name,
            "windows": [window.to_dict() for window in windows],
            **(experiment_config or {}),
        }
        self._write_artifacts(run_dir, config_payload, summary, all_results, feature_importance_frames)
        self._write_diagnostics(run_dir, dataset, summary)
        return {
            "results": all_results,
            "summary": summary,
            "output_dir": str(run_dir),
            "config": config_payload,
        }

    def run_benchmark_suite(
        self,
        dataset: dict[str, pd.DataFrame],
        model_specs: list[tuple[str, Callable[[], Any]]],
        validation_specs: list[tuple[str, list[ValidationWindow]]],
        backtester: Any,
        threshold: float = 0.6,
        ticker_names: dict[str, str] | None = None,
        suite_name: str = "benchmark_suite",
        market_status_map: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        suite_root = self.output_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suite_name}"
        suite_root.mkdir(parents=True, exist_ok=True)

        benchmark_rows: list[dict[str, Any]] = []
        experiment_runs: list[dict[str, Any]] = []

        for model_name, factory in model_specs:
            for mode_name, windows in validation_specs:
                run = self.run_walk_forward(
                    dataset=dataset,
                    windows=windows,
                    model_factory=factory,
                    backtester=backtester,
                    threshold=threshold,
                    ticker_names=ticker_names,
                    experiment_config={
                        "suite_name": suite_name,
                        "model_name": model_name,
                        "validation": {"mode": mode_name},
                    },
                    run_name=f"{suite_name}_{model_name}_{mode_name}",
                    market_status_map=market_status_map,
                )
                summary = run["summary"]
                benchmark_rows.append(
                    {
                        "model_name": model_name,
                        "validation_mode": mode_name,
                        "result_count": summary.get("result_count", 0),
                        "avg_total_return": summary.get("avg_total_return", 0.0),
                        "avg_annual_return": summary.get("avg_annual_return", 0.0),
                        "avg_max_drawdown": summary.get("avg_max_drawdown", 0.0),
                        "avg_sharpe": summary.get("avg_sharpe", 0.0),
                        "avg_calmar": summary.get("avg_calmar", 0.0),
                        "avg_turnover": summary.get("avg_turnover", 0.0),
                        "positive_ratio": summary.get("positive_ratio", 0.0),
                        "overall_win_rate": summary.get("overall_win_rate", 0.0),
                        "output_dir": run.get("output_dir"),
                    }
                )
                experiment_runs.append(run)

        benchmark_df = pd.DataFrame(benchmark_rows)
        selection = self._select_champion_challenger(benchmark_df)
        history = self._build_benchmark_history(benchmark_df)
        benchmark_df.to_csv(suite_root / "benchmark_summary.csv", index=False)
        (suite_root / "benchmark_summary.json").write_text(
            json.dumps(benchmark_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (suite_root / "benchmark_selection.json").write_text(
            json.dumps(selection, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (suite_root / "benchmark_history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        markdown_lines = ["# Benchmark Summary", ""]
        for row in benchmark_rows:
            markdown_lines.append(
                f"- {row['model_name']} / {row['validation_mode']}: avg_return={row['avg_total_return']:.4f} avg_sharpe={row['avg_sharpe']:.4f} avg_max_drawdown={row['avg_max_drawdown']:.4f}"
            )
        markdown_lines.extend(
            [
                "",
                "## Selection",
                "",
                f"- Champion: {selection.get('champion', {}).get('model_name', '-')}/{selection.get('champion', {}).get('validation_mode', '-')}",
                f"- Challenger: {selection.get('challenger', {}).get('model_name', '-')}/{selection.get('challenger', {}).get('validation_mode', '-')}",
                f"- Entries: {len(history)}",
            ]
        )
        (suite_root / "benchmark_summary.md").write_text("\n".join(markdown_lines), encoding="utf-8")

        return {
            "suite_dir": str(suite_root),
            "benchmark_rows": benchmark_rows,
            "selection": selection,
            "history": history,
            "runs": experiment_runs,
        }

    def persist_selected_model(
        self,
        dataset: dict[str, pd.DataFrame],
        model_name: str,
        model_factory: Callable[[], Any],
    ) -> dict[str, Any]:
        model = model_factory()
        frames = [df for df in dataset.values() if not df.empty]
        if not frames:
            return {"saved": False, "reason": "empty dataset"}

        full_train_df = pd.concat(frames, ignore_index=True)
        model.train(full_train_df)
        if not getattr(model, "is_trained", False):
            return {"saved": False, "reason": f"{model_name} training failed"}

        save_model = getattr(model, "save_model", None)
        if not callable(save_model):
            return {"saved": False, "reason": f"{model_name} has no save_model"}

        save_model()
        return {"saved": True, "model_name": model_name}

    @staticmethod
    def _select_champion_challenger(benchmark_df: pd.DataFrame) -> dict[str, Any]:
        if benchmark_df.empty:
            return {"champion": {}, "challenger": {}}

        ranking = benchmark_df.copy()
        ranking = ranking[ranking["result_count"].astype(float) > 0].copy()
        if ranking.empty:
            return {"champion": {}, "challenger": {}}
        ranking["selection_score"] = (
            ranking["avg_total_return"].astype(float)
            + 0.30 * ranking["avg_sharpe"].astype(float)
            + 0.10 * ranking["positive_ratio"].astype(float)
            - 0.50 * ranking["avg_max_drawdown"].astype(float)
        )
        ranking = ranking.sort_values(by="selection_score", ascending=False).reset_index(drop=True)

        champion = ranking.iloc[0].to_dict()
        challenger = ranking.iloc[1].to_dict() if len(ranking) > 1 else {}
        return {
            "champion": champion,
            "challenger": challenger,
        }

    @staticmethod
    def _build_benchmark_history(benchmark_df: pd.DataFrame) -> list[dict[str, Any]]:
        if benchmark_df.empty:
            return []
        history_df = benchmark_df.copy().sort_values(
            by=["model_name", "validation_mode"],
            ascending=[True, True],
        )
        return history_df.to_dict(orient="records")

    def _create_run_dir(self, run_name: str | None) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = f"_{run_name}" if run_name else ""
        run_dir = self.output_root / f"{timestamp}{safe_name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _extract_feature_importance(self, model: Any, fold: int) -> pd.DataFrame | None:
        if not hasattr(model, "get_feature_importance"):
            return None
        frame = model.get_feature_importance()
        if frame is None or frame.empty:
            return None
        enriched = frame.copy()
        enriched.insert(0, "fold", fold)
        return enriched

    def _write_artifacts(
        self,
        run_dir: Path,
        config_payload: dict[str, Any],
        summary: dict[str, Any],
        results: list[dict[str, Any]],
        feature_importance_frames: list[pd.DataFrame],
    ) -> None:
        (run_dir / "config.json").write_text(
            json.dumps(config_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (run_dir / "metrics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        trades_rows = []
        equity_rows = []
        daily_pnl_rows = []
        position_rows = []
        result_rows = []

        scalar_keys = {
            "code",
            "name",
            "fold",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "validation_mode",
            "total_return",
            "annual_return",
            "win_rate",
            "num_trades",
            "final_equity",
            "max_drawdown",
            "volatility",
            "sharpe",
            "calmar",
            "turnover",
            "avg_holding_days",
        }

        for result in results:
            result_rows.append({key: result.get(key) for key in scalar_keys})

            base_fields = {
                "code": result.get("code"),
                "name": result.get("name"),
                "fold": result.get("fold"),
            }
            for trade in result.get("trades", []):
                trades_rows.append({**base_fields, **trade})
            for point in result.get("equity_curve", []):
                equity_rows.append({**base_fields, **point})
            for point in result.get("daily_pnl", []):
                daily_pnl_rows.append({**base_fields, **point})
            for point in result.get("position_history", []):
                position_rows.append({**base_fields, **point})

        pd.DataFrame(result_rows).to_csv(run_dir / "results.csv", index=False)
        pd.DataFrame(trades_rows).to_csv(run_dir / "trades.csv", index=False)
        pd.DataFrame(equity_rows).to_csv(run_dir / "equity_curve.csv", index=False)
        pd.DataFrame(daily_pnl_rows).to_csv(run_dir / "daily_pnl.csv", index=False)
        pd.DataFrame(position_rows).to_csv(run_dir / "position_history.csv", index=False)

        feature_importance_df = (
            pd.concat(feature_importance_frames, ignore_index=True)
            if feature_importance_frames
            else pd.DataFrame(columns=["fold", "feature", "importance"])
        )
        feature_importance_df.to_csv(run_dir / "feature_importance.csv", index=False)

        summary_lines = [
            "# Validation Summary",
            "",
            f"- Result count: {summary.get('result_count', 0)}",
            f"- Average total return: {summary.get('avg_total_return', 0.0):.4f}",
            f"- Average annual return: {summary.get('avg_annual_return', 0.0):.4f}",
            f"- Average max drawdown: {summary.get('avg_max_drawdown', 0.0):.4f}",
            f"- Average sharpe: {summary.get('avg_sharpe', 0.0):.4f}",
            f"- Average calmar: {summary.get('avg_calmar', 0.0):.4f}",
            f"- Average turnover: {summary.get('avg_turnover', 0.0):.4f}",
            f"- Total trades: {summary.get('total_trades', 0)}",
        ]
        regime_breakdown = summary.get("regime_breakdown", {}) or {}
        if regime_breakdown:
            summary_lines.extend(["", "## Regime Breakdown", ""])
            for regime, metrics in regime_breakdown.items():
                summary_lines.append(
                    f"- {regime}: count={metrics.get('count', 0)} avg_return={metrics.get('avg_total_return', 0.0):.4f} avg_max_drawdown={metrics.get('avg_max_drawdown', 0.0):.4f} avg_sharpe={metrics.get('avg_sharpe', 0.0):.4f}"
                )
        (run_dir / "validation_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    def _write_diagnostics(
        self,
        run_dir: Path,
        dataset: dict[str, pd.DataFrame],
        summary: dict[str, Any],
    ) -> None:
        coverage = build_coverage_report(dataset)
        leakage = build_leakage_report(dataset)

        (run_dir / "feature_coverage.json").write_text(
            json.dumps(coverage, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (run_dir / "leakage_check.json").write_text(
            json.dumps(leakage, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        leakage_lines = [
            "# Leakage Check",
            "",
            f"- Checked tickers: {leakage.get('checked_tickers', 0)}",
            f"- Issues found: {leakage.get('issues_found', 0)}",
            "",
        ]
        for row in leakage.get("rows", []):
            leakage_lines.append(
                f"- {row.get('code')}: trailing_nan_ok={row.get('trailing_nan_ok')} notes={row.get('notes')}"
            )
        (run_dir / "leakage_check.md").write_text("\n".join(leakage_lines), encoding="utf-8")

        coverage_lines = [
            "# Feature Coverage",
            "",
            f"- Ticker count: {coverage.get('ticker_count', 0)}",
            f"- Summary result count: {summary.get('result_count', 0)}",
            "",
            "## Per Ticker",
            "",
        ]
        for row in coverage.get("tickers", []):
            coverage_lines.append(
                f"- {row.get('code')}: rows={row.get('rows')} missing_ratio={row.get('missing_ratio'):.4f} window={row.get('start_date')}~{row.get('end_date')}"
            )
        (run_dir / "feature_coverage.md").write_text("\n".join(coverage_lines), encoding="utf-8")
