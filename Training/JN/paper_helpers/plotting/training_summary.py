# Pedagogical Header
# Module: paper_plots/val_loss_and_timings.py
# Purpose: Summarize validation loss and runtime statistics.
# Used by: experim.ipynb (JN) and sibling helper modules.
# Behavior: intended to match original JN implementation.

"""
training_summary_plotter.py

Utilities for summarizing and plotting loss and runtime statistics across
multiple training configurations.

This module is designed for experiments where each configuration contains
multiple repeated training runs (for example, different random seeds). It
provides tools to:

1. Aggregate validation-loss, training-loss, and timing statistics across runs
2. Compute running-min envelopes for loss trajectories
3. Identify the best run and best epoch per configuration
4. Visualize final loss and total runtime together in a dual-axis bar plot

Main public entry points
------------------------
- summarize_training_runs:
    Collects per-configuration summary statistics from nested model-wrapper
    dictionaries.

- plot_loss_and_runtime_summary:
    Builds a dual-axis bar plot where the left axis shows loss and the right
    axis shows total runtime.

Expected input structure
------------------------
The main input is typically a nested dictionary of the form:

    {
        config_key_1: {seed_1: modelWrapper, seed_2: modelWrapper, ...},
        config_key_2: {seed_1: modelWrapper, seed_2: modelWrapper, ...},
        ...
    }

Each modelWrapper is expected to provide:
- val_loss_list
- train_loss_list
- epoch_times
- best_val_loss
- last_improved   (optional)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch



class TrainingSummaryPlotter:
    """Summarize repeated training runs and plot loss/runtime comparisons."""

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_plot_params(plot_params: Optional[Dict[Any, Sequence[Any]]]) -> Dict[Any, Sequence[Any]]:
        return plot_params or {}

    @staticmethod
    def _style_for_key(
        key: Any,
        index: int,
        plot_params: Dict[Any, Sequence[Any]],
    ) -> Tuple[Any, Any, Any, str]:
        default_style = [f"C{index}", "-", ".", f"Run {index + 1}"]
        color, linestyle, markerstyle, label = plot_params.get(key, default_style)
        return color, linestyle, markerstyle, label

    @staticmethod
    def _collect_validation_losses(model_wrappers: Sequence[Any]) -> List[np.ndarray]:
        losses_raw = []
        for m in model_wrappers:
            if len(m.val_loss_list):
                losses = np.array(m.val_loss_list, dtype=np.float64)
            else:
                losses = np.array(m.train_loss_list, dtype=np.float64)
            losses_raw.append(losses)
        return losses_raw

    @staticmethod
    def _collect_training_losses(model_wrappers: Sequence[Any]) -> List[np.ndarray]:
        losses_raw = []
        for m in model_wrappers:
            losses_raw.append(np.array(m.train_loss_list, dtype=np.float64))
        return losses_raw

    @staticmethod
    def _collect_epoch_times(
        model_wrappers: Sequence[Any],
        percentile: float = 80,
    ) -> List[np.ndarray]:
        epoch_times_arr = []
        for m in model_wrappers:
            arr = np.array(m.epoch_times, dtype=np.float64)
            if len(arr) == 0:
                epoch_times_arr.append(arr)
                continue
            p = np.percentile(arr, percentile)
            epoch_times_arr.append(arr[arr <= p])
        return epoch_times_arr

    @staticmethod
    def _pad_and_stack(losses_raw: Sequence[np.ndarray]) -> Tuple[np.ndarray, int, List[int]]:
        lengths = [len(v) for v in losses_raw]
        max_len = max(lengths) if lengths else 0

        if max_len == 0:
            return np.full((len(losses_raw), 0), np.nan, dtype=np.float64), 0, lengths

        losses = np.full((len(losses_raw), max_len), np.nan, dtype=np.float64)
        for i, (v, L) in enumerate(zip(losses_raw, lengths)):
            losses[i, :L] = v

        return losses, max_len, lengths

    @staticmethod
    def _running_min_envelope(losses: np.ndarray, lengths: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        if losses.size == 0:
            return np.array([]), np.array([])

        running_mins = np.full_like(losses, np.nan)
        for j in range(losses.shape[0]):
            L = lengths[j]
            if L == 0:
                continue
            running_mins[j, :L] = np.minimum.accumulate(losses[j, :L])

        return np.nanmin(running_mins, axis=0), np.nanmax(running_mins, axis=0)

    @staticmethod
    def _final_value_stats(losses: np.ndarray) -> Tuple[float, float, np.ndarray]:
        if losses.size == 0:
            return np.nan, np.nan, np.array([])

        last_valid_vals = np.array([
            row[~np.isnan(row)][-1] if np.any(~np.isnan(row)) else np.nan
            for row in losses
        ])
        return (
            float(np.nanmean(last_valid_vals)),
            float(np.nanstd(last_valid_vals)),
            last_valid_vals,
        )

    @staticmethod
    def _best_validation_epochs_and_levels(
        model_wrappers: Sequence[Any],
        validation_losses_raw: Sequence[np.ndarray],
    ) -> Tuple[List[int], List[float], float]:
        best_epochs = []
        best_levels = []

        for idx, m in enumerate(model_wrappers):
            if len(validation_losses_raw[idx]) == 0:
                best_epoch = 0
            else:
                best_epoch = int(np.argmin(np.abs(validation_losses_raw[idx] - m.best_val_loss)))
            best_epochs.append(best_epoch)
            best_levels.append(m.best_val_loss)

        mean_level = float(np.mean(best_levels)) if best_levels else np.nan
        return best_epochs, best_levels, mean_level

    @staticmethod
    def _best_training_epochs_and_levels(
        training_losses_raw: Sequence[np.ndarray],
    ) -> Tuple[List[int], List[float], float]:
        best_epochs = []
        best_levels = []

        for losses in training_losses_raw:
            if len(losses) == 0:
                best_epochs.append(0)
                best_levels.append(np.nan)
            else:
                best_epoch = int(np.argmin(losses))
                best_epochs.append(best_epoch)
                best_levels.append(float(losses[best_epoch]))

        mean_level = float(np.nanmean(best_levels)) if best_levels else np.nan
        return best_epochs, best_levels, mean_level

    @staticmethod
    def _select_best_validation_run(model_wrappers: Sequence[Any]) -> Tuple[Any, int, List[float]]:
        best_model_loss = [m.best_val_loss for m in model_wrappers]
        best_seed_idx = int(np.nanargmin(best_model_loss))
        return model_wrappers[best_seed_idx], best_seed_idx, best_model_loss

    @staticmethod
    def _select_best_training_run(
        model_wrappers: Sequence[Any],
        training_losses_raw: Sequence[np.ndarray],
    ) -> Tuple[Any, int, List[float]]:
        best_model_loss = [
            float(np.min(losses)) if len(losses) else np.nan
            for losses in training_losses_raw
        ]
        best_seed_idx = int(np.nanargmin(best_model_loss))
        return model_wrappers[best_seed_idx], best_seed_idx, best_model_loss

    @staticmethod
    def _resolve_best_epoch_index(
        best_model: Any,
        best_seed_idx: int,
        best_epochs: Sequence[int],
        max_len: int,
        attr_name: Optional[str] = "last_improved",
    ) -> int:
        best_epoch_idx = None
        if attr_name is not None:
            best_epoch_idx = getattr(best_model, attr_name, None)

        if best_epoch_idx is None:
            best_epoch_idx = best_epochs[best_seed_idx]

        best_epoch_idx = int(best_epoch_idx)

        if max_len > 0 and best_epoch_idx >= max_len and (best_epoch_idx - 1) < max_len:
            best_epoch_idx -= 1

        if max_len > 0:
            best_epoch_idx = max(0, min(best_epoch_idx, max_len - 1))
        else:
            best_epoch_idx = 0

        return best_epoch_idx

    @staticmethod
    def _timing_stats(
        epoch_times_arr: Sequence[np.ndarray],
        model_wrappers: Sequence[Any],
        sampling_factor: int = 1,
    ) -> Tuple[float, float, float, float]:
        if not epoch_times_arr:
            return np.nan, np.nan, np.nan, np.nan

        mean_epoch_times = [np.mean(times) for times in epoch_times_arr if len(times)]
        if not mean_epoch_times:
            return np.nan, np.nan, np.nan, np.nan

        avg_epoch_time = float(np.mean(mean_epoch_times))
        std_epoch_time = float(np.std(mean_epoch_times))

        total_num_epochs = [len(m.epoch_times) for m in model_wrappers]
        total_time_arr = [
            np.mean(times) * epoch_num * sampling_factor if len(times) else np.nan
            for times, epoch_num in zip(epoch_times_arr, total_num_epochs)
        ]

        run_time_mean = float(np.nanmean(total_time_arr))
        run_time_std = float(np.nanstd(total_time_arr))

        return avg_epoch_time, std_epoch_time, run_time_mean, run_time_std

    @staticmethod
    def _label_with_stats(label: str, mean_final: float, std_final: float) -> str:
        mean_str = f"{mean_final:.2e}" if not np.isnan(mean_final) else "nan"
        std_str = f"{std_final:.1e}" if not np.isnan(std_final) else "nan"
        return f"{label} ({mean_str} ± {std_str})"

    # ------------------------------------------------------------------
    # Public summary method
    # ------------------------------------------------------------------
    def summarize_training_runs(
        self,
        models_dics: Dict[Any, Dict[Any, Any]],
        plot_params: Optional[Dict[Any, Sequence[Any]]] = None,
        stop_validation_at_best_epoch: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Aggregate loss and timing statistics for each model configuration.

        Parameters
        ----------
        models_dics
            Nested dictionary of model wrappers:
            {config_key: {seed: wrapper, ...}, ...}
        plot_params
            Optional style overrides:
            {config_key: [color, linestyle, markerstyle, label]}
        stop_validation_at_best_epoch
            If True, truncate validation running-min curves at the selected best epoch.

        Returns
        -------
        (run_data, raw_timing_data)
            run_data contains per-configuration loss summaries.
            raw_timing_data contains per-configuration timing summaries.
        """
        plot_params = self._ensure_plot_params(plot_params)

        run_data = []
        raw_timing_data = []

        model_keys = list(models_dics.keys())
        dic_values = list(models_dics.values())

        for i, (key, model_dic) in enumerate(zip(model_keys, dic_values)):
            model_wrappers = list(model_dic.values())
            if len(model_wrappers) == 0:
                continue

            sampling_factor = 1
            color, linestyle, markerstyle, label = self._style_for_key(key, i, plot_params)

            # Validation
            validation_losses_raw = self._collect_validation_losses(model_wrappers)
            epoch_times_arr = self._collect_epoch_times(model_wrappers, percentile=80)

            validation_losses, max_len_val, lengths_val = self._pad_and_stack(validation_losses_raw)
            running_min_min, running_min_max = self._running_min_envelope(validation_losses, lengths_val)
            epochs = np.arange(max_len_val) if max_len_val > 0 else np.array([])

            mean_final, std_final, _ = self._final_value_stats(validation_losses)

            best_epochs_lst, hticks_vals, hticks = self._best_validation_epochs_and_levels(
                model_wrappers, validation_losses_raw
            )

            best_model, best_seed_idx, best_model_loss = self._select_best_validation_run(model_wrappers)
            best_epoch_idx = self._resolve_best_epoch_index(
                best_model=best_model,
                best_seed_idx=best_seed_idx,
                best_epochs=best_epochs_lst,
                max_len=max_len_val,
                attr_name="last_improved",
            )

            # Training
            training_losses_raw = self._collect_training_losses(model_wrappers)
            training_losses, max_len_train, lengths_train = self._pad_and_stack(training_losses_raw)
            train_running_min_min, train_running_min_max = self._running_min_envelope(
                training_losses, lengths_train
            )
            train_epochs = np.arange(max_len_train) if max_len_train > 0 else np.array([])

            train_mean_final, train_std_final, _ = self._final_value_stats(training_losses)

            train_best_epochs_lst, train_hticks_vals, train_hticks = self._best_training_epochs_and_levels(
                training_losses_raw
            )

            best_model_train, best_seed_idx_train, best_model_loss_train = self._select_best_training_run(
                model_wrappers, training_losses_raw
            )

            train_best_epoch_idx = self._resolve_best_epoch_index(
                best_model=best_model_train,
                best_seed_idx=best_seed_idx_train,
                best_epochs=train_best_epochs_lst,
                max_len=max_len_train,
                attr_name=None,
            )

            # Timing
            avg_epoch_time, std_epoch_time, run_time_mean, run_time_std = self._timing_stats(
                epoch_times_arr, model_wrappers, sampling_factor=sampling_factor
            )

            label_with_stats = self._label_with_stats(label, mean_final, std_final)

            if stop_validation_at_best_epoch and len(epochs) > 0:
                running_min_min = running_min_min[: best_epoch_idx + 1]
                running_min_max = running_min_max[: best_epoch_idx + 1]
                epochs = epochs[: best_epoch_idx + 1]

            run_data.append(
                {
                    "label": label,
                    "mean_best_model_loss": float(np.mean(best_model_loss)),
                    "std_best_model_loss": float(np.std(best_model_loss)),
                    "min_best_model_loss": float(np.min(best_model_loss)),
                    "max_best_model_loss": float(np.max(best_model_loss)),
                    "mean_final": float(mean_final),
                    "hticks": float(hticks),
                    "best_epochs_lst": best_epochs_lst,
                    "epochs": epochs,
                    "running_min_min": running_min_min,
                    "running_min_max": running_min_max,
                    "best_epoch_seed_idx": [best_epoch_idx, best_seed_idx],
                    "label_with_stats": label_with_stats,
                    "color": color,
                    "linestyle": linestyle,
                    "markerstyle": markerstyle,
                    "train_mean_best_model_loss": float(np.nanmean(best_model_loss_train)),
                    "train_std_best_model_loss": float(np.nanstd(best_model_loss_train)),
                    "train_min_best_model_loss": float(np.nanmin(best_model_loss_train)),
                    "train_max_best_model_loss": float(np.nanmax(best_model_loss_train)),
                    "train_mean_final": float(train_mean_final),
                    "train_hticks": float(train_hticks),
                    "train_best_epochs_lst": train_best_epochs_lst,
                    "train_epochs": train_epochs,
                    "train_running_min_min": train_running_min_min,
                    "train_running_min_max": train_running_min_max,
                    "train_best_epoch_seed_idx": [train_best_epoch_idx, best_seed_idx_train],
                }
            )

            raw_timing_data.append(
                {
                    "label": label,
                    "avg_epoch_time": float(avg_epoch_time),
                    "std_epoch_time": float(std_epoch_time),
                    "run_time_mean": float(run_time_mean),
                    "run_time_std": float(run_time_std),
                    "color": color,
                }
            )

        return run_data, raw_timing_data

    # ------------------------------------------------------------------
    # Public plotting method
    # ------------------------------------------------------------------
    def plot_loss_and_runtime_summary(
        self,
        models_dics_list: Sequence[Dict[Any, Dict[Any, Any]]],
        label_list: Optional[Sequence[str]] = None,
        plot_params: Optional[Dict[Any, Sequence[Any]]] = None,
        name: Optional[str] = None,
        figsize: Tuple[float, float] = (10, 5),
        legend_title: str = "ES",
        y_label: str = "Validation loss [a.u.]",
        y2_label: str = "Total Run Time [s]",
        hatch_patterns: Sequence[str] = ("", "////", "\\\\\\\\", "xxxx", "++++", "....", "ooo", "***"),
        colors: Optional[Sequence[str]] = None,
        hatch_linewidth: float = 1,
        x_label: str = r"$N_u$",
        x_label_fontsize: int = 14,
        legend_size: int = 16,
        legend_bool: bool = False,
        intra_spacing: float = 0.15,
        inter_spacing: float = 0.80,
        alpha: float = 0.3,
        y2lim: Optional[Tuple[float, float]] = None,
        err_ecolor: str = "black",
        err_elinewidth: float = 1.5,
        err_capsize: float = 5,
        err_capthick: float = 1.5,
        print_summary: bool = True,
        print_precision: int = 6,
    ) -> None:
        """
        Create a dual-axis bar plot summarizing validation loss and runtime.

        Left y-axis:
            validation loss
        Right y-axis:
            total runtime

        Parameters
        ----------
        models_dics_list
            List of nested configuration dictionaries.
        label_list
            Labels for the outer groups.
        plot_params
            Optional plotting-style overrides passed to summarize_training_runs.
        """
        if colors is None:
            colors = ["#0033A0", "#1E90FF", "#6699CC", "#A4C8E1", "#D6EAF8"]

        if label_list is None:
            label_list = [f"Group {i + 1}" for i in range(len(models_dics_list))]

        grouped_loss = []
        grouped_time = []
        all_labels = []

        for models_dics in models_dics_list:
            run_data, raw_data = self.summarize_training_runs(
                models_dics,
                plot_params=plot_params,
            )

            loss_map = {d["label"]: d for d in run_data}
            time_map = {d["label"]: d for d in raw_data}

            grouped_loss.append(loss_map)
            grouped_time.append(time_map)

            for lbl in loss_map.keys():
                if lbl not in all_labels:
                    all_labels.append(lbl)

        num_groups = len(models_dics_list)
        num_labels = len(all_labels)
        label_color_map = {lbl: colors[i % len(colors)] for i, lbl in enumerate(all_labels)}

        if print_summary:
            fmt = f".{print_precision}g"
            print("\n" + "=" * 100)
            print("Validation loss and total learning time summary")
            print("=" * 100)

            for gi in range(num_groups):
                group_name = label_list[gi] if gi < len(label_list) else f"Group {gi + 1}"
                print(f"\n--- {group_name} ---")

                for label in all_labels:
                    if label not in grouped_loss[gi] or label not in grouped_time[gi]:
                        print(f"{label:>15s} | missing data")
                        continue

                    loss_d = grouped_loss[gi][label]
                    time_d = grouped_time[gi][label]

                    m_loss = loss_d["mean_best_model_loss"]
                    min_l = loss_d.get("min_best_model_loss", m_loss - loss_d["std_best_model_loss"])
                    max_l = loss_d.get("max_best_model_loss", m_loss + loss_d["std_best_model_loss"])

                    m_time = time_d["run_time_mean"]
                    min_t = time_d.get("run_time_min", m_time - time_d["run_time_std"])
                    max_t = time_d.get("run_time_max", m_time + time_d["run_time_std"])

                    print(
                        f"{label:>15s} | "
                        f"val loss: mean={m_loss:{fmt}}, range=[{min_l:{fmt}}, {max_l:{fmt}}] | "
                        f"time: mean={m_time:{fmt}} s, range=[{min_t:{fmt}}, {max_t:{fmt}}] s"
                    )

            print("=" * 100 + "\n")

        x_positions = np.arange(num_labels) * inter_spacing
        slot_width = 0.35
        bar_width = slot_width * 0.40
        loss_offset = -intra_spacing / 2
        time_offset = +intra_spacing / 2

        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()

        old_hlw = mpl.rcParams["hatch.linewidth"]
        old_hc = mpl.rcParams["hatch.color"]
        mpl.rcParams["hatch.linewidth"] = hatch_linewidth
        mpl.rcParams["hatch.color"] = "gray"

        try:
            for li, label in enumerate(all_labels):
                base_x = x_positions[li]
                hatch = hatch_patterns[li % len(hatch_patterns)]

                for gi in range(num_groups):
                    if label not in grouped_loss[gi] or label not in grouped_time[gi]:
                        continue

                    loss_d = grouped_loss[gi][label]
                    time_d = grouped_time[gi][label]

                    m_loss = loss_d["mean_best_model_loss"]
                    min_l = loss_d.get("min_best_model_loss", m_loss - loss_d["std_best_model_loss"])
                    max_l = loss_d.get("max_best_model_loss", m_loss + loss_d["std_best_model_loss"])
                    yerr_loss = np.array([[m_loss - min_l], [max_l - m_loss]])

                    m_time = time_d["run_time_mean"]
                    min_t = time_d.get("run_time_min", m_time - time_d["run_time_std"])
                    max_t = time_d.get("run_time_max", m_time + time_d["run_time_std"])
                    yerr_time = np.array([[m_time - min_t], [max_t - m_time]])

                    color = label_color_map[label]
                    slot_x = base_x + gi * slot_width

                    loss_error_kw = dict(
                        ecolor=err_ecolor,
                        elinewidth=err_elinewidth,
                        capsize=err_capsize,
                        capthick=err_capthick,
                        alpha=0.75,
                    )
                    time_error_kw = dict(
                        ecolor=err_ecolor,
                        elinewidth=err_elinewidth,
                        capsize=err_capsize,
                        capthick=err_capthick,
                        alpha=alpha,
                    )

                    ax1.bar(
                        slot_x + loss_offset,
                        m_loss,
                        width=bar_width,
                        yerr=yerr_loss,
                        error_kw=loss_error_kw,
                        facecolor=color,
                        alpha=0.75,
                        edgecolor="black",
                        hatch=hatch,
                        linewidth=1.5,
                    )

                    ax2.bar(
                        slot_x + time_offset,
                        m_time,
                        width=bar_width,
                        yerr=yerr_time,
                        error_kw=time_error_kw,
                        facecolor=color,
                        alpha=alpha,
                        edgecolor="black",
                        hatch=hatch,
                        linewidth=1.5,
                    )

            handles = [
                Patch(
                    facecolor="white",
                    edgecolor="black",
                    hatch=hatch_patterns[i % len(hatch_patterns)],
                    label=label_list[i],
                    linewidth=1.5,
                )
                for i in range(num_groups)
            ]
            if legend_bool:
                ax1.legend(handles, label_list, title=legend_title, fontsize=legend_size)

            ax1.set_ylabel(y_label, fontsize=14)
            ax2.set_ylabel(y2_label, fontsize=14, color="gray")
            ax1.set_yscale("log")

            ax1.set_xticks(x_positions)
            ax1.set_xticklabels(all_labels, fontsize=14)

            if y2lim is not None:
                ax2.set_ylim(y2lim)

            ax2.tick_params(axis="y", colors="gray")
            ax1.set_xlabel(x_label, fontsize=x_label_fontsize)

            fig.tight_layout()

            if name:
                plt.savefig(name, dpi=120, bbox_inches="tight")
                print("Saved plot:", name)

            plt.show()

        finally:
            mpl.rcParams["hatch.linewidth"] = old_hlw
            mpl.rcParams["hatch.color"] = old_hc