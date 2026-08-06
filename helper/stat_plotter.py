import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

DPI = 300

# Genuinely time-varying, per-frame signals -> line graph vs frame number.
TIME_SERIES_STATS = {
    'fps': 'FPS',
    'latency_ms': 'Latency (ms)',
}

# Running/summary stats that converge to one meaningful number per camera per run
# (the same four the perf CSV's own "# Summary:" trailer line reports) -> single bar graph.
SUMMARY_STATS = {
    'frames': 'Total Frames',
    'avg_fps': 'Avg FPS',
    'avg_latency_ms': 'Avg Latency (ms)',
    'peak_latency_ms': 'Peak Latency (ms)',
}

CAMERA_COLORS = {
    'cam1': 'tab:blue',
    'cam2': 'tab:orange',
}


def load_perf_csv(filepath):
    """Loads a --log perf CSV (camera_id,frame_number,fps,latency_ms,avg_fps,
    avg_latency_ms,peak_latency_ms), ignoring the '# Summary: ...' trailer line.
    Returns None (with a warning, not a crash) if the file is missing or empty."""
    if not os.path.exists(filepath):
        print(f"Warning: '{filepath}' not found. Run the pipeline with --log first to generate perf data.")
        return None

    df = pd.read_csv(filepath, comment='#')
    if df.empty:
        print(f"Warning: '{filepath}' has no data rows yet.")
        return None

    return df


def camera_label(df):
    """The camera_id column is constant per file -- pull it out once for titles/legends."""
    return str(df['camera_id'].iloc[0])


def plot_time_series(datasets, column, ylabel, title, out_path):
    """One line per camera, stat vs frame_number, saved as a high-resolution PNG."""
    fig, ax = plt.subplots(figsize=(14, 7), dpi=DPI)

    for label, df in datasets:
        color = CAMERA_COLORS.get(label, None)
        ax.plot(df['frame_number'], df[column], label=label, color=color, linewidth=1.2)

    ax.set_xlabel('Frame Number')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    print(f"Saved {out_path}")
    return fig


def plot_summary_bar(datasets, out_path):
    """All summary stats for both cameras in one grouped bar chart. The stats span
    wildly different scales (fps ~0-15 vs frame counts in the thousands), so this
    uses a log-scale y-axis with the exact value labeled on top of each bar --
    without that, bars for the small-magnitude stats would be visually invisible
    next to the frame-count bars on a linear scale."""
    stat_keys = list(SUMMARY_STATS.keys())
    n_stats = len(stat_keys)
    n_cams = len(datasets)

    fig, ax = plt.subplots(figsize=(12, 7), dpi=DPI)

    group_width = 0.8
    bar_width = group_width / max(n_cams, 1)
    x = np.arange(n_stats)

    for i, (label, df) in enumerate(datasets):
        # Every row shares the same running-summary values by the last row of the run;
        # the final row is exactly the state reported in the CSV's own summary trailer.
        last_row = df.iloc[-1]
        values = [
            df['frame_number'].max() if key == 'frames' else last_row[key]
            for key in stat_keys
        ]

        offset = (i - (n_cams - 1) / 2) * bar_width
        color = CAMERA_COLORS.get(label, None)
        bars = ax.bar(x + offset, values, width=bar_width * 0.9, label=label, color=color)

        for bar, value in zip(bars, values):
            ax.annotate(f'{value:,.2f}' if value < 1000 else f'{value:,.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8, rotation=90)

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([SUMMARY_STATS[k] for k in stat_keys])
    ax.set_ylabel('Value (log scale)')
    ax.set_title('Pipeline Summary Stats by Camera')
    ax.grid(True, axis='y', alpha=0.3, which='both')
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    print(f"Saved {out_path}")
    return fig


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logging_dir = os.path.join(script_dir, '..', 'build', 'logging')
    plots_dir = os.path.join(script_dir, '..', 'build', 'stats')
    os.makedirs(plots_dir, exist_ok=True)

    df1 = load_perf_csv(os.path.join(logging_dir, 'perf_1.csv'))
    df2 = load_perf_csv(os.path.join(logging_dir, 'perf_2.csv'))

    datasets = [(camera_label(df), df) for df in (df1, df2) if df is not None]

    if not datasets:
        print("No perf data found for either camera. Run the pipeline with --log first.")
        return

    for column, ylabel in TIME_SERIES_STATS.items():
        out_path = os.path.join(plots_dir, f'{column}_over_time.png')
        plot_time_series(datasets, column, ylabel, f'{ylabel} Over Time', out_path)

    plot_summary_bar(datasets, os.path.join(plots_dir, 'summary_stats.png'))

    print(f"\nAll plots saved to: {plots_dir}")
    plt.show()


if __name__ == '__main__':
    main()
