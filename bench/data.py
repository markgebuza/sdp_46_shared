"""Load the GIFT-Eval style Arrow files and build rolling test windows."""

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.ipc as ipc
from gluonts.time_feature import get_seasonality

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRED_LEN = {"M": 12, "W": 8, "D": 30, "H": 48}
TEST_SPLIT = 0.1
MAX_WINDOWS = 20
MAX_TASKS = 600
VAL_WINDOW = -1


@dataclass(frozen=True)
class Task:
    series: int
    window: int  # VAL_WINDOW for the calibration window that precedes the test region
    start: int  # index of the first forecast step


@dataclass
class Dataset:
    name: str
    freq: str
    season: int
    horizon: int
    windows: int
    series: list[np.ndarray]
    item_ids: list[str]
    starts: list[pd.Timestamp]
    tasks: list[Task] = field(default_factory=list)
    val_tasks: list[Task] = field(default_factory=list)

    def history(self, task: Task) -> np.ndarray:
        return self.series[task.series][: task.start]

    def target(self, task: Task) -> np.ndarray:
        return self.series[task.series][task.start : task.start + self.horizon]

    def season_for(self, task: Task) -> int:
        return self.season if task.start > self.season else 1


def dataset_names() -> list[str]:
    return sorted(p.stem for p in DATA_DIR.glob("*.arrow"))


def load_dataset(name: str, max_tasks: int = MAX_TASKS, seed: int = 0) -> Dataset:
    table = ipc.open_stream(DATA_DIR / f"{name}.arrow").read_all()
    freq = table.column("freq")[0].as_py()
    horizon = PRED_LEN[freq.split("-")[0][0]]

    series, item_ids, starts = [], [], []
    for row in table.select(["item_id", "start", "target"]).to_pylist():
        target = row["target"]
        variates = target if isinstance(target[0], list) else [target]
        for v, values in enumerate(variates):
            series.append(np.asarray(values, dtype=np.float64))
            item_ids.append(f"{row['item_id']}" + (f"_{v}" if len(variates) > 1 else ""))
            starts.append(pd.Timestamp(row["start"]))

    min_len = min(len(s) for s in series)
    windows = 1 if name.startswith("m4") else min(max(1, math.ceil(TEST_SPLIT * min_len / horizon)), MAX_WINDOWS)

    ds = Dataset(name, freq, get_seasonality(freq), horizon, windows, series, item_ids, starts)
    all_tasks = [
        Task(i, w, len(s) - (windows - w) * horizon) for i, s in enumerate(series) for w in range(windows)
    ]
    if len(all_tasks) > max_tasks:
        keep = np.sort(np.random.default_rng(seed).choice(len(all_tasks), max_tasks, replace=False))
        all_tasks = [all_tasks[k] for k in keep]
    ds.tasks = all_tasks

    for i in sorted({t.series for t in all_tasks}):
        start = len(series[i]) - (windows + 1) * horizon
        if start >= horizon and np.isfinite(series[i][:start]).any():
            ds.val_tasks.append(Task(i, VAL_WINDOW, start))
    return ds
