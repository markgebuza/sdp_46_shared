"""Uniform wrappers: predict(contexts, horizon) -> float array (n, horizon, 9) of quantiles 0.1..0.9.

"Raw" means each library's own inference path with its optional extras switched off; NaNs in the
context are passed through and handled by whatever the model does natively.
"""

import numpy as np
import torch

BATCH = 64


def _batches(order: np.ndarray):
    for i in range(0, len(order), BATCH):
        yield order[i : i + BATCH]


class Chronos2:
    name = "Chronos-2"
    checkpoint = "amazon/chronos-2"

    def __init__(self):
        from chronos import Chronos2Pipeline

        self.pipe = Chronos2Pipeline.from_pretrained(self.checkpoint, device_map="cpu")

    def predict(self, contexts: list[np.ndarray], horizon: int) -> np.ndarray:
        inputs = [torch.tensor(c, dtype=torch.float32) for c in contexts]
        q, _ = self.pipe.predict_quantiles(inputs, prediction_length=horizon, batch_size=BATCH)
        return np.stack([x[0].numpy() for x in q])


class TimesFM3:
    name = "TimesFM-3.0"
    checkpoint = "google/timesfm-3.0-pytorch"

    def __init__(self):
        from timesfm3 import ModelConfig, TimesFM3Forecaster

        self.model = TimesFM3Forecaster(
            ModelConfig(checkpoint_path=self.checkpoint, per_core_batch_size=BATCH, device="cpu")
        )

    def predict(self, contexts: list[np.ndarray], horizon: int) -> np.ndarray:
        out = np.empty((len(contexts), horizon, 9))
        order = np.argsort([len(c) for c in contexts], kind="stable")
        for idx in _batches(order):
            res = self.model.predict_batch(
                [contexts[i].astype(np.float32) for i in idx],
                horizon=horizon,
                return_quantiles=True,
                use_symmetric_averaging=False,
                make_positive=False,
                sort_quantiles=False,
            )
            for i, r in zip(idx, res):
                out[i] = r.quantiles
        return out


class Toto2:
    name = "Toto-2.0"
    checkpoint = "Datadog/Toto-2.0-313m"

    def __init__(self):
        from toto2 import Toto2Model
        from toto2.model import ffill_imputation

        self.model = Toto2Model.from_pretrained(self.checkpoint).eval()
        self.patch = self.model.config.patch_size
        self.ffill = ffill_imputation

    def predict(self, contexts: list[np.ndarray], horizon: int) -> np.ndarray:
        out = np.empty((len(contexts), horizon, 9))
        order = np.argsort([len(c) for c in contexts], kind="stable")
        for idx in _batches(order):
            length = -(-max(len(contexts[i]) for i in idx) // self.patch) * self.patch
            target = np.zeros((len(idx), 1, length), dtype=np.float32)
            mask = np.zeros_like(target, dtype=bool)
            for row, i in enumerate(idx):
                c = contexts[i].astype(np.float32)
                observed = np.isfinite(c)
                # Same defaults as Toto's own GluonTS predictor: ffill values, mask them as unobserved.
                target[row, 0, length - len(c) :] = self.ffill(c)
                mask[row, 0, length - len(c) :] = observed
            with torch.inference_mode():
                q = self.model.forecast(
                    {
                        "target": torch.from_numpy(target),
                        "target_mask": torch.from_numpy(mask),
                        "series_ids": torch.zeros(len(idx), 1, dtype=torch.long),
                    },
                    horizon=horizon,
                    decode_block_size=768,
                    has_missing_values=True,
                    scaler_fallback_min_obs=8,
                )
            out[idx] = q[:, :, 0, :horizon].permute(1, 2, 0).numpy()
        return out


MODELS = {m.name: m for m in (Chronos2, TimesFM3, Toto2)}
