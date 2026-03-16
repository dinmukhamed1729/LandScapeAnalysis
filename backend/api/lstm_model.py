import threading
from pathlib import Path

import torch
import torch.nn as nn

MODEL_PATH = Path(__file__).resolve().parent.parent / 'lstm_model.pt'

_model = None
_model_lock = threading.Lock()
_model_meta: dict = {}


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: [B, T, F]
        out, _ = self.lstm(x)   # [B, T, H]
        last = out[:, -1, :]    # [B, H]
        y = self.head(last)     # [B, 1]
        return y.squeeze(-1)    # [B]


def _infer_params(state_dict: dict) -> dict:
    """Infer model hyper-parameters from saved state dict shapes."""
    # weight_ih_l0: (4 * hidden_size, input_size)
    wih = state_dict['lstm.weight_ih_l0']
    hidden_size = wih.shape[0] // 4
    input_size = wih.shape[1]

    # count LSTM layers
    num_layers = sum(1 for k in state_dict if k.startswith('lstm.weight_ih_l'))

    return {'input_size': input_size, 'hidden_size': hidden_size, 'num_layers': num_layers}


def get_model() -> tuple['LSTMRegressor', dict]:
    """Return cached model instance and its meta-parameters."""
    global _model, _model_meta
    if _model is not None:
        return _model, _model_meta

    with _model_lock:
        if _model is not None:
            return _model, _model_meta

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f'Model file not found: {MODEL_PATH}')

        state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
        params = _infer_params(state_dict)
        model = LSTMRegressor(**params)
        model.load_state_dict(state_dict)
        model.eval()

        _model = model
        _model_meta = params

    return _model, _model_meta
