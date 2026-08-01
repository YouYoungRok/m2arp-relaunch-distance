"""
lstm_torch.py
-------------
lstm_numpy.py와 동일한 역할(임베딩 -> LSTM -> 선형 회귀 head로 재실행 거리를
예측)을 PyTorch로 다시 구현한 "선택적" 버전입니다.

⚠️ 매우 중요 — 반드시 읽어주세요
--------------------------------
이 파일은 Claude가 실행해서 검증하지 **못한** 상태로 작성되었습니다.
(코드를 작성한 샌드박스 환경 자체에 PyTorch가 설치되어 있지 않고, 그 환경은
네트워크가 막혀 있어 설치도 불가능했습니다.) API 사용법 자체는 표준적이고
안정적인 PyTorch 문법(nn.Embedding, nn.LSTM, nn.Linear, Adam)만 사용했지만,
실제로 이 컴퓨터에서 처음 실행했을 때 사소한 오류가 날 가능성이 있습니다.
그런 경우 에러 메시지를 그대로 붙여넣어 주시면 바로 고쳐드릴 수 있습니다.

반대로 `lstm_numpy.py`는 (1) 수치미분 대비 gradient check, (2) 전체
파이프라인 실행까지 이 환경에서 직접 확인을 마친 **검증된 기본 경로**입니다.
이 파일은 "원 논문이 실제로 사용한 것과 더 가까운 스택(자동미분 기반
LSTM)"을 보여주고 싶을 때 쓰는 선택적 대안입니다.

인터페이스(predict, save, load, train_step)는 NumpyLSTMRegressor와 최대한
동일하게 맞춰서, policies.py / trace_generator.py / relaunch.py / dataset.py는
전혀 수정하지 않고 그대로 재사용할 수 있게 했습니다.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "lstm_torch.py를 쓰려면 먼저 PyTorch를 설치하세요: pip install torch\n"
        "(https://pytorch.org/get-started/locally/ 에서 본인 환경에 맞는 설치 명령을 확인하세요.)"
    ) from e


class TorchLSTMRegressor(nn.Module):
    """NumpyLSTMRegressor와 동일한 목적의 PyTorch 버전."""

    def __init__(self, n_apps: int, embed_dim: int = 16, hidden_dim: int = 32, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.n_apps = n_apps
        self.ed = embed_dim
        self.hd = hidden_dim

        self.embedding = nn.Embedding(n_apps, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, X):
        """X: (B, T) app id 배열 (numpy 또는 torch LongTensor)."""
        if not torch.is_tensor(X):
            X = torch.as_tensor(X, dtype=torch.long)
        X = X.to(self.device)
        emb = self.embedding(X)                 # (B, T, embed_dim)
        _, (h_n, _) = self.lstm(emb)             # h_n: (num_layers, B, hidden_dim)
        h_last = h_n[-1]                          # (B, hidden_dim) — 마지막 층의 마지막 hidden state
        y_hat = self.head(h_last).squeeze(-1)    # (B,)
        return y_hat

    @staticmethod
    def smse_loss(y_hat, target, mask):
        """Selective MSE (원 논문 Eq. 4) — censored(mask=0) 샘플은 손실에서 제외."""
        device = y_hat.device if torch.is_tensor(y_hat) else "cpu"
        if not torch.is_tensor(target):
            target = torch.as_tensor(target, dtype=torch.float32, device=device)
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask, dtype=torch.float32, device=device)
        n_valid = torch.clamp(mask.sum(), min=1.0)
        diff = (y_hat - target) * mask
        return (diff ** 2).sum() / n_valid

    def train_step(self, X, target, mask, lr: float = None) -> float:
        if lr is not None:
            for g in self.optimizer.param_groups:
                g["lr"] = lr
        self.train()
        self.optimizer.zero_grad()
        y_hat = self.forward(X)
        loss = self.smse_loss(y_hat, target, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        self.optimizer.step()
        return float(loss.item())

    def predict(self, X):
        """policies.py의 m2arp_distance_fn과 동일하게 predict(x)[0] 형태로 쓸 수 있도록
        numpy 배열을 반환한다 (NumpyLSTMRegressor.predict와 동일한 반환 형식)."""
        self.eval()
        with torch.no_grad():
            y_hat = self.forward(X).clamp(min=0.0)  # 재실행 거리는 음수가 될 수 없음
        return y_hat.cpu().numpy()

    def save(self, path):
        torch.save({"state_dict": self.state_dict(),
                     "n_apps": self.n_apps, "ed": self.ed, "hd": self.hd}, path)

    @classmethod
    def load(cls, path):
        try:
            ckpt = torch.load(path, map_location="cpu")
        except Exception:
            # PyTorch 2.6+ 는 torch.load의 weights_only 기본값이 True로 바뀌어
            # 일반 dict 체크포인트 로드가 실패할 수 있다. 우리가 직접 저장한
            # 신뢰할 수 있는 파일이므로 weights_only=False로 재시도한다.
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(ckpt["n_apps"], ckpt["ed"], ckpt["hd"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model
