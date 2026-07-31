"""
lstm_numpy.py
-------------
딥러닝 프레임워크(PyTorch/TensorFlow) 없이 NumPy만으로 LSTM을 직접 구현한다.
(엣지/온디바이스 환경에서 경량 추론기를 직접 이해·구현하는 것을 목표로 한
포트폴리오 취지에 맞춰, 프레임워크에 의존하지 않고 forward/backward(BPTT)를
직접 유도·구현했다.)

구성 요소
---------
1. Embedding lookup (앱 ID -> 임베딩 벡터)
2. 표준 LSTM Cell (forget/input/output/candidate gate)
3. 마지막 hidden state -> Linear 회귀 head (재실행 거리 예측)
4. SMSE(Selective MSE) 손실: 유효한(재실행 기록이 있는) 샘플에 대해서만
   오차를 계산하고, censored(재실행 기록이 없는) 샘플은 마스킹하여 제외한다.
5. Adam optimizer를 직접 구현.

정확성 검증을 위해 파일 하단에 수치미분(numerical gradient) 대비
gradient check 테스트를 포함한다.
"""
from __future__ import annotations
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class NumpyLSTMRegressor:
    def __init__(self, n_apps: int, embed_dim: int = 16, hidden_dim: int = 32, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_apps = n_apps
        self.ed = embed_dim
        self.hd = hidden_dim

        def glorot(shape):
            limit = np.sqrt(6.0 / (shape[0] + shape[1]))
            return rng.uniform(-limit, limit, size=shape)

        # 임베딩
        self.E = rng.normal(0, 0.05, size=(n_apps, embed_dim))

        # LSTM 결합 가중치: [i, f, o, g] 순서로 4*hd 출력
        z_dim = embed_dim + hidden_dim
        self.Wz = glorot((4 * hidden_dim, z_dim))
        self.bz = np.zeros(4 * hidden_dim)
        # forget gate bias는 1로 초기화 (학습 안정성을 위한 표준 관행)
        self.bz[hidden_dim:2 * hidden_dim] = 1.0

        # 출력 head (마지막 hidden state -> 스칼라 회귀)
        self.Wout = glorot((1, hidden_dim))
        self.bout = np.zeros(1)

        self._init_adam()

    # ---------------- parameter (de)serialization ----------------
    def params(self):
        return {"E": self.E, "Wz": self.Wz, "bz": self.bz, "Wout": self.Wout, "bout": self.bout}

    def _init_adam(self):
        self.m = {k: np.zeros_like(v) for k, v in self.params().items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params().items()}
        self.t = 0

    def save(self, path):
        np.savez(path, E=self.E, Wz=self.Wz, bz=self.bz, Wout=self.Wout, bout=self.bout,
                 n_apps=self.n_apps, ed=self.ed, hd=self.hd)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        m = cls(int(d["n_apps"]), int(d["ed"]), int(d["hd"]))
        m.E, m.Wz, m.bz, m.Wout, m.bout = d["E"], d["Wz"], d["bz"], d["Wout"], d["bout"]
        return m

    # ---------------- forward ----------------
    def forward(self, X):
        """
        X: (B, T) int array of app ids
        returns y_hat (B,) and a cache dict for backward()
        """
        B, T = X.shape
        hd = self.hd
        Emb = self.E[X]  # (B, T, ed)

        h = np.zeros((B, hd))
        c = np.zeros((B, hd))
        cache = {"Emb": Emb, "h_list": [h], "c_list": [c],
                  "i_list": [], "f_list": [], "o_list": [], "g_list": [], "z_list": []}

        for t in range(T):
            z = np.concatenate([Emb[:, t, :], h], axis=1)  # (B, ed+hd)
            a = z @ self.Wz.T + self.bz  # (B, 4hd)
            ai, af, ao, ag = np.split(a, 4, axis=1)
            i = sigmoid(ai); f = sigmoid(af); o = sigmoid(ao); g = np.tanh(ag)
            c = f * c + i * g
            h = o * np.tanh(c)

            cache["z_list"].append(z)
            cache["i_list"].append(i); cache["f_list"].append(f)
            cache["o_list"].append(o); cache["g_list"].append(g)
            cache["h_list"].append(h); cache["c_list"].append(c)

        y_hat = h @ self.Wout.T + self.bout  # (B,1)
        cache["y_hat"] = y_hat
        cache["T"] = T
        cache["B"] = B
        return y_hat[:, 0], cache

    # ---------------- SMSE loss ----------------
    @staticmethod
    def smse_loss(y_hat, target, mask):
        """
        Selective MSE: mask==0 인 샘플(재실행 기록 없음, censored)은 손실에서 제외.
        """
        mask = mask.astype(np.float64)
        n_valid = max(mask.sum(), 1.0)
        diff = (y_hat - target) * mask
        loss = np.sum(diff ** 2) / n_valid
        dy_hat = (2.0 / n_valid) * diff  # dL/dy_hat, invalid 샘플은 자동으로 0
        return loss, dy_hat

    # ---------------- backward (BPTT) ----------------
    def backward(self, X, cache, dy_hat):
        B, T, hd = cache["B"], cache["T"], self.hd
        grads = {k: np.zeros_like(v) for k, v in self.params().items()}

        h_last = cache["h_list"][-1]
        grads["Wout"] += dy_hat[:, None].T @ h_last
        grads["bout"] += dy_hat.sum(axis=0, keepdims=True)[0] if dy_hat.ndim == 1 else dy_hat.sum(axis=0)
        # dy_hat is (B,), fix bout grad shape
        grads["bout"] = np.array([dy_hat.sum()])

        dh_next = dy_hat[:, None] @ self.Wout  # (B, hd), gradient flowing into h_T
        dc_next = np.zeros((B, hd))

        for t in reversed(range(T)):
            h_prev = cache["h_list"][t]      # h_{t-1}
            c_prev = cache["c_list"][t]      # c_{t-1}
            c_t = cache["c_list"][t + 1]
            i_t, f_t, o_t, g_t = (cache["i_list"][t], cache["f_list"][t],
                                    cache["o_list"][t], cache["g_list"][t])
            z_t = cache["z_list"][t]

            tanh_c = np.tanh(c_t)
            dh = dh_next
            do = dh * tanh_c
            dc = dc_next + dh * o_t * (1 - tanh_c ** 2)

            di = dc * g_t
            dg = dc * i_t
            df = dc * c_prev

            do_pre = do * o_t * (1 - o_t)
            di_pre = di * i_t * (1 - i_t)
            df_pre = df * f_t * (1 - f_t)
            dg_pre = dg * (1 - g_t ** 2)

            da = np.concatenate([di_pre, df_pre, do_pre, dg_pre], axis=1)  # (B,4hd)

            grads["Wz"] += da.T @ z_t
            grads["bz"] += da.sum(axis=0)

            dz = da @ self.Wz  # (B, ed+hd)
            dEmb_t = dz[:, :self.ed]
            dh_prev = dz[:, self.ed:]

            # embedding gradient accumulation (scatter-add per app id at time t)
            np.add.at(grads["E"], X[:, t], dEmb_t)

            dh_next = dh_prev
            dc_next = dc * f_t

        return grads

    # ---------------- Adam update ----------------
    def adam_step(self, grads, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, clip=5.0):
        self.t += 1
        for k, p in self.params().items():
            g = grads[k]
            # gradient clipping (norm)
            norm = np.linalg.norm(g)
            if norm > clip:
                g = g * (clip / (norm + 1e-12))
            self.m[k] = beta1 * self.m[k] + (1 - beta1) * g
            self.v[k] = beta2 * self.v[k] + (1 - beta2) * (g ** 2)
            m_hat = self.m[k] / (1 - beta1 ** self.t)
            v_hat = self.v[k] / (1 - beta2 ** self.t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # ---------------- convenience: train / predict ----------------
    def train_step(self, X, target, mask, lr=0.01):
        y_hat, cache = self.forward(X)
        loss, dy_hat = self.smse_loss(y_hat, target, mask)
        grads = self.backward(X, cache, dy_hat)
        self.adam_step(grads, lr=lr)
        return loss

    def predict(self, X):
        y_hat, _ = self.forward(X)
        return np.maximum(y_hat, 0.0)  # 재실행 거리는 음수가 될 수 없음


# ------------------------------------------------------------------
# Gradient check: 수치미분과 비교하여 BPTT 구현의 정확성을 검증
# ------------------------------------------------------------------
def _gradient_check():
    rng = np.random.default_rng(0)
    n_apps, ed, hd, T, B = 6, 3, 4, 5, 3
    model = NumpyLSTMRegressor(n_apps, ed, hd, seed=1)
    X = rng.integers(0, n_apps, size=(B, T))
    target = rng.normal(size=B)
    mask = np.array([1.0, 1.0, 0.0])  # 하나는 censored(=학습 제외) 케이스 포함

    y_hat, cache = model.forward(X)
    loss, dy_hat = model.smse_loss(y_hat, target, mask)
    grads = model.backward(X, cache, dy_hat)

    eps = 1e-5
    max_rel_err = 0.0
    for name in ["Wz", "Wout"]:
        p = model.params()[name]
        flat = p.reshape(-1)
        g_flat = grads[name].reshape(-1)
        idxs = rng.choice(len(flat), size=min(10, len(flat)), replace=False)
        for idx in idxs:
            orig = flat[idx]
            flat[idx] = orig + eps
            y1, _ = model.forward(X); l1, _ = model.smse_loss(y1, target, mask)
            flat[idx] = orig - eps
            y2, _ = model.forward(X); l2, _ = model.smse_loss(y2, target, mask)
            flat[idx] = orig
            num_grad = (l1 - l2) / (2 * eps)
            ana_grad = g_flat[idx]
            rel_err = abs(num_grad - ana_grad) / max(1e-6, abs(num_grad) + abs(ana_grad))
            max_rel_err = max(max_rel_err, rel_err)
    print(f"[gradient check] max relative error over sampled params: {max_rel_err:.2e}")
    assert max_rel_err < 1e-3, "gradient check failed!"
    print("[gradient check] PASSED")


if __name__ == "__main__":
    _gradient_check()
