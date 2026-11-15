# Статический графовый автоэнкодер (порог корреляции для рёбер).
# Запускай ПОСЛЕ ячейки с results в main_fixed.ipynb. Блоки "# %%" вставь в отдельные ячейки.
# Нужен: pip install torch  (PyTorch Geometric не нужен: свёртка GCN написана на матрицах)

# %% [markdown]
# ## Статический графовый автоэнкодер
# Узлы = акции, ребро = |corr| > порога в окне. Признаки узла: средняя доходность, волатильность,
# средняя |корреляция| с остальными акциями. Энкодер: два слоя GCN. Декодер восстанавливает
# рёбра (скалярное произведение эмбеддингов) и признаки узлов. Обучение только на train.
# Ошибка по окну = среднее по узлам (макро), ошибка по узлу = микроуровень.

# %%
# 1. Построение графов окон (только numpy и pandas)
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class GraphWindows:
    A: np.ndarray        # (W, N, N) бинарная матрица смежности, диагональ нулевая
    A_norm: np.ndarray   # (W, N, N) нормированная D^-1/2 (A+I) D^-1/2
    X: np.ndarray        # (W, N, 3) сырые признаки узлов
    tickers: list


def build_graph_windows(returns, window, step, threshold=0.5):
    R = returns.values
    N = R.shape[1]
    off = ~np.eye(N, dtype=bool)
    A_list, X_list = [], []
    for end in range(window, len(R) + 1, step):
        w = R[end - window:end]
        corr = np.nan_to_num(np.corrcoef(w, rowvar=False))
        abs_corr = np.abs(corr) * off
        A_list.append(((np.abs(corr) > threshold) & off).astype(np.float32))
        X_list.append(np.stack([w.mean(axis=0), w.std(axis=0), abs_corr.sum(axis=1) / (N - 1)], axis=1))
    A = np.stack(A_list)
    X = np.stack(X_list).astype(np.float32)

    A_hat = A + np.eye(N, dtype=np.float32)[None]
    deg_inv_sqrt = 1.0 / np.sqrt(A_hat.sum(axis=-1))
    A_norm = (A_hat * deg_inv_sqrt[:, :, None] * deg_inv_sqrt[:, None, :]).astype(np.float32)
    return GraphWindows(A, A_norm, X, list(returns.columns))


def split_fit_val(train_mask, window, step, fit_share=0.8):
    """Делит train по времени на fit и val; между ними зазор, потому что соседние окна перекрываются."""
    train_idx = np.where(train_mask)[0]
    n_fit = int(fit_share * len(train_idx))
    gap = window // step
    return train_idx[:n_fit], train_idx[n_fit + gap:]


# %%
# 2. Модель и детектор
import copy
import torch
import torch.nn as nn
import torch.nn.functional as Fn


class DenseGCNLayer(nn.Module):
    """H' = A_norm @ (H W): графовая свёртка Kipf & Welling на плотных матрицах."""
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, x, a_norm):          # x: (B, N, d_in), a_norm: (B, N, N)
        return torch.bmm(a_norm, self.lin(x))


class _GAENet(nn.Module):
    def __init__(self, d_feat, d_hid, d_lat, dropout):
        super().__init__()
        self.g1 = DenseGCNLayer(d_feat, d_hid)
        self.g2 = DenseGCNLayer(d_hid, d_lat)
        self.drop = nn.Dropout(dropout)
        self.feat_dec = nn.Sequential(nn.Linear(d_lat, d_hid), nn.ReLU(), nn.Linear(d_hid, d_feat))

    def forward(self, x, a_norm):
        h = self.drop(torch.relu(self.g1(x, a_norm)))
        z = self.g2(h, a_norm)                         # эмбеддинги узлов (B, N, d_lat)
        adj_logits = torch.bmm(z, z.transpose(1, 2))   # логиты рёбер (B, N, N)
        return adj_logits, self.feat_dec(z)


class GraphAEDetector:
    """Графовый автоэнкодер для детекции аномалий.

    fit(data, fit_idx, val_idx)    обучение с ранней остановкой по val
    node_scores(data, idx)         ошибка каждого узла в каждом окне (W, N)
    score(data, idx)               ошибка окна = среднее по узлам
    fit_threshold(data, idx, q)    порог по квантилю на train + статистика узлов для z-оценок
    predict(data)                  1 = тревога, 0 = норма
    node_zscores(data, idx)        насколько ошибка узла выше обычной для этого узла (микроуровень)
    """

    def __init__(self, d_hid=32, d_lat=8, dropout=0.1, lr=1e-3, weight_decay=1e-4,
                 epochs=300, patience=30, batch_size=16, lam=1.0, seed=0):
        self.d_hid, self.d_lat, self.dropout = d_hid, d_lat, dropout
        self.lr, self.weight_decay = lr, weight_decay
        self.epochs, self.patience, self.batch_size = epochs, patience, batch_size
        self.lam, self.seed = lam, seed
        self.model_ = None
        self.threshold_ = None

    def _std_x(self, data):
        return ((data.X - self.mu_) / self.sd_).astype(np.float32)

    def _errors(self, model, a, a_norm, x):
        logits, x_hat = model(x, a_norm)
        bce = Fn.binary_cross_entropy_with_logits(logits, a, pos_weight=self.pos_weight_, reduction="none")
        err_adj = (bce * self.off_).sum(dim=-1) / (a.shape[1] - 1)   # ошибка рёбер узла
        err_feat = ((x_hat - x) ** 2).mean(dim=-1)                    # ошибка признаков узла
        return err_adj + self.lam * err_feat                          # (B, N)

    def fit(self, data, fit_idx, val_idx):
        torch.manual_seed(self.seed)
        rng = np.random.RandomState(self.seed)
        N = data.A.shape[1]

        # статистики только по fit-части
        self.mu_ = data.X[fit_idx].mean(axis=(0, 1))
        self.sd_ = data.X[fit_idx].std(axis=(0, 1)) + 1e-6
        density = data.A[fit_idx].sum() / (len(fit_idx) * N * (N - 1))
        self.pos_weight_ = torch.tensor((1 - density) / max(density, 1e-6), dtype=torch.float32)
        self.off_ = 1 - torch.eye(N)

        Xs = self._std_x(data)
        model = _GAENet(Xs.shape[2], self.d_hid, self.d_lat, self.dropout)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        def batch_loss(idx):
            a = torch.from_numpy(data.A[idx])
            an = torch.from_numpy(data.A_norm[idx])
            x = torch.from_numpy(Xs[idx])
            return self._errors(model, a, an, x).mean()

        best, best_state, bad = np.inf, None, 0
        self.val_history_ = []
        for epoch in range(1, self.epochs + 1):
            model.train()
            perm = rng.permutation(fit_idx)
            for i in range(0, len(perm), self.batch_size):
                opt.zero_grad()
                loss = batch_loss(perm[i:i + self.batch_size])
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                v = batch_loss(val_idx).item()
            self.val_history_.append(v)
            if v < best - 1e-5:
                best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        self.model_ = model
        self.n_epochs_ = epoch
        self.best_val_loss_ = best
        return self

    def node_scores(self, data, idx=None):
        if self.model_ is None:
            raise RuntimeError("Сначала вызови fit()")
        idx = np.arange(len(data.A)) if idx is None else np.asarray(idx)
        Xs = self._std_x(data)
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), 64):
                b = idx[i:i + 64]
                out.append(self._errors(self.model_, torch.from_numpy(data.A[b]),
                                        torch.from_numpy(data.A_norm[b]), torch.from_numpy(Xs[b])).numpy())
        return np.concatenate(out)

    def score(self, data, idx=None):
        return self.node_scores(data, idx).mean(axis=1)

    def fit_threshold(self, data, idx, q=0.99):
        ns = self.node_scores(data, idx)
        self.threshold_ = float(np.quantile(ns.mean(axis=1), q))
        self.node_mu_ = ns.mean(axis=0)
        self.node_sd_ = ns.std(axis=0) + 1e-6
        return self

    def predict(self, data):
        if self.threshold_ is None:
            raise RuntimeError("Сначала вызови fit_threshold()")
        return (self.score(data) > self.threshold_).astype(int)

    def node_zscores(self, data, idx):
        return (self.node_scores(data, idx) - self.node_mu_) / self.node_sd_

