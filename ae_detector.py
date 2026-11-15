# Автоэнкодер на корреляциях окна в виде класса.
# Запускай ПОСЛЕ ячейки с results в main_fixed.ipynb. Блоки "# %%" вставь в отдельные ячейки.
# Нужен: pip install torch

# %% [markdown]
# ## Автоэнкодер на корреляциях окна (класс)
# Класс `AutoencoderDetector` не зависит от переменных ноутбука: ему передаются только
# матрица признаков и индексы обучающих окон. Использование в ноутбуке показано в блоке ниже.

# %%
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def window_features(returns, window, step):
    """Верхний треугольник корреляционной матрицы для каждого окна. Строка = окно."""
    iu = np.triu_indices(returns.shape[1], k=1)
    feats = []
    for end in range(window, len(returns) + 1, step):
        w = returns.iloc[end - window:end]
        feats.append(np.corrcoef(w.values, rowvar=False)[iu])
    return np.array(feats, dtype=np.float32)


def split_fit_val(train_mask, window, step, fit_share=0.8):
    """Делит train по времени на fit и val. Между ними зазор, потому что соседние окна перекрываются."""
    train_idx = np.where(train_mask)[0]
    n_fit = int(fit_share * len(train_idx))
    gap = window // step
    return train_idx[:n_fit], train_idx[n_fit + gap:]


class _Net(nn.Module):
    def __init__(self, d_in, d_hid, d_lat, dropout):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_hid, d_lat))
        self.dec = nn.Sequential(nn.Linear(d_lat, d_hid), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_hid, d_in))

    def forward(self, x):
        return self.dec(self.enc(x))


class AutoencoderDetector:
    """Детектор аномалий: ошибка реконструкции автоэнкодера, порог по квантилю на train.

    fit(F, fit_idx, val_idx)   обучение с ранней остановкой по val
    score(F)                   ошибка реконструкции каждой строки F (больше = аномальнее)
    fit_threshold(F, idx, q)   порог = квантиль ошибки на строках idx (только train)
    predict(F)                 1 = тревога, 0 = норма
    """

    def __init__(self, d_hid=128, d_lat=8, dropout=0.1, lr=1e-3, weight_decay=1e-4,
                 epochs=300, patience=30, batch_size=32, seed=0):
        self.d_hid, self.d_lat, self.dropout = d_hid, d_lat, dropout
        self.lr, self.weight_decay = lr, weight_decay
        self.epochs, self.patience, self.batch_size = epochs, patience, batch_size
        self.seed = seed
        self.model_ = None
        self.threshold_ = None

    # --- служебное ---
    def _transform(self, F):
        return ((F - self.mu_) / self.sd_).astype(np.float32)

    # --- обучение ---
    def fit(self, F, fit_idx, val_idx):
        torch.manual_seed(self.seed)
        rng = np.random.RandomState(self.seed)

        # стандартизация только по fit-части: без заглядывания в val и test
        self.mu_ = F[fit_idx].mean(axis=0)
        self.sd_ = F[fit_idx].std(axis=0) + 1e-6
        Z = self._transform(F)

        model = _Net(Z.shape[1], self.d_hid, self.d_lat, self.dropout)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        x_val = torch.from_numpy(Z[val_idx])

        best, best_state, bad = np.inf, None, 0
        self.val_history_ = []
        for epoch in range(1, self.epochs + 1):
            model.train()
            perm = rng.permutation(fit_idx)
            for i in range(0, len(perm), self.batch_size):
                xb = torch.from_numpy(Z[perm[i:i + self.batch_size]])
                opt.zero_grad()
                loss = loss_fn(model(xb), xb)
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                v = loss_fn(model(x_val), x_val).item()
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

    # --- применение ---
    def score(self, F):
        if self.model_ is None:
            raise RuntimeError("Сначала вызови fit()")
        x = torch.from_numpy(self._transform(F))
        with torch.no_grad():
            err = ((self.model_(x) - x) ** 2).mean(dim=1).numpy()
        return err

    def fit_threshold(self, F, idx, q=0.99):
        self.threshold_ = float(np.quantile(self.score(F[idx]), q))
        return self

    def predict(self, F):
        if self.threshold_ is None:
            raise RuntimeError("Сначала вызови fit_threshold()")
        return (self.score(F) > self.threshold_).astype(int)

class MarketAutoencoder(nn.Module):
    def __init__(self, input_dim=84, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)



