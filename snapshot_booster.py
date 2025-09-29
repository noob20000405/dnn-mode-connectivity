import os, glob, json
from typing import List, Union, Callable, Iterable, Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F

# 你现有的工具 & 模型工厂：
# - build_model(): () -> torch.nn.Module  (同一架构、同一 num_classes)
# - utils.bn_update(loader_train_aug, model)
# - device 已在外部设置
# 这里通过依赖注入的方式传进来，避免耦合训练脚本。

class SnapshotBooster:
    def __init__(
        self,
        build_model_fn: Callable[[], torch.nn.Module],
        bn_loader_train_aug,   # DataLoader，使用 transform_train（带增强）
        device: torch.device,
        # α-learning 超参
        mode: str = "linear",  # "linear" or "logpool"
        steps: int = 150,
        lr: float = 0.3,
        l2: float = 1e-3,      # toward uniform
        eta: float = 1e-2,     # correlation penalty
        crit_fraction: float = 0.5,  # 选取验证集底部 margin 的比例；=0 或 >=1 表示不用子集
        utils_module=None,     # 传入你仓库里的 utils（必须含 bn_update）
    ):
        assert mode in ("linear", "logpool")
        self.build_model = build_model_fn
        self.bn_loader = bn_loader_train_aug
        self.device = device
        self.mode = mode
        self.steps = steps
        self.lr = lr
        self.l2 = l2
        self.eta = eta
        self.crit_fraction = crit_fraction
        self.utils = utils_module

    # ------------ 快照来源统一适配 ------------
    @staticmethod
    def _iter_snapshots(
        snapshots: Union[List[Dict[str, torch.Tensor]], List[str]],
        ckpt_key: Optional[str] = None,
        map_fn: Optional[Callable[[Any], Dict[str, torch.Tensor]]] = None,
    ) -> Iterable[Dict[str, torch.Tensor]]:
        """
        - 若传入 list[state_dict]：直接 yield
        - 若传入 list[path]：torch.load(path)，然后：
            * 若 ckpt_key 给出，则用 ckpt[ckpt_key]
            * 否则若有 "state_dict" 就用它，否则直接当成 state_dict
        - 若给了 map_fn，则最后用 map_fn(ckpt) 取出 state_dict
        """
        if len(snapshots) == 0:
            return
        first = snapshots[0]
        if isinstance(first, dict):
            for sd in snapshots:
                yield sd
        else:
            for p in snapshots:
                ckpt = torch.load(p, map_location="cpu")
                if map_fn is not None:
                    sd = map_fn(ckpt)
                else:
                    if ckpt_key is not None:
                        sd = ckpt[ckpt_key]
                    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
                        sd = ckpt["state_dict"]
                    else:
                        sd = ckpt
                yield sd

    # ------------ 选难样本子集 ------------
    @torch.no_grad()
    def _select_critical_indices(self, ref_model, val_loader, frac: float):
        if not (0.0 < frac < 1.0):
            return None
        # BN 对齐训练分布
        self.utils.bn_update(self.bn_loader, ref_model)
        ref_model.eval()
        logits_all, labels_all = [], []
        for x, y in val_loader:
            x = x.to(self.device, non_blocking=True)
            logits_all.append(ref_model(x).detach().cpu())
            labels_all.append(y.detach().cpu())
        logits = torch.cat(logits_all, 0)
        labels = torch.cat(labels_all, 0)
        true = logits[torch.arange(labels.size(0)), labels]
        tmp = logits.clone()
        tmp[torch.arange(labels.size(0)), labels] = -1e9
        other = tmp.max(1).values
        margins = (true - other).numpy()
        thr = np.quantile(margins, frac)
        return np.nonzero(margins <= thr)[0].astype(np.int64)

    # ------------ 在 val 上学习 α ------------
    def learn_alpha(
        self,
        snapshots: Union[List[Dict[str, torch.Tensor]], List[str]],
        val_loader,                        # 建议用 no-aug 的 transform_test
        ref_for_subset: Optional[torch.nn.Module] = None,
        ckpt_key: Optional[str] = None,
        map_fn: Optional[Callable[[Any], Dict[str, torch.Tensor]]] = None,
    ) -> np.ndarray:
        # 收集标签
        y_list = []
        for _, y in val_loader:
            y_list.append(y.numpy())
        y = np.concatenate(y_list, 0).astype(np.int64)
        N = y.shape[0]

        # 选取子集（可选）
        if ref_for_subset is not None and (0.0 < self.crit_fraction < 1.0):
            S_idx = self._select_critical_indices(ref_for_subset, val_loader, self.crit_fraction)
        else:
            S_idx = None

        # 缓存每个快照在 val 上的 probs/log-probs，以及用于相关性矩阵的 probs
        P_list, L_list = [], []   # probs / log-probs
        for sd in self._iter_snapshots(snapshots, ckpt_key=ckpt_key, map_fn=map_fn):
            m = self.build_model()
            m.load_state_dict(sd, strict=True)
            m.to(self.device)
            self.utils.bn_update(self.bn_loader, m)   # ★ 关键：用训练增强分布重估 BN
            m.eval()

            probs_chunks, logp_chunks = [], []
            with torch.no_grad():
                for x, _ in val_loader:
                    x = x.to(self.device, non_blocking=True)
                    out = m(x)
                    if self.mode == "logpool":
                        logp_chunks.append(F.log_softmax(out, dim=1).double().cpu().numpy())
                        probs_chunks.append(F.softmax(out, dim=1).double().cpu().numpy())
                    else:
                        probs_chunks.append(F.softmax(out, dim=1).double().cpu().numpy())
            if self.mode == "logpool":
                L_list.append(np.concatenate(logp_chunks, 0).astype(np.float64))
            P_list.append(np.concatenate(probs_chunks, 0).astype(np.float64))

            del m
            torch.cuda.empty_cache()

        K = len(P_list)
        assert K > 0
        C = P_list[0].shape[1]
        P = np.stack(P_list, 0).astype(np.float64)        # [K,N,C]
        if self.mode == "logpool":
            L = np.stack(L_list, 0).astype(np.float64)    # [K,N,C]

        # 相关性矩阵 M：基于 true-class prob 的行中心化协方差
        F_tc = np.stack([Pi[np.arange(N), y] for Pi in P_list], 0).astype(np.float64)  # [K,N]
        F_tc -= F_tc.mean(axis=1, keepdims=True)
        M = (F_tc @ F_tc.T) / float(max(1, N))            # [K,K]
        tr = np.trace(M)
        if np.isfinite(tr) and tr > 1e-12:
            M *= (K / tr)

        # α 参数化为 softmax(phi)
        phi = torch.zeros(K, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([phi], lr=self.lr)

        y_t = torch.from_numpy(y).long()
        u_t = torch.ones(K, dtype=torch.float64) / float(K)
        M_t = torch.from_numpy(M).to(torch.float64)

        if S_idx is not None:
            S_t = torch.from_numpy(S_idx).long()
        else:
            S_t = None

        P_t = torch.from_numpy(P).to(torch.float64)
        if self.mode == "logpool":
            L_t = torch.from_numpy(L).to(torch.float64)

        for _ in range(1, self.steps + 1):
            opt.zero_grad()
            w = torch.softmax(phi, dim=0)  # [K]
            if self.mode == "logpool":
                log_mix = torch.einsum('k,knc->nc', w, L_t)
                if S_t is not None:
                    log_sel = log_mix.index_select(0, S_t)
                    y_sel = y_t.index_select(0, S_t)
                    ce = F.nll_loss(F.log_softmax(log_sel, dim=1).to(torch.float64), y_sel)
                else:
                    ce = F.nll_loss(F.log_softmax(log_mix, dim=1).to(torch.float64), y_t)
            else:
                mix = torch.einsum('k,knc->nc', w, P_t)
                if S_t is not None:
                    mix_sel = mix.index_select(0, S_t)
                    y_sel = y_t.index_select(0, S_t)
                    ce = F.nll_loss(torch.log(mix_sel.clamp_min(1e-12)).to(torch.float64), y_sel)
                else:
                    ce = F.nll_loss(torch.log(mix.clamp_min(1e-12)).to(torch.float64), y_t)

            reg_l2 = self.l2 * torch.sum((w - u_t) * (w - u_t))
            quad = self.eta * torch.sum(w * (M_t @ w))
            loss = ce + reg_l2 + quad
            loss.backward()
            opt.step()

        with torch.no_grad():
            alpha = torch.softmax(phi, dim=0).cpu().numpy().astype(np.float64)
        return alpha

    # ------------ 评测（流式，不缓存全部预测） ------------
    @torch.no_grad()
    def evaluate(
        self,
        snapshots: Union[List[Dict[str, torch.Tensor]], List[str]],
        test_loader,
        alpha: np.ndarray,
        ckpt_key: Optional[str] = None,
        map_fn: Optional[Callable[[Any], Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, float]:
        alpha = np.asarray(alpha, dtype=np.float64)
        mix_accum = None
        labels_all = []

        for sd, w in zip(self._iter_snapshots(snapshots, ckpt_key=ckpt_key, map_fn=map_fn), alpha):
            m = self.build_model()
            m.load_state_dict(sd, strict=True)
            m.to(self.device)
            self.utils.bn_update(self.bn_loader, m)  # ★ 训练分布 BN
            m.eval()

            chunks = []
            for x, y in test_loader:
                x = x.to(self.device, non_blocking=True)
                if self.mode == "logpool":
                    chunks.append(F.log_softmax(m(x), dim=1).float().cpu().numpy())
                else:
                    chunks.append(F.softmax(m(x), dim=1).float().cpu().numpy())
                labels_all.append(y.numpy())
            pred = np.concatenate(chunks, 0)  # [N,C]
            contrib = w * pred
            mix_accum = contrib if mix_accum is None else (mix_accum + contrib)

            del m
            torch.cuda.empty_cache()

        labels = np.concatenate(labels_all, 0)
        if self.mode == "logpool":
            probs = torch.softmax(torch.from_numpy(mix_accum), dim=1).numpy().astype(np.float32)
        else:
            probs = mix_accum.astype(np.float32)

        acc = float((probs.argmax(1) == labels).mean())
        nll = float(-np.log(np.clip(probs[np.arange(labels.size), labels], 1e-12, 1.0)).mean())
        # 15-bin ECE
        bins = np.linspace(0, 1, 16)
        conf = probs.max(1)
        correct = (probs.argmax(1) == labels).astype(np.float32)
        ece = 0.0
        for i in range(15):
            lo, hi = bins[i], bins[i+1]
            m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
            if m.any():
                ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
        return {"acc": acc, "nll": nll, "ece": ece}
