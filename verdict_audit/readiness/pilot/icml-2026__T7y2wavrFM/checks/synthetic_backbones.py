"""Three small stand-in RRG backbones that consume the repository's batch schema
(images, context_images, caption ids, cls_labels, context_cls_labels, ...) and return a list of
per-task losses, mirroring models/blip.py: loss_lm and loss_cls(+context loss) with a
[B,4,14] classification layout and the base_probs logit adjustment. They are NOT the real
RRG models; they only provide different gradient-producing backbones for a drop-in test."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_COND, N_STATE = 14, 4


class RRGNet(nn.Module):
    def __init__(self, kind, vocab=30527, D=64, n_aux=1, prompt_len=15):
        super().__init__()
        self.kind, self.D, self.n_aux, self.prompt_len = kind, D, n_aux, prompt_len
        if kind == "mlp":
            self.enc = nn.Sequential(nn.AdaptiveAvgPool2d(8), nn.Flatten(), nn.Linear(192, 128),
                                     nn.ReLU(), nn.Linear(128, D))
        elif kind == "cnn":
            self.enc = nn.Sequential(nn.Conv2d(3, 8, 8, 4), nn.ReLU(), nn.Conv2d(8, 16, 4, 4), nn.ReLU(),
                                     nn.Conv2d(16, 32, 2, 2), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                                     nn.Flatten(), nn.Linear(32, D))
        elif kind == "transformer":
            self.patch = nn.Conv2d(3, D, 32, 32)
            self.enc_tf = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(D, 4, 128, dropout=0.0, batch_first=True), 2)
        else:
            raise ValueError(kind)
        self.emb = nn.Embedding(vocab, D)
        if kind == "transformer":
            self.dec = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(D, 4, 128, dropout=0.0, batch_first=True), 2)
        else:
            self.dec = nn.GRU(D, D, batch_first=True)
        self.out = nn.Linear(D, vocab)
        self.cls_head1 = nn.Linear(D, N_COND * N_STATE)
        self.cls_head2 = nn.Linear(D, N_COND * N_STATE)
        for h in (self.cls_head1, self.cls_head2):
            nn.init.normal_(h.weight, std=0.001)
            nn.init.constant_(h.bias, 0)
        self.align = nn.Linear(D, D)

    def encode(self, img):
        if self.kind == "transformer":
            t = self.patch(img).flatten(2).transpose(1, 2)
            t = self.enc_tf(t)
            return t, t.mean(1)
        f = self.enc(img)
        return f.unsqueeze(1), f

    def forward(self, b, base_probs):
        tok, pooled = self.encode(b["images"])
        _, ctx_pooled = self.encode(b["context_images"])
        # classification (layout and logit adjustment as in models/blip.py)
        cp = self.cls_head1(pooled).view(-1, N_STATE, N_COND)
        cc = self.cls_head2(ctx_pooled).view(-1, N_COND, N_STATE).permute(0, 2, 1)
        adj = torch.log(torch.from_numpy(base_probs).float()).view(1, -1)
        cp = torch.cat([cp[:, :1], cp[:, 1:2] + adj, cp[:, 2:]], 1)
        cc = torch.cat([cc[:, :1], cc[:, 1:2] + adj, cc[:, 2:]], 1)
        ce = nn.CrossEntropyLoss()
        loss_cls = ce(cp, b["cls_labels"]) + ce(cc, b["context_cls_labels"])
        # language modelling: targets masked on pad and on the 1+14 prompt positions
        ids, att = b["caption_ids"], b["caption_attmask"]
        x = self.emb(ids[:, :-1])
        if self.kind == "transformer":
            T = x.shape[1]
            causal = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
            h = self.dec(x, tok, tgt_mask=causal)
        else:
            h, _ = self.dec(x, pooled.unsqueeze(0).contiguous())
        logits = self.out(h)
        tgt = ids[:, 1:].clone()
        tgt[att[:, 1:] == 0] = -100
        tgt[:, : self.prompt_len - 1] = -100
        loss_lm = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)
        losses = [loss_lm, loss_cls]
        if self.n_aux >= 2:  # image-text alignment auxiliary task (InfoNCE on pooled features)
            mask = (att[:, 1:] == 1).float().unsqueeze(-1)
            txt = (self.emb(ids[:, 1:]) * mask).sum(1) / mask.sum(1).clamp(min=1)
            a = F.normalize(self.align(pooled), dim=-1)
            t = F.normalize(txt, dim=-1)
            sim = a @ t.T / 0.07
            lab = torch.arange(sim.shape[0])
            losses.append(0.5 * (F.cross_entropy(sim, lab) + F.cross_entropy(sim.T, lab)))
        return losses


def load_split(path):
    d = np.load(path)
    return {k: d[k] for k in d.files if k != "captions"}


def batches(split, bs, n_steps):
    n = split["images"].shape[0]
    for s in range(n_steps):
        i = (s * bs) % n
        sl = slice(i, i + bs)
        yield {
            "images": torch.from_numpy(split["images"][sl]),
            "context_images": torch.from_numpy(split["context_images"][sl]),
            "cls_labels": torch.from_numpy(split["cls_labels"][sl]),
            "context_cls_labels": torch.from_numpy(split["context_cls_labels"][sl]),
            "caption_ids": torch.from_numpy(split["caption_ids"][sl]),
            "caption_attmask": torch.from_numpy(split["caption_attmask"][sl]),
        }
