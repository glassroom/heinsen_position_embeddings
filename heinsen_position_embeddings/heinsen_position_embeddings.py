# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbedPosition(nn.Module):
    """
    Reference implementation of "Encoding Position by Decaying and Updating
    Different Exponentiated States Differently" (Heinsen, 2024).

    Args:
        d_emb: int, number of embedding features.
        d_hid: int, number of hidden position-encoding features.
    
    Inputs:
        x: float tensor of shape [..., n_tok, d_emb] with token states.
        using_prev_context: bool. If True, use previously cached state.

    Output:
        y: float tensor of shape [..., n_tok, d_emb] with updated tokens.
    """
    def __init__(self, d_emb, d_hid):
        super().__init__()
        self.d_emb, self.d_hid = (d_emb, d_hid)
        self.H = nn.Linear(d_emb, d_hid * 2)     # function H in paper
        self.R = nn.Linear(d_hid, d_emb)         # function R in paper

    def extra_repr(self):
        return 'd_emb={}, d_hid={}'.format(d_emb, d_hid)

    def _log_linear_recurrence(self, log_coeffs, prepended_logits):
        "Applies method proposed in https://arxiv.org/abs/2311.06281."
        a_star = F.pad(log_coeffs.cumsum(dim=-2), (0,0, 1,0), value=0)              # [..., 1 + n_tok, d_hid]
        logit0_plus_b_star = torch.logcumsumexp(prepended_logits - a_star, dim=-2)  # [..., 1 + n_tok, d_hid]
        log_linear_recurrence = a_star + logit0_plus_b_star                         # [..., 1 + n_tok, d_hid]
        return log_linear_recurrence[..., 1:, :]                                    # [..., n_tok, d_hid]

    def forward(self, x, using_prev_context=False):
        tup = self.H(x).split(self.d_hid, dim=-1)                    # [..., n_tok, d_hid] x 2
        log_p, h = (F.logsigmoid(tup[0]), tup[1])                    # [..., n_tok, d_hid] x 2

        if using_prev_context:
            prepended_h = torch.cat([self.prev_context, h], dim=-2)  # [..., 1 + n_tok, d_hid]
        else: 
            prepended_h = F.pad(h, (0,0, 1,0), value=0)              # [..., 1 + n_tok, d_hid]

        s = self._log_linear_recurrence(log_p, prepended_h)          # [..., n_tok, d_hid]
        self.prev_context = s[..., -1:, :].detach()                  # [..., 1, d_hid]

        y = x + self.R(s)                                            # [..., n_tok, d_emb]
        return y

