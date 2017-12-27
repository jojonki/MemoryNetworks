import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import to_var
import copy


class MemNN(nn.Module):
    def __init__(self, vocab_size, embd_size, ans_size, story_len, hops=3, dropout=0.0, pe=True):
        super(MemNN, self).__init__()
        self.hops = hops
        self.embd_size = embd_size
        self.position_encoding = pe

        init_rng = 0.1
        self.dropout = nn.Dropout(p=dropout)
        self.A = nn.ModuleList([nn.Embedding(vocab_size, embd_size) for _ in range(hops+1)])
        for i in range(len(self.A)):
            self.A[i].weight = nn.Parameter(torch.randn(vocab_size, embd_size).normal_(-init_rng, init_rng))
        self.B = self.A[0] # query encoder

        # Temporal Encoding: see 4.1
        self.TA = nn.Parameter(torch.randn(1, story_len, 1).normal_(-init_rng, init_rng))
        self.TC = nn.Parameter(torch.randn(1, story_len, 1).normal_(-init_rng, init_rng))

    def forward(self, x, q):
        # x (bs, story_len, s_sent_len)
        # q (bs, q_sent_len)

        bs = x.size(0)
        story_len = x.size(1)
        s_sent_len = x.size(2)

        x = x.view(bs*story_len, -1) # (bs*s_sent_len, s_sent_len)

        u = self.dropout(self.B(q)) # (bs, q_sent_len, embd_size)
        u = torch.sum(u, 1) # (bs, embd_size)

        if self.position_encoding:
            J = s_sent_len
            d = self.embd_size
            pe = to_var(torch.zeros(self.hops, J))
            for k in range(self.hops):
                for j in range(J):
                    l_kj = (1 - j / J) - (k + 1 / d) * (1 - 2 * j / J) # k starts 0
                    pe[k][j] = l_kj

        # Adjacent weight tying
        for k in range(self.hops):
            m = self.dropout(self.A[k](x))            # (bs*story_len, s_sent_len, embd_size)
            m = m.view(bs, story_len, s_sent_len, -1) # (bs, story_len, s_sent_len, embd_size)
            if self.position_encoding:
                l_k = copy.deepcopy(pe[k])                         # (s_sent_len)
                l_k = l_k.unsqueeze(0).unsqueeze(1).unsqueeze(3)   # (1, 1, s_sent_len, 1)
                l_k = l_k.repeat(bs, story_len, 1, self.embd_size) # (bs, story_len, s_sent_len, embd_size)
                m *= l_k                                           # (bs, story_len, s_sent_len, embd_size)
            m = torch.sum(m, 2) # (bs, story_len, embd_size)
            m += self.TA.repeat(bs, 1, self.embd_size)

            c = self.dropout(self.A[k+1](x))           # (bs*story_len, s_sent_len, embd_size)
            c = c.view(bs, story_len, s_sent_len, -1)  # (bs, story_len, s_sent_len, embd_size)
            c = torch.sum(c, 2)                        # (bs, story_len, embd_size)
            c += self.TC.repeat(bs, 1, self.embd_size) # (bs, story_len, embd_size)

            p = torch.bmm(m, u.unsqueeze(2)).squeeze()      # (bs, story_len)
            p = F.softmax(p, -1)                            # (bs, story_len)
            p = p.unsqueeze(2).repeat(1, 1, self.embd_size) # (bs, story_len, embd_size)
            o = c * p # use m as c, (bs, story_len, embd_size)
            o = torch.sum(o, 1)   # (bs, embd_size)
            u = o + u # (bs, embd_size)

        W = torch.t(self.A[-1].weight) # (embd_size, vocab_size)
        out = torch.bmm(u.unsqueeze(1), W.unsqueeze(0).repeat(bs, 1, 1)).squeeze() # (bs, ans_size)

        return F.log_softmax(out, -1)
