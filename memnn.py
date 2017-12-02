import torch
import torch.nn as nn
import torch.nn.functional as F


class MemNN(nn.Module):
    def __init__(self, vocab_size, embd_size, ans_size, story_len, hops=3, dropout=0.0):
        super(MemNN, self).__init__()
        self.hops = hops
        self.embd_size = embd_size

        init_rng = 0.1
        self.dropout = nn.Dropout(p=dropout)
        self.A = nn.ModuleList([nn.Embedding(vocab_size, embd_size) for _ in range(hops+1)])
        for i in range(len(self.A)):
            self.A[i].weight = nn.Parameter(torch.randn(vocab_size, embd_size).normal_(-init_rng, init_rng))
        self.TA = nn.Parameter(torch.randn(1, story_len, 1).normal_(-init_rng, init_rng))
        self.TC = nn.Parameter(torch.randn(1, story_len, 1).normal_(-init_rng, init_rng))

    def forward(self, x, q):
        # x (bs, story_len, s_sent_len)
        # q (bs, q_sent_len)

        bs = x.size(0)
        story_len = x.size(1)
        s_sent_len = x.size(2)

        x = x.view(bs*story_len, -1) # (bs*s_sent_len, s_sent_len)

        u = self.dropout(self.A[1](q)) # (bs, q_sent_len, embd_size)
        u = torch.sum(u, 1) # (bs, embd_size)

        for k in range(self.hops):
            m = self.dropout(self.A[k](x)) # (bs*story_len, s_sent_len, embd_size)
            m = m.view(bs, story_len, s_sent_len, -1) # (bs, story_len, s_sent_len, embd_size)
            m = torch.sum(m, 2) # (bs, story_len, embd_size)
            m += self.TA.repeat(bs, 1, self.embd_size)

            c = self.dropout(self.A[k+1](x)) # (bs*story_len, s_sent_len, embd_size)
            c = c.view(bs, story_len, s_sent_len, -1) # (bs, story_len, s_sent_len, embd_size)
            c = torch.sum(c, 2) # (bs, story_len, embd_size)
            c += self.TC.repeat(bs, 1, self.embd_size)

            p = torch.bmm(m, u.unsqueeze(2)) # (bs, story_len, 1)
            p = F.softmax(p.squeeze()).unsqueeze(2)
            o = c * p # use m as c, (bs, story_len, embd_size)
            o = torch.sum(o, 1) # (bs, embd_size)
            u = o + u # (bs, embd_size)
        W = torch.t(self.A[-1].weight) # (embd_size, vocab_size)
        out = torch.bmm(u.unsqueeze(1), W.unsqueeze(0).repeat(bs, 1, 1)).squeeze() # (bs, ans_size)
        return F.log_softmax(out)
