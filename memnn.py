import torch
import torch.nn as nn
import torch.nn.functional as F


class MemNN(nn.Module):
    def __init__(self, vocab_size, embd_size, ans_size, hops=3):
        super(MemNN, self).__init__()
        self.hops = hops

        self.A = nn.ModuleList([nn.Embedding(vocab_size, embd_size) for _ in range(hops+1)])
        # self.A.weight = nn.Parameter(torch.randn(vocab_size, embd_size).normal_(-0.01, 0.01))
        # self.B = nn.Embedding(vocab_size, embd_size)
        # self.B.weight = nn.Parameter(torch.randn(vocab_size, embd_size).normal_(-0.01, 0.01))
        self.fc = nn.Linear(embd_size, ans_size)

    def forward(self, x, q):
        # x (bs, story_len, s_sent_len)
        # q (bs, q_sent_len)

        bs = x.size(0)
        story_len = x.size(1)
        s_sent_len = x.size(2)

        x = x.view(bs*story_len, -1) # (bs*s_sent_len, s_sent_len)

        # TODO temporarl encoding
        m = self.A(x) # (bs*story_len, s_sent_len, embd_size)
        m = m.view(bs, story_len, s_sent_len, -1) # (bs, story_len, s_sent_len, embd_size)
        m = torch.sum(m, 2) # (bs, story_len, embd_size)

        c = self.B(x) # (bs*story_len, s_sent_len, embd_size)
        c = c.view(bs, story_len, s_sent_len, -1) # (bs, story_len, s_sent_len, embd_size)
        c = torch.sum(c, 2) # (bs, story_len, embd_size)

        u = self.A[0](q) # (bs, q_sent_len, embd_size)
        u = torch.sum(u, 1) # (bs, embd_size)

        for _ in range(self.hops):
            p = torch.bmm(m, u.unsqueeze(2)) # (bs, story_len, 1)
            o = c * p # use m as c, (bs, story_len, embd_size)
            o = torch.sum(o, 1) # (bs, embd_size)
            u = o + u # (bs, embd_size)
        out = self.fc(u) # (bs, ans_size)
        return F.log_softmax(out)
