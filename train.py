import random
import torch
import torch.nn as nn
from utils import load_data, to_var, vectorize
from memnn import MemNN


train_data, test_data, vocab = load_data('./data/tasks_1-20_v1-2/en', 0, 1)
data = train_data + test_data
print('sample', train_data[0])
w2i = dict((w, i) for i, w in enumerate(vocab, 1))
UNK = '<UNK>'
w2i[UNK] = 0
vocab_size = len(vocab) + 1
story_len = max(len(s) for s, q, a in data)
s_sent_len = max(len(ss) for s, q, a in data for ss in s)
q_sent_len = max(len(q) for s, q, a in data)
print('train num', len(train_data))
print('test num', len(test_data))
print('story_len', story_len)
print('s_sent_len', s_sent_len)
print('q_sent_len', q_sent_len)


def train(model, data, test_data, optimizer, loss_fn, batch_size=64, n_epoch=100):
    random.shuffle(data)
    for epoch in range(n_epoch):
        correct = 0
        count = 0
        for i in range(0, len(data)-batch_size, batch_size):
            batch_data = data[i:i+batch_size]
            story = [d[0] for d in batch_data]
            q = [d[1] for d in batch_data]
            a = [d[2][0] for d in batch_data]
            story = to_var(torch.LongTensor(story))
            q = to_var(torch.LongTensor(q))
            a = to_var(torch.LongTensor(a))
            pred = model(story, q)
            pred_idx = pred.max(1)[1]
            correct += torch.sum(pred_idx == a).data[0]
            count += batch_size

            loss = loss_fn(pred, a)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Acc: {:.2f}% - '.format(correct/count*100), correct, '/', count)

embd_size = 100
model = MemNN(vocab_size, embd_size, vocab_size, story_len)
if torch.cuda.is_available():
    model.cuda()
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.NLLLoss()
vec_train = vectorize(train_data, w2i, story_len, s_sent_len, q_sent_len)
vec_test = vectorize(test_data, w2i, story_len, s_sent_len, q_sent_len)
train(model, vec_train, vec_test, optimizer, loss_fn)
# pred = model(x, q)
