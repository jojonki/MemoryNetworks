import random
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import load_data, to_var, vectorize
from memnn import MemNN


def test(model, data, batch_size):
    correct = 0
    count = 0
    for i in tqdm(range(0, len(data)-batch_size, batch_size)):
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

        # loss = loss_fn(pred, a)
    print('Test Acc: {:.2f}% - '.format(correct/count*100), correct, '/', count)


def adjust_lr(optimizer, epoch):
    if (epoch+1) % 25 == 0: # see 4.2
        for pg in optimizer.param_groups:
            pg['lr'] *= 0.5
            print('Learning rate is set to', pg['lr'])


def train(model, data, test_data, optimizer, loss_fn, batch_size=32, n_epoch=100):
    for epoch in range(n_epoch):
        # print('epoch', epoch)
        correct = 0
        count = 0
        random.shuffle(data)
        for i in range(0, len(data)-batch_size, batch_size):
            batch_data = data[i:i+batch_size]
            story = [d[0] for d in batch_data]
            q = [d[1] for d in batch_data]
            a = [d[2][0] for d in batch_data]
            story = to_var(torch.LongTensor(story))
            q = to_var(torch.LongTensor(q))
            a = to_var(torch.LongTensor(a))

            optimizer.zero_grad()
            pred = model(story, q)

            loss = loss_fn(pred, a)
            loss.backward()
            optimizer.step()

            pred_idx = pred.max(1)[1]
            correct += torch.sum(pred_idx == a).data[0]
            count += batch_size

            # for p in model.parameters():
            #     torch.nn.utils.clip_grad_norm(p, 40.0)

        if epoch % 10 == 0:
            print('=======Epoch {}======='.format(epoch))
            print('Training Acc: {:.2f}% - '.format(correct/count*100), correct, '/', count)
            test(model, test_data, batch_size)

        # adjust_lr(optimizer, epoch)


embd_size = 25
UNK = '<UNK>'
lr = 0.01

def run():
    for i in range(20):
        print('Task', i+1)
        train_data, test_data, vocab = load_data('./data/tasks_1-20_v1-2/en-10k', 0, i+1)
        data = train_data + test_data
        print('sample', train_data[0])

        w2i = dict((w, i) for i, w in enumerate(vocab, 1))
        w2i[UNK] = 0
        vocab_size = len(vocab) + 1
        story_len = max(len(s) for s, q, a in data)
        s_sent_len = max(len(ss) for s, q, a in data for ss in s)
        q_sent_len = max(len(q) for s, q, a in data)
        print('train num', len(train_data))
        print('test num', len(test_data))
        print('vocab_size', vocab_size)
        print('embd_size', embd_size)
        print('story_len', story_len)
        print('s_sent_len', s_sent_len)
        print('q_sent_len', q_sent_len)

        model = MemNN(vocab_size, embd_size, vocab_size, story_len)
        if torch.cuda.is_available():
            model.cuda()
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
        loss_fn = nn.NLLLoss()
        vec_train = vectorize(train_data, w2i, story_len, s_sent_len, q_sent_len)
        vec_test = vectorize(test_data, w2i, story_len, s_sent_len, q_sent_len)
        train(model, vec_train, vec_test, optimizer, loss_fn)

        print('Final Acc')
        test(model, vec_test, 32)

run()
