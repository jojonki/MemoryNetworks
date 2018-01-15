import random
import torch
import torch.nn as nn
from utils import load_data, to_var, vectorize
from memnn import MemNN


def test(model, data, w2i, batch_size, task_id):
    model.eval()
    correct = 0
    count = 0
    for i in range(0, len(data)-batch_size, batch_size):
        batch_data = data[i:i+batch_size]
        story = [d[0] for d in batch_data]
        q = [d[1] for d in batch_data]
        a = [d[2][0] for d in batch_data]

        story_len = min(max_story_len, max([len(s) for s in story]))
        s_sent_len = max([len(sent) for s in story for sent in s])
        q_sent_len = max([len(sent) for sent in q])

        vec_data = vectorize(batch_data, w2i, story_len, s_sent_len, q_sent_len)
        story = [d[0] for d in vec_data]
        q = [d[1] for d in vec_data]
        a = [d[2][0] for d in vec_data]

        story = to_var(torch.LongTensor(story))
        q = to_var(torch.LongTensor(q))
        a = to_var(torch.LongTensor(a))
        pred = model(story, q)
        pred_idx = pred.max(1)[1]
        correct += torch.sum(pred_idx == a).data[0]
        count += batch_size
    acc = correct/count*100
    print('Task {} Test Acc: {:.2f}% - '.format(task_id, acc), correct, '/', count)
    return acc


def adjust_lr(optimizer, epoch):
    if (epoch+1) % 25 == 0: # see 4.2
        for pg in optimizer.param_groups:
            pg['lr'] *= 0.5
            print('Learning rate is set to', pg['lr'])


def train(model, train_data, test_data, optimizer, loss_fn, w2i, task_id, batch_size=16, n_epoch=100):
    for epoch in range(n_epoch):
        model.train()
        # print('epoch', epoch)
        correct = 0
        count = 0
        random.shuffle(train_data)
        for i in range(0, len(train_data)-batch_size, batch_size):
            batch_data = train_data[i:i+batch_size]
            story = [d[0] for d in batch_data]
            story_len = min(max_story_len, max([len(s) for s in story]))
            s_sent_len = max([len(sent) for s in story for sent in s])
            q = [d[1] for d in batch_data]
            q_sent_len = max([len(sent) for sent in q])

            vec_data = vectorize(batch_data, w2i, story_len, s_sent_len, q_sent_len)
            story = [d[0] for d in vec_data]
            q = [d[1] for d in vec_data]
            a = [d[2][0] for d in vec_data]

            story = to_var(torch.LongTensor(story))
            q = to_var(torch.LongTensor(q))
            a = to_var(torch.LongTensor(a))

            pred = model(story, q)

            loss = loss_fn(pred, a)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_idx = pred.max(1)[1]
            correct += torch.sum(pred_idx == a).data[0]
            count += batch_size

            # for p in model.parameters():
            #     torch.nn.utils.clip_grad_norm(p, 40.0)

        if epoch % 20 == 0:
            print('=======Epoch {}======='.format(epoch))
            print('Training Acc: {:.2f}% - '.format(correct/count*100), correct, '/', count)
            test(model, test_data, w2i, batch_size, task_id)

        # adjust_lr(optimizer, epoch)


# Set the random seed manually for reproducibility.
seed = 1111
torch.manual_seed(seed)

use_10k = True
max_story_len = 25 # see 4.2 original is 50
embd_size = 30
PAD = '<PAD>'


def run():
    test_acc_results = []
    # for task_id in [2, 3, 4, 6, 11, 14, 15, 18]:
    for task_id in range(1, 20+1):
        print('-*_*_*_*_*_*_*_*_ Task', task_id)
        if use_10k:
            train_data, test_data, vocab = load_data('./data/tasks_1-20_v1-2/en-10k', 0, task_id)
        else:
            train_data, test_data, vocab = load_data('./data/tasks_1-20_v1-2/en', 0, task_id)
        data = train_data + test_data
        print('sample', train_data[0])

        w2i = dict((w, i) for i, w in enumerate(vocab, 1))
        w2i[PAD] = 0
        vocab_size = len(vocab) + 1
        story_len = min(max_story_len, max(len(s) for s, q, a in data))
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
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = nn.NLLLoss()
        # vec_train = vectorize(train_data, w2i, story_len, s_sent_len, q_sent_len)
        # vec_test = vectorize(test_data, w2i, story_len, s_sent_len, q_sent_len)
        # train(model, vec_train, vec_test, optimizer, loss_fn, task_id)
        train(model, train_data, test_data, optimizer, loss_fn, w2i, task_id)

        print('Final Acc')
        acc = test(model, test_data, w2i, 32, task_id)
        test_acc_results.append(acc)

    for i, acc in enumerate(test_acc_results):
        print('Task {}: Acc {:.2f}%'.format(i+1, acc))


run()
