# MemoryNetworks

- Jason Weston, Antoine Bordes, Sumit Chopra, Tomas Mikolov, Alexander M. Rush,
  "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks",
  http://arxiv.org/abs/1502.05698
- Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus,
  "End-To-End Memory Networks",
  http://arxiv.org/abs/1503.08895

## Requirements
- PyTorch 0.3-
- Download the 20 QA bAbI tasks http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz

## Results

I tested my model with 10k dataset. `()` is original MemNNs's performance (1k, 3 hops, PE).


- Task 1: Acc 100.00%   (99.9%)
- Task 2: Acc 97.78%    (78.4%)
- Task 3: Acc 93.55%    (35.8%)
- Task 4: Acc 78.63%    (96.2%) ?
- Task 5: Acc 91.13%    (85.9%)
- Task 6: Acc 93.55%    (92.1%)
- Task 7: Acc 89.42%    (78.4%)
- Task 8: Acc 95.56%    (87.4%)
- Task 9: Acc 96.77%    (76.7%)
- Task 10: Acc 87.90%   (82.6%)
- Task 11: Acc 94.86%   (95.7%)
- Task 12: Acc 100.00%  (99.7%)
- Task 13: Acc 94.76%   (90.1%)
- Task 14: Acc 100.00%  (98.2%)
- Task 15: Acc 100.00%  (100.0%)
- Task 16: Acc 48.39%   (47.9%)
- Task 17: Acc 52.82%   (49.9%)
- Task 18: Acc 56.65%   (86.4%)
- Task 19: Acc 20.67%   (12.6%)
- Task 20: Acc 100.00%  (100.0%)

## TODO
- [ ] Random noise (RN)
- [ ] Linear start (LS)
- [ ] joint training
- [ ] compare results with FAIR team (the performance of some tasks is very low)
- [ ] correct optimizer and learning rate
