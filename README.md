# MemoryNetworks

- Jason Weston, Antoine Bordes, Sumit Chopra, Tomas Mikolov, Alexander M. Rush,
  "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks",
  http://arxiv.org/abs/1502.05698
- Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus,
  "End-To-End Memory Networks",
  http://arxiv.org/abs/1503.08895

## Requirements
- PyTorch 0.2.0_3
- Download the 20 QA bAbI tasks http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz

## Results

Some tasks' performance is not good compared to original results :(

- Task 1: 99.29%
- Task 2: 29.74%
- Task 3: 19.46%
- Task 4: 72.08%
- Task 5: 81.05%
- Task 6: 85.69%
- Task 7: 74.29%
- Task 8: 87.20%
- Task 9: 74.70%
- Task 10: 62.50%
- Task 11: 72.38%
- Task 12: 15.52%
- Task 13: 90.83%
- Task 14: 29.54%   ????
- Task 15: 99.70%
- Task 16: 47.58%
- Task 17: 47.58%
- Task 18: 51.92%
- Task 19: 11.39%
- Task 20: 93.55%


## TODO
- [ ] Random noise (RN)
- [ ] Linear start (RN)
- [ ] joint training
- [ ] set maximum story size
- [ ] compare results with FAIR team (the performance of some tasks is very low)
- [ ] correct optimizer and learning rate
