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

For test results of 1k set. The performance is not good so far :(

- Task 1: 99.80%
- Task 2: 16.94%
- Task 3: 19.25%
- Task 4: 67.14%
- Task 5: 81.96%
- Task 6: 47.08%
- Task 7: 22.18%
- Task 8: 86.19%
- Task 9: 59.48%
- Task 10: 43.55%
- Task 11: 84.27%
- Task 12: 16.53%
- Task 13: 93.65%
- Task 14: 78.84%
- Task 15: 80.65%
- Task 16: 42.74%
- Task 17: 50.50%
- Task 18: 50.81%
- Task 19: 7.96%
- Task 20: 99.90%


## TODO
- [ ] joint training
- [ ] set maximum story size
- [ ] compare results with FAIR team (the performance of some tasks is very low)
- [ ] correct optimizer and learning rate
