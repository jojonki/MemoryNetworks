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

Some tasks' performance is not good compared to original results :(

- Task 1: 99.70%
- Task 2: 31.35%
- Task 3: 24.29%
- Task 4: 71.47%
- Task 5: 83.97%
- Task 6: 54.13%
- Task 7: 86.79%
- Task 8: 86.49%
- Task 9: 79.94%
- Task 10: 65.02%
- Task 11: 86.29%
- Task 12: 97.58%
- Task 13: 91.13%
- Task 14: 71.77%
- Task 15: 55.95%
- Task 16: 47.48%
- Task 17: 52.02%
- Task 18: 53.73%
- Task 19: 9.38%
- Task 20: 98.89%


## TODO
- [ ] Random noise (RN)
- [ ] Linear start (RN)
- [ ] joint training
- [ ] set maximum story size
- [ ] compare results with FAIR team (the performance of some tasks is very low)
- [ ] correct optimizer and learning rate
