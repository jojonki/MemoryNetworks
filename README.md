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

- Task 1: Acc 100.00%
- Task 2: Acc 25.30%
- Task 3: Acc 23.08%
- Task 4: Acc 75.60%
- Task 5: Acc 83.06%
- Task 6: Acc 72.38%
- Task 7: Acc 77.12%
- Task 8: Acc 85.99%
- Task 9: Acc 75.20%
- Task 10: Acc 81.55%
- Task 11: Acc 87.30%
- Task 12: Acc 99.90%
- Task 13: Acc 91.13%
- Task 14: Acc 84.88%
- Task 15: Acc 56.55%
- Task 16: Acc 44.46%
- Task 17: Acc 54.54%
- Task 18: Acc 56.25%
- Task 19: Acc 10.69%
- Task 20: Acc 98.79%


## TODO
- [ ] Random noise (RN)
- [ ] Linear start (RN)
- [ ] joint training
- [ ] set maximum story size
- [ ] compare results with FAIR team (the performance of some tasks is very low)
- [ ] correct optimizer and learning rate
