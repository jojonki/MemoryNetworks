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

For test results of 1k set.

- Task 1: 100.00%
- Task 2: 27.62%
- Task 3: 20.67%
- Task 4: 63.00%
- Task 5: 81.35%
- Task 6: 93.04%
- Task 7: 82.16%
- Task 8: 87.00%
- Task 9: 61.79%
- Task 10: 60.28%
- Task 11: 87.60%
- Task 12: 99.90%
- Task 13: 21.75%
- Task 14: 84.72%
- Task 15: 40.22%
- Task 16: 43.75%
- Task 17: 50.10%
- Task 18: 56.35%
- Task 19: 9.07%
- Task 20: 100.0%


## TODO
- [ ] joing training
- [ ] correct optimizer and learning rate
- [ ] set maximum story size
- [ ] compare results with FAIR team (the performance of some tasks is very low)
