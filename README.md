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

<!-- - Task 1: Acc 100.00% (99.9%) -->
<!-- - Task 2: Acc 25.30% (78.4%) ?? -->
<!-- - Task 3: Acc 23.08% (35.8%) ?  -->
<!-- - Task 4: Acc 75.60% (96.2%) ? -->
<!-- - Task 5: Acc 83.06% (85.9%) -->
<!-- - Task 6: Acc 72.38% (92.1%) ? -->
<!-- - Task 7: Acc 77.12% (78.4%) -->
<!-- - Task 8: Acc 85.99% (87.4%) -->
<!-- - Task 9: Acc 75.20% (76.7%) -->
<!-- - Task 10: Acc 81.55% (82.6%) -->
<!-- - Task 11: Acc 87.30% (95.7%) ? -->
<!-- - Task 12: Acc 99.90% (99.7%) -->
<!-- - Task 13: Acc 91.13% (90.1%) -->
<!-- - Task 14: Acc 84.88% (98.2%) ? -->
<!-- - Task 15: Acc 56.55% (100.0%) ?? -->
<!-- - Task 16: Acc 44.46% (47.9%) -->
<!-- - Task 17: Acc 54.54% (49.9%) -->
<!-- - Task 18: Acc 56.25% (86.4%)  ?? -->
<!-- - Task 19: Acc 10.69% (12.6%) -->
<!-- - Task 20: Acc 98.79% (100.0%%) -->
- Task 1: Acc 99.80%
- Task 2: Acc 31.96%
- Task 3: Acc 31.25%
- Task 4: Acc 70.46%
- Task 5: Acc 82.76%
- Task 6: Acc 86.90%
- Task 7: Acc 72.68%
- Task 8: Acc 85.48%
- Task 9: Acc 86.29%
- Task 10: Acc 83.47%
- Task 11: Acc 86.39%
- Task 12: Acc 99.90%
- Task 13: Acc 93.75%
- Task 14: Acc 85.79%
- Task 15: Acc 87.00%
- Task 16: Acc 45.77%
- Task 17: Acc 49.09%
- Task 18: Acc 55.75%
- Task 19: Acc 12.40%
- Task 20: Acc 99.90%


## TODO
- [ ] Random noise (RN)
- [ ] Linear start (LS)
- [ ] joint training
- [ ] compare results with FAIR team (the performance of some tasks is very low)
- [ ] correct optimizer and learning rate
