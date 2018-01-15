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

- Task 1: Acc 100.00%  (99.9%)
- Task 2: Acc 97.28%   (78.4%)
- Task 3: Acc 84.98%   (35.8%)
- Task 4: Acc 78.33% ? (96.2%)
- Task 5: Acc 88.51%   (85.9%)
- Task 6: Acc 96.17%   (92.1%)
- Task 7: Acc 92.74%   (78.4%)
- Task 8: Acc 95.97%   (87.4%)
- Task 9: Acc 96.67%   (76.7%)
- Task 10: Acc 88.41%  (82.6%)
- Task 11: Acc 94.25%  (95.7%)
- Task 12: Acc 100.00%  (99.7%)
- Task 13: Acc 94.66%  (90.1%)
- Task 14: Acc 100.00%  (98.2%)
- Task 15: Acc 100.00%  (100.0%)
- Task 16: Acc 48.08%  (47.9%)
- Task 17: Acc 53.83%  (49.9%)
- Task 18: Acc 58.67%  (86.4%)
- Task 19: Acc 29.94%  (12.6%)
- Task 20: Acc 100.00%  (100.0%)


## TODO
- [ ] Impl save and load functions
- [ ] Random noise (RN)
- [ ] Linear start (LS)
- [ ] joint training
- [ ] compare results with FAIR team (the performance of some tasks is very low)
- [ ] correct optimizer and learning rate
