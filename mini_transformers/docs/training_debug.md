# Training Debug

Recommended checks:

- Print batch tensor shapes before the first forward pass.
- Check label ranges before cross entropy.
- Monitor loss values for NaN or Inf.
- Overfit a tiny dataset before scaling up.
- Verify validation metrics at a fixed interval.
- The trainer raises `FloatingPointError` when loss contains NaN or Inf.
- Use `--save-steps` and `--resume-from-checkpoint` while debugging long runs.
