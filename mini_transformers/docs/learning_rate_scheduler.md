# Learning Rate Scheduler

The first supported schedules are linear warmup with linear decay and cosine
warmup with cosine decay. Scheduler behavior should be tested with small step
counts before long training runs.

Select one with `--scheduler-type linear`, `cosine`, or `none`. Warmup is
controlled by `--warmup-steps`.
