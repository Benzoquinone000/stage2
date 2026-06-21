# Data Pipeline

The expected pipeline is:

1. Read raw examples.
2. Clean and normalize text.
3. Tokenize text.
4. Convert tokens to ids.
5. Build masks and labels.
6. Collate variable-length examples into batches.
