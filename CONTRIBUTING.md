# Contributing

Thanks for your interest. Two ground rules keep this project healthy:

## 1. Correctness gates

Any kernel or engine change must pass, in order:
- the iron rule — greedy answers must stay exact after EVERY kernel change,
  before speed is even measured (wrong-but-fast kernels look plausible):
  ```sh
  printf "What is the capital of France?\nWhat is 17 * 23?\n" | \
    ./q36 -n 250 --ctx 4096 2>&1 | tr -cd "\11\12\15\40-\176" | grep -cE "Paris|391"
  # expect 7
  ```
- `q36_ppl` on WikiText-2 (no unexplained regression),
- [scripts/needle_sweep.sh](scripts/needle_sweep.sh) for long-context changes.
Benchmarks accompany performance claims (pinned GPU, `q36_bench`, mean ± std).

## 2. Contributor License Agreement (required)

This project is licensed under the AGPL-3.0, but the copyright holder reserves the right to offer this software under other license terms (dual-licensing) in the future. To keep that option open, all contributions require agreement to the [Contributor License Agreement](CLA.md) -- you keep your copyright, and grant Ambud Sharma the right to relicense your contribution, plus a standard patent license.

Sign by stating your agreement in your first pull request and adding
`Signed-off-by` trailers to your commits (`git commit -s`), as described
at the bottom of CLA.md. Pull requests without a signed CLA cannot be
merged.

If you cannot agree, please open an issue describing the change instead
of submitting code.
