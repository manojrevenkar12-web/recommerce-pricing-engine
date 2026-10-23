# Working in this repository

- `rpe run` rebuilds everything (history, contracts, warehouse, models, API benchmark, evaluation,
  steering report) and ends with the release gate; `pytest -q -m "not pipeline"` runs in seconds,
  `pytest -q` also runs the pipeline tests (builds the data and models once if missing).
- Only `src/rpe/sim` and `src/rpe/evaluation` may read `data/truth` or `config/world.yaml`. Models,
  engine, API and steering never do (`tests/test_pipeline.py` checks it).
- Business settings go in `config/` (pricing.yaml, costs.yaml, steering.yaml, release.yaml), not in code.
- New raw sources need a contract in `config/contracts/`.
- Never change the engine to pass `scenarios/heldout/`. Their first-run results are in
  `scenarios/heldout/FIRST_RUN.md` and are not rewritten; if you study their failures, move them to
  `scenarios/dev/` and have someone who has not seen the engine write new held-out scenarios.
- Price response is estimated only from randomised-test rows; keep it that way (docs/adr/0003).
