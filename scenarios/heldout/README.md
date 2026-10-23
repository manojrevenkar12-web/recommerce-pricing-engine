# Held-out scenarios

These five scenarios were written using only `docs/SCENARIOS.md` (the format and what each override
does), without seeing the pricing engine, its code, its training data, the dev scenarios or any
results. Run each once and record its first-run result in `FIRST_RUN.md` before changing anything.

- **textbook-curriculum-switch**: on day 60 a curriculum change halves textbook demand overnight, textbook prices slide 40% over a month, and seller sessions jump 60% for two months as students sell obsolete books.
- **spring-clearout-warehouse-squeeze**: the warehouse runs at 65% capacity all period while a spring clean-out lifts seller sessions 80% from late February to mid-May and CD/DVD demand slowly declines 30%.
- **tv-adaptation-boom-bust**: a hit TV adaptation nearly doubles fiction demand on day 30 (prices +40%, comics demand +50%), then everything snaps back to normal on day 90.
- **competitor-exit-and-reentry**: a competitor closes on day 20 (sessions +70%, games/Blu-ray/DVD prices +20%), then relaunches on day 106 with a games price war that drops games prices to 30% below normal.
- **slow-burn-recession**: over four months demand erodes 25% (games a further 20%), prices soften 15%, and seller sessions rise 50%, with 15% less warehouse capacity throughout.
