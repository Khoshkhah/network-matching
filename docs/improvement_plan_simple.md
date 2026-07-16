# Improvement Plan — the simple version

*(Plain-language summary of [improvement_plan.md](improvement_plan.md). That file has the details,
decisions, and file references; this one just explains the idea.)*

## The situation

Our matcher is better than Hootenanny (the big US-government conflation tool) at the core job:
deciding which road in map A corresponds to which road in map B, accurately and without needing
training data. But Hootenanny does four useful things around the matching that we don't do yet:

1. It checks that all the matches **agree with each other** across the whole network.
2. It says **how sure** it is, and flags doubtful cases for a human to look at.
3. It can **carry information** (names, speed limits, classes) from one map onto the other.
4. It has **published evidence** that it works.

The plan closes those four gaps — and stays a small, easy-to-install library. We are
deliberately **not** trying to become Hootenanny.

## What we'll do, in order

Think of it as ten steps. Each one is useful on its own, so we can stop after any step and still
be better off.

**Step 0 — Clean the house (1 week).**
Fix the tests so they actually fail when something breaks, add automatic test runs, add a
LICENSE file, and regenerate the sample data once with all the useful columns (road class,
speed, one-way). Boring, but every later step stands on this.

**Step 1 — Fix the paper (2–3 weeks).**
The DAG-DTW paper draft describes an old version of the algorithm and mentions functions that no
longer exist. Rewrite it to match the code as it is today. The longer we wait, the worse the
drift gets.

**Step 2 — Record *which part* of each road was matched (3 days).**
Today we know "A matched B"; after this we know "A matched meters 40–180 of B." Three later
steps need this, and it's cheap.

**Step 3 — Copy attributes across maps (1.5 weeks).**
The thing a paying customer actually wants: "take the speed limits and names from the official
map and put them on my OpenStreetMap roads," with a note saying where each value came from and
how trustworthy it is. This becomes one function call and a GeoPackage file.

**Step 4 — A confidence number and a review pile (1.5 weeks).**
Instead of just keep/discard, every match gets a confidence between 0 and 1 and lands in one of
three buckets: **good**, **needs a human look**, or **no match**. No machine learning needed yet —
we reuse statistics the library already computes. The sales pitch number falls out of this:
"only X% needs manual review."

**Step 5 — Prove it works (3.5 weeks).**
Hand-label ~400 roads once (carefully, split so we never grade ourselves on our own homework),
build a script that measures precision/recall/speed/memory, and compare against simpler methods.
Also run once on all of Stockholm county to check we survive bigger data. These numbers go in
the paper and in front of customers.

**Step 6 — The Canada demo (1 week).**
Run the whole chain — match, confidence, attribute copy, evaluation — on the British Columbia
road data we already have in the osm-dra-conflation project. The output package (enriched map
file + review page + accuracy numbers) is what we'd show a BC customer.

**Step 7 — Use names and road classes (1 week).**
Two roads running side by side look the same geometrically; their *names* usually differ. Add
optional name/class comparison as extra evidence. Off by default, so nothing changes unless you
turn it on.

**Step 8 — Find where matches disagree (3–4 days).**
A report that spots two A-roads claiming the same stretch of B-road, or neighbors whose matches
don't line up. This report also tells us whether the expensive follow-up (below) is worth
building at all.

## What we're *not* doing (on purpose)

- **No geometry merging** — producing one fused map is Hootenanny's game and a maintenance
  swamp. We match and transfer attributes; that's most of the value at a tenth of the cost.
- **No machine-learning confidence model yet** — only if the simple version proves too weak,
  and only after the hand-labelled data from Step 5 exists.
- **No automatic global "solver" yet** — only if Step 8's report shows it would actually fix
  real mistakes.
- **No Hootenanny head-to-head yet** — it's the riskiest, fiddliest task (2 weeks, might fail
  for boring technical reasons), and it only makes sense to run once, after our pipeline stops
  changing. Timeboxed: if it fights back, we publish our numbers without it.
- **No country-scale promises** until the Stockholm-county test in Step 5 passes.
- **No cyclic-network research** until the paper is out and a quick survey shows it's worth it.

## The honest math

Everything anyone proposed adds up to ~5–6 months of full-time work — too much. The steps above
are ~11–12 weeks, sequenced so the most sellable things come first. If the Canada opportunity
suddenly gets hot, the shortcut is: Steps 0 → 2 → 3 → 4 → 6 (about 5.5 weeks) and skip the rest
until later.
