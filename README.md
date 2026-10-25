# Budget allocation case study

A worked demo of a weekly paid-media budget allocation system, built on **synthetic data**. It shows how to move from a loop that chases last week's same-day ROAS to one that values customers over 12 months, respects stock limits, and keeps learning.

> **Important:** all data is synthetic and was built to contain the problems below. The results show how the pieces fit together. They are not a prediction of what any real business would earn.

## The problem

An online retailer spends on several paid channels (Meta, Google, TikTok, Pinterest). Each week, managers move budget toward whatever had the best same-day new-customer ROAS. That loop has five weaknesses:

- **Biased signal.** Same-day revenue sees only part of a customer's 12-month value, and the share differs by channel, so rankings can be wrong.
- **Wrong target.** It chases average return, not the return on the next dollar.
- **No stock awareness.** It spends on categories that cannot ship and cuts well-stocked ones.
- **Disagreeing measurements.** Platform numbers, attribution, the marketing mix model and experiments are never reconciled.
- **No learning.** It reacts to noise and never tests anything new.

## The approach

| Tier | Decision | Cadence | How |
| --- | --- | --- | --- |
| Shared measurement layer | One estimate per channel with error bars | Ongoing | Experiments anchor the other sources, which are corrected against them |
| Tier 1: channel envelope | How much each channel gets | Monthly plan | Given in v1. The system only recommends bounded shifts when the probability is high, and a human approves |
| Tier 2: weekly split | How a channel's budget splits across campaign types and categories | Weekly | Response curves with uncertainty, a Thompson draw, and a greedy fill under constraints |
| Tier 3: daily pacing | How the week's budget is spent day by day across ad sets | Daily | Weekday demand weights, a tighter move limit, held-steady rules, paced stock |

**Tier 2 objective:** maximize expected 12-month contribution from new customers, subject to

- each channel's weekly budget,
- a stockout chance of at most 10% per category,
- a payback floor per category (the next dollar must earn at least a set return),
- a 20% weekly move limit, with inventory overriding it when needed.

An exploration reserve (5% of a channel's budget) funds the most uncertain eligible cells. Its expected cost is capped, and it is trimmed if the cap would be exceeded.

## What the notebook shows

`quince_allocation_demo.ipynb` reads the CSV files in `data/` and walks through:

1. Loading the data
2. The current loop, which is the comparison
3. Why the current loop goes wrong (average return versus the true marginal return)
4. Fixing the signal by estimating 12-month value from same-day revenue
5. Reconciling measurement sources against experiments
6. Response curves with uncertainty (grid posterior with partial pooling)
7. The weekly allocator
8. The weekly recommendation, with a reason for each move
9. A 26 week head to head against the current loop
10. The tier 1 channel shift rule
11. The weekly envelope with rollover
12. Tier 3 daily pacing
13. What the demo covers and what it does not

## Results on the synthetic data

| | Spend | 12-month contribution | Spend wasted on stockouts | Net profit |
| --- | --- | --- | --- | --- |
| Current loop | $8.45M | $8.71M | $0.7M | $0.26M |
| New approach | $4.48M | $7.01M | $0.0M | $2.53M |

- The new approach spends about half as much because it declines dollars that would not pay back. Its total contribution is lower, but its net profit and contribution per dollar are higher.
- Rank correlation with the true 12-month return per dollar rises from 0.41 (same-day ROAS) to 0.94 (corrected signal).
- Over 300 redraws of the source readings, the average error is about 26% for platform numbers, 13% for attribution and 9% for the test-anchored combination.

## Repository

```
.
├── README.md
├── quince_allocation_demo.ipynb   # the notebook, with outputs
├── generate_data.py               # recreates the files in data/
├── requirements.txt
└── data/
    ├── synthetic_quince_weekly_history.csv
    ├── synthetic_quince_cohorts.csv
    ├── synthetic_quince_source_readings.csv
    ├── synthetic_quince_channel_budgets.csv
    ├── synthetic_quince_stock_caps.csv
    └── synthetic_quince_cells_truth.csv
```

| File | What it is | Used for |
| --- | --- | --- |
| `synthetic_quince_weekly_history.csv` | 52 weeks of spend, new customers and same-day revenue per cell, from the current loop | All learning |
| `synthetic_quince_cohorts.csv` | Mature cohorts: same-day revenue as a share of 12-month revenue | Estimating the same-day share |
| `synthetic_quince_source_readings.csv` | Platform, attribution and experiment readings of 12-month ROAS | Reconciling sources |
| `synthetic_quince_channel_budgets.csv` | Weekly budget per channel | The budget constraint |
| `synthetic_quince_stock_caps.csv` | Customers per week the stock can serve, by category | The stock constraint |
| `synthetic_quince_cells_truth.csv` | Hidden true parameters per cell | Scoring results and simulating future weeks only. Never used to learn |

A **cell** is one channel, campaign type and category (30 cells).

## Run it

```bash
pip install -r requirements.txt
jupyter notebook quince_allocation_demo.ipynb
```

Run all cells (about 10 seconds). The notebook only reads `data/`. To recreate the data from scratch, run `python generate_data.py`.

## Assumptions to tune

These are starting points, not facts: a 20% weekly and 10% daily move limit, a 5% exploration reserve with a 2% loss cap, a 10% accepted stockout chance, an 80% probability threshold and 3% shift for tier 1, and payback floors between 0.9 and 1.1 by category.

## Not covered

The notebook does not model correlated errors between sources, adstock or creative effects, holdouts that measure lift, a sell-through bonus for overstocked items, sub-department stock, value net of retention marketing cost, human approval and overrides, or a live test against the manual process. The last section of the notebook lists these.
