"""Generate the synthetic Quince-style data files in data/.

The notebook (quince_allocation_demo.ipynb) only reads these files. Run this script
(python generate_data.py) to recreate them. All numbers are made up. The data is built to
contain the problems described in the case: a same-day signal that sees only part of a
customer's value, saturating returns, and tight stock in one category.
"""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

# (max weekly customers a, half-saturation spend k, same-day share s) by channel and campaign type
types = {
    ("Meta", "prospecting"):      (900, 40000, 0.30),
    ("Meta", "DPA"):              (500, 12000, 0.65),
    ("Google", "prospecting"):    (500, 30000, 0.35),
    ("Google", "shopping"):       (700, 15000, 0.70),
    ("TikTok", "prospecting"):    (480, 25000, 0.25),
    ("Pinterest", "prospecting"): (300, 25000, 0.30),
}
# base 12-month revenue per customer, contribution margin, demand size factor
cats = {
    "Sweaters":    (168, 0.55, 1.20),
    "Shirts":      (114, 0.60, 1.00),
    "Jackets":     (252, 0.50, 0.60),
    "Pants":       (144, 0.55, 0.80),
    "Accessories": (72,  0.65, 0.90),
}
type_value_mult = {"prospecting": 1.15, "DPA": 0.90, "shopping": 0.90}   # prospecting customers repeat more
weekly_budget = {"Meta": 150_000, "Google": 120_000, "TikTok": 30_000, "Pinterest": 25_000}


def g(x, a, k):
    return a * x / (k + x)


# ---- cells and their hidden truth -------------------------------------------------------------
rows = []
for (ch, ct), (a, k, s) in types.items():
    for cat, (v0, m, size) in cats.items():
        rows.append(dict(cell_id=f"{ch}|{ct}|{cat}", channel=ch, ctype=ct, category=cat,
                         a=a * size * rng.lognormal(0, 0.12), k=k * rng.lognormal(0, 0.12),
                         s=float(np.clip(s * rng.lognormal(0, 0.08), 0.1, 0.9)),
                         V12=v0 * type_value_mult[ct], margin=m))
cells = pd.DataFrame(rows)
cells["spend0"] = cells.channel.map(weekly_budget) / cells.groupby("channel").cell_id.transform("count")
cells["cust0"] = g(cells.spend0, cells.a, cells.k)

# ---- stock caps: customers per week that stock can serve (Jackets are tight) ----------------
demand = cells.groupby("category").cust0.sum()
cust_cap = (demand * pd.Series({"Sweaters": 1.5, "Shirts": 1.5, "Jackets": 0.45, "Pants": 1.5, "Accessories": 1.5})).round()

# ---- a year of the current loop (chase same-day ROAS, ignore stock) ---------------------------
TARGET = 0.9
noise = lambda n, sd: rng.lognormal(0, sd, n)

def current_loop_step(spend, roas):
    new = spend * (1 + 0.10 * (roas / TARGET - 1)).clip(0.8, 1.25)
    for ch, B in weekly_budget.items():
        m = (cells.channel == ch).values
        even = B / m.sum()
        for _ in range(3):
            new[m] = new[m].clip(0.25 * even, 3.0 * even)
            new[m] = new[m] * B / new[m].sum()
    return new

new_cells = ["TikTok|prospecting|Pants", "TikTok|prospecting|Shirts", "Google|prospecting|Pants"]
is_new = cells.cell_id.isin(new_cells).values            # launched 3 weeks ago, so almost no history

spend = cells.spend0.values * noise(len(cells), 0.2)
for ch, B in weekly_budget.items():
    m = (cells.channel == ch).values
    spend[m] = spend[m] * B / spend[m].sum()

hist = []
for week in range(-52, 0):
    active = ~(is_new & (week < -3))
    sp = np.where(active, spend, 0.0)
    cust = g(sp, cells.a.values, cells.k.values) * noise(len(cells), 0.10)
    sd_rev = cust * cells.V12.values * cells.s.values * noise(len(cells), 0.05)
    hist.append(pd.DataFrame(dict(week=week, cell_id=cells.cell_id, spend=sp, customers=cust, sameday_rev=sd_rev)))
    roas = np.where(sp > 0, sd_rev / np.maximum(sp, 1), np.nan)
    spend = current_loop_step(spend, np.nan_to_num(roas, nan=TARGET))
    if week == -4:
        spend[is_new] = cells.spend0.values[is_new] * 0.4     # new cells launch with small budgets
hist = pd.concat(hist, ignore_index=True).merge(cells[["cell_id", "channel", "ctype", "category"]], on="cell_id")
hist = hist[["week", "cell_id", "channel", "ctype", "category", "spend", "customers", "sameday_rev"]]

# ---- mature cohorts: same-day revenue as a share of 12-month revenue --------------------------
coh = []
for (ch, ct), (a, k, s) in types.items():
    for i, r in enumerate(s * rng.lognormal(0, 0.15, 20), start=1):
        coh.append(dict(channel=ch, ctype=ct, cohort=i, sameday_to_12m_ratio=r))
cohorts = pd.DataFrame(coh)

# ---- measurement sources: platform, attribution (MTA) and a few experiments -------------------
last = hist[hist.week == -1].set_index("cell_id").spend.reindex(cells.cell_id).values
true12 = pd.Series(g(last, cells.a.values, cells.k.values) * cells.V12.values / last, index=cells.index)
true_roas12 = true12.groupby([cells.channel, cells.ctype]).apply(lambda v: np.average(v, weights=last[v.index]))
bias_platform = {"prospecting": 1.10, "DPA": 1.55, "shopping": 1.60}
bias_mta = {"prospecting": 0.95, "DPA": 1.25, "shopping": 1.30}
tested = [("Meta", "prospecting"), ("Meta", "DPA"), ("Google", "shopping"), ("TikTok", "prospecting")]
src = []
for (ch, ct), tr in true_roas12.items():
    src.append(dict(channel=ch, ctype=ct,
                    platform_roas=tr * bias_platform[ct] * rng.lognormal(0, 0.06),
                    mta_roas=tr * bias_mta[ct] * rng.lognormal(0, 0.06),
                    test_roas=tr * rng.lognormal(0, 0.12) if (ch, ct) in tested else np.nan))
sources = pd.DataFrame(src)

# ---- write the files -------------------------------------------------------------------------
os.makedirs("data", exist_ok=True)
hist.round(3).to_csv("data/synthetic_quince_weekly_history.csv", index=False)
cells[["cell_id", "channel", "ctype", "category", "a", "k", "s", "V12", "margin"]].round(4).to_csv("data/synthetic_quince_cells_truth.csv", index=False)
cust_cap.rename("weekly_customer_stock_cap").to_csv("data/synthetic_quince_stock_caps.csv")
pd.Series(weekly_budget, name="weekly_budget").rename_axis("channel").to_csv("data/synthetic_quince_channel_budgets.csv")
cohorts.round(4).to_csv("data/synthetic_quince_cohorts.csv", index=False)
sources.round(4).to_csv("data/synthetic_quince_source_readings.csv", index=False)
print("Wrote:", ", ".join(sorted(os.listdir("data"))))
