# Full Project Walkthrough — Customer Segmentation with K-Means

Companion to `viva_prep_notes.md`. This document explains the project top to bottom: what every
cell does, what every graph shows and what it actually produced, how testing was done, how EDI and
sustainability were handled, and who else has worked on this dataset.

All numbers below were verified by re-running the pipeline, not read off the write-up. [Certain]

---

# PART 1 — What the project is

## 1.1 The business problem

A retailer has a customer database and a marketing budget. It cannot afford to send every offer to
every customer, and sending the wrong offer wastes money and irritates people. It needs to split the
customer base into a small number of groups where members of a group behave similarly enough that
one campaign suits all of them.

The classical way to do this is a marketing manager drawing lines by hand ("anyone over £50k is
premium"). The AI approach is to let an algorithm find the groupings in the transaction data itself.

## 1.2 Why unsupervised learning

There is no label in the data saying which segment a customer belongs to — segments do not exist
until you create them. That rules out supervised learning entirely. Clustering is the correct family
of technique: it finds structure without being told what to look for.

## 1.3 Why K-Means specifically

The submission's argument (Part B) is that marketing needs a small, *fixed*, explainable number of
segments, because the binding constraint is how many distinct campaigns the team can design and
staff — not how many natural groups happen to exist in the data. K-Means takes the number of
clusters (K) as an input, which matches that constraint, and its output is a *centroid* — literally
an average customer profile a marketer can read.

The two rejected alternatives:

- **Hierarchical clustering** — doesn't need K upfront (you cut the dendrogram wherever you like),
  and the dendrogram shows sub-segment structure. But it is O(n²) in time and memory, so it breaks
  down on real CRM databases.
- **DBSCAN** — finds arbitrarily-shaped clusters and labels outliers as noise rather than forcing
  everyone into a group. But it needs two sensitive parameters (`eps`, `min_samples`), it handles
  clusters of differing density badly, and it gives you whatever number of clusters the data
  produces, which may be 2 or 17.

## 1.4 How K-Means works, in one paragraph

Pick K starting points. Assign every customer to the nearest one by Euclidean distance. Move each
point to the average position of the customers assigned to it. Repeat until nothing moves. What it
is minimising is **inertia** — the total squared distance from each customer to its own centre.
Because the starting points are random, the result can vary, so the code uses `n_init=10` (ten
restarts, best result kept) and `random_state=42` (fixed seed for reproducibility).

Because it uses Euclidean distance, **features must be scaled first**. Income is in tens of
thousands and Recency is 0–99; without standardisation, income would dominate the distance
calculation purely because its numbers are bigger. This is what `StandardScaler` handles.

---

# PART 2 — The dataset

## 2.1 Provenance

The *Customer Personality Analysis* dataset on Kaggle, uploaded by user `imakash3011` in 2021.
<cite index="10-3">It originates with Dr. Omar Romero-Hernandez and covers 2,240 customers of a European retail brand across 2012–2014, with purchase records spanning six product categories and responses to five email campaigns.</cite> <cite index="11-1">Romero-Hernandez is a professor at U.C. Berkeley.</cite>

**A point worth raising before an examiner does:** the dataset name is a misnomer. It contains
**no personality data at all** — no psychometric instrument, no Big Five inventory, nothing. It is
demographics plus transactions plus campaign flags. Several published papers on this dataset claim
to identify traits such as extroversion, openness, conscientiousness, agreeableness and neuroticism
from it, which is not supportable from these columns. Knowing that the title oversells the content
is a credible signal of having read the data rather than the label. [Certain — verified against the
29 columns]

## 2.2 Format

Tab-separated (`sep="\t"`), 2,240 rows × 29 columns, one row per customer. Embedded as a Python
string literal `CSV_DATA` inside the notebook so the marker can run it with no file and no network
access — this is what satisfies the rubric's "include the input data in the notebook" requirement
and comfortably clears the >1,000 data point threshold.

## 2.3 The columns, grouped

| Group | Columns |
|---|---|
| Identity | `ID` |
| Demographics | `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome` |
| Relationship | `Dt_Customer` (enrolment date, DD-MM-YYYY text), `Recency` (days since last purchase) |
| Spend, 6 categories | `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds` |
| Channels, 5 | `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth` |
| Campaign response | `AcceptedCmp1`–`AcceptedCmp5`, `Response` (the 6th/most recent), `Complain` |
| Constants | `Z_CostContact` (always 3), `Z_Revenue` (always 11) |

`Dt_Customer` is the one column pandas will not parse correctly on its own — `dayfirst=True` is
required or dates silently misparse.

---

# PART 3 — Step-by-step walkthrough with every graph

## Step 1 — Load and explore

```
df = pd.read_csv(io.StringIO(CSV_DATA), sep="\t")     → (2240, 29)
df.drop(columns=["Z_CostContact","Z_Revenue","Complain"])
pd.to_datetime(df["Dt_Customer"], dayfirst=True)
df.dropna()                                           → 2216 rows (24 dropped)
```

`Z_CostContact` and `Z_Revenue` are dropped because they are constant — a column with zero variance
contributes zero to any distance calculation and is pure noise in a correlation matrix. `Complain`
is dropped as "unused," which is the weaker of the three justifications (see Part 7).

The 24 dropped rows are all missing `Income`, about 1% of the data. Dropping rather than imputing
avoids injecting 24 artificial customers at the exact centre of the strongest clustering feature.

### GRAPH 1 — Missing Value Map (`sns.heatmap(df.isnull())`)

**What it is meant to show:** a black-and-white map of the whole table, one pixel per cell, where a
marked cell means missing. Used in EDA to see *whether* missingness is patterned (concentrated in
one column, or in a block of rows) or random.

**What it actually shows here: nothing.** The plot is drawn *after* `dropna()` has already run, so
every cell is non-null and the figure is a uniform blank rectangle. The `print` immediately after it
confirms this — it outputs an empty Series.

**This is a real flaw**, and a cheap one to be caught on. The plot should have been drawn before the
drop, where it would have shown a thin scatter of marks in the `Income` column only, supporting the
decision to drop rather than impute. If asked, say the ordering is wrong and the diagnostic value
was lost. Do not claim it "confirms the data is clean" — that is circular.

## Step 2 — Clean, engineer, cap, scale

### Feature engineering

| New feature | Built from | Meaning |
|---|---|---|
| `Age` | `2015 − Year_Birth` | Age at dataset collection |
| `Total_Spending` | Sum of the 6 `Mnt*` columns | Monetary value (the M in RFM) |
| `Total_Purchases` | Sum of 4 purchase-channel columns | Frequency (the F in RFM) |
| `Total_Campaigns_Accepted` | Sum of 6 campaign flags | Marketing responsiveness |
| `Children` | `Kidhome + Teenhome` | Household composition |
| `Customer_Tenure_Days` | `2015-01-01 − Dt_Customer` | Relationship length |

`Recency` already exists in the raw data and supplies the R in RFM. So the engineered set is a
textbook **RFM model plus income**.

**Why the reference date is hard-coded to 2015-01-01 rather than `Timestamp.now()`:** the data was
collected 2012–2014 (earliest `Dt_Customer` is 2012-07-30). Using today's date would add a decade to
every age and make tenure meaningless. This is the correct choice and you should defend it
confidently.

Three customers with `Age > 90` are dropped — birth years 1893, 1899 and 1900. These are almost
certainly data-entry errors. Row count: 2,216 → **2,213**.

### Outlier capping (winsorising)

`Income` and `Total_Spending` are clipped to their 1st and 99th percentiles. This matters because
K-Means minimises *squared* distance, so a single extreme point drags a centroid a long way.

The extreme case in this dataset is an income of **666,666** — a value that is transparently fake,
and about 7× the 99th percentile of 94,461. Capping affects **46 rows** in Income. Clipping instead
of deleting keeps the sample size, which the write-up gives as the reason.

### GRAPH 2 — Correlation Heatmap

**What it shows:** a square colour matrix of Pearson correlations between every numeric column,
diverging red/blue around zero. It answers "which variables move together?" — used to spot
redundancy and to sanity-check the engineered features.

**What it actually produced:**

| Strongest positive | r |
|---|---|
| `MntWines` ↔ `Total_Spending` | +0.893 |
| `MntMeatProducts` ↔ `Total_Spending` | +0.844 |
| `NumStorePurchases` ↔ `Total_Purchases` | +0.822 |
| **`Income` ↔ `Total_Spending`** | **+0.820** |
| `Total_Spending` ↔ `Total_Purchases` | +0.760 |

| Strongest negative | r |
|---|---|
| **`Income` ↔ `NumWebVisitsMonth`** | **−0.651** |
| `Kidhome` ↔ `Total_Spending` | −0.560 |
| `Income` ↔ `Kidhome` | −0.529 |
| `MntMeatProducts` ↔ `NumWebVisitsMonth` | −0.539 |

**Three readings that matter for the viva:**

1. **Wine dominates spend** (r = 0.89). "Total spending" in this dataset is largely "wine spending."
   A useful caveat when you present the high-value segments — they are, disproportionately, wine
   buyers.
2. **The inverse web-visit relationship** (r = −0.65) is the most interesting finding in the whole
   heatmap and the notebook does not comment on it. High-income customers visit the website *less*
   yet spend more; low-income customers browse frequently and buy little. This is the empirical
   basis for the "web-heavy browser, lower conversion" tag on Clusters 1 and 3, and it is a
   genuinely actionable insight — it suggests the website is functioning as a browsing/comparison
   surface for price-sensitive customers while high-value purchasing happens through catalogue and
   store.
3. **Recency correlates with essentially nothing** — its largest absolute correlation with any
   column is 0.200 (with `Response`), and 0.09 with `Total_Campaigns_Accepted`. Everything else is
   under 0.04. Flag this now; it becomes decisive later.

### GRAPH 3 — Three distribution histograms (Age, Income, Total Spending)

**What they show:** the shape of each key variable, with a KDE curve overlaid. Used to check for
skew, bimodality, and implausible values before modelling.

**What they produced:**

| Variable | Mean | Median | Min | Max | Skew | Shape |
|---|---|---|---|---|---|---|
| Age | 46.1 | 45.0 | 19 | 75 | +0.09 | Near-symmetric, roughly normal, peak in the 40s |
| Income | 51,759 | 51,373 | 7,563 | 94,461 | +0.01 | Symmetric — *because capping removed the tail* |
| Total_Spending | 605 | 397 | 13 | 2,126 | **+0.82** | Strongly right-skewed, large mass near zero |

**The key reading is the spending histogram.** Mean (605) is over 1.5× the median (397): most
customers spend very little and a minority spend a great deal. This is the classic retail
value-concentration pattern and it is exactly what makes segmentation worth doing — a single
"average customer" (£605) describes almost nobody.

Note honestly that Income *looks* clean and symmetric only because you capped it three lines
earlier. Pre-capping it had a 666,666 outlier. The histogram is showing post-treatment data.

### Feature selection and scaling

Clustering uses **only four** features: `Income`, `Total_Spending`, `Recency`, `Total_Purchases`.
The engineered `Age`, `Children`, `Customer_Tenure_Days` and `Total_Campaigns_Accepted` are computed
but used **only for profiling afterwards**, not for forming the clusters.

`StandardScaler` transforms each to mean 0, standard deviation 1. Verification from the notebook's
own output: all four means are ~1e-17 and all four SDs are 1.000. Correct.

Rows entering the model: **2,213 × 4**.

## Step 3 — Choose K and fit

### GRAPH 4a — Elbow plot (inertia vs K)

**What it shows:** total within-cluster squared distance for K = 2 to 10. Inertia always falls as K
rises (at K = n it reaches zero), so you look for the "elbow" — the point after which adding
clusters stops buying much.

**What it produced:**

| K | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| Inertia | 4378 | 3460 | 2722 | 2327 | 2049 | 1881 | 1743 | 1618 | 1521 |
| Drop from previous | — | 918 | 738 | 396 | 278 | 168 | 138 | 126 | 96 |

The curve bends most between K = 4 and K = 5: the gain from 3→4 is 738, from 4→5 it more than halves
to 396. **This supports K = 4** and it is the strongest quantitative argument the submission has for
that choice — it is worth leading with, because the silhouette argument goes the other way.

### GRAPH 4b — Silhouette score vs K

**What it shows:** average silhouette coefficient per K. For each point, silhouette compares its
average distance to its own cluster against its average distance to the nearest other cluster.
Range −1 to +1. Above ~0.5 is reasonable structure; 0.25–0.5 is weak/overlapping; below 0.25 means
essentially no separation.

**What it produced:**

| K | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| Silhouette | **0.433** | 0.355 | 0.318 | 0.318 | 0.307 | 0.298 | 0.262 | 0.261 | 0.263 |

The notebook prints "K with highest silhouette score: 2" — this is honest and it is important not to
hide it. The score declines monotonically as K rises, which is the signature of a dataset with **no
strong natural cluster count**. Real behavioural data is usually a continuum, not islands.

### GRAPH 5a — PCA scatter plot (the segmentation picture)

**What it shows:** the 4-dimensional scaled data projected onto its two highest-variance directions,
coloured by cluster. This is the headline visual of the whole project.

**Critical clarification:** PCA is used **for visualisation only**. The model is fitted on all four
scaled features. Do not let anyone leave believing you clustered in PCA space.

**What it produced:**

- PC1 explains **62.9%** of variance, PC2 explains **25.0%** — 87.9% retained in two dimensions,
  which is why the picture is trustworthy as a summary.
- **PC1 loadings:** Income 0.577, Total_Spending 0.594, Total_Purchases 0.561, Recency 0.014.
  PC1 is a near-equal blend of the three value features — it *is* "customer value."
- **PC2 loadings:** Recency **1.000**, everything else ≈ 0.00. PC2 *is* Recency, alone.

So the scatter plot is not an abstract projection. It is a readable 2×2 grid: **horizontal axis =
customer value, vertical axis = recency**. Cluster means confirm it — (PC1, PC2) = Cluster 0
(1.44, −0.90), Cluster 1 (−1.42, −0.86), Cluster 2 (1.55, +0.82), Cluster 3 (−1.28, +0.87).

Visually you should expect four blocks with **clean horizontal boundaries** (recency splits sharply
at the midpoint) and **fuzzy vertical boundaries** (value is continuous, so the left/right groups
bleed into each other). That asymmetry is the honest visual explanation of the 0.318 silhouette.

### GRAPH 5b — Cluster size bar chart

**What it shows:** how many customers landed in each cluster. Guards against the common K-Means
failure of one giant cluster plus three tiny ones.

**What it produced:** 514 / 570 / 535 / 594 — between 23% and 27% each. Remarkably balanced, and a
legitimate strength: every segment is large enough to run a real campaign against and to A/B test.

Be ready for the follow-up though: the balance is partly an artefact. Recency is uniform, so it
splits 50/50 by construction, and value splits roughly 50/50 too. Balance here is not evidence that
the clusters are natural.

## Step 4 — Profile and recommend

The cluster profile table takes the mean of nine descriptive features per cluster — including the
five that were *not* used for clustering, which is the right way round: cluster on a small
interpretable set, then describe the clusters using everything you know.

| Cluster | Age | Income | Spend | Purchases | Campaigns | Recency | Children | Web visits | Tenure | n |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 47.7 | 68,807 | 1,096 | 21.6 | 0.8 | 23.5 | 0.7 | 4.2 | 554 | 514 |
| 1 | 43.4 | 35,102 | 121 | 8.5 | 0.2 | 23.5 | 1.2 | 6.5 | 518 | 570 |
| 2 | 48.6 | 69,614 | 1,166 | 21.5 | 0.7 | 73.3 | 0.6 | 4.0 | 572 | 535 |
| 3 | 45.0 | 36,909 | 140 | 9.3 | 0.1 | 73.7 | 1.3 | 6.4 | 519 | 594 |

**Read the structure:** 0 and 2 are the same high-value customer differing only in recency (23.5 vs
73.3). 1 and 3 are the same low-value customer differing only in recency. It is a 2×2.

**Two business numbers not in the notebook that you should have ready** — I computed them and they
strengthen the case considerably:

- **Revenue concentration:** Clusters 0 and 2 together are 47% of customers but **88.7% of total
  spending** (42.1% + 46.6%). Clusters 1 and 3 are 53% of customers and 11.4% of revenue. That is
  the number that justifies the whole exercise. [Certain]
- **Campaign acceptance rate (accepted ≥1 campaign):** Cluster 0 = 43.8%, Cluster 2 = 38.1%,
  Cluster 1 = 19.3%, Cluster 3 = 10.9%. High-value customers are roughly 3–4× more responsive.
  [Certain]

### GRAPH 6 — Radar / spider chart of normalised cluster profiles

**What it shows:** each cluster as a polygon across nine axes, after min–max normalising each feature
across the four cluster means. Intended for at-a-glance shape comparison.

**What it produced:** Clusters 0 and 2 trace large polygons on income/spend/purchases/campaigns with
small values on web visits; Clusters 1 and 3 are the mirror image. Cluster 2 sits at 1.0 on recency
and tenure; Cluster 0 at 0.0 on recency.

**This chart is the weakest visualisation in the notebook and you should pre-empt criticism of it.**
Min–max normalising across only four values forces *every* axis to have exactly one cluster at 0.0
and one at 1.0, regardless of whether the underlying difference is large or trivial. Concrete
example: `Customer_Tenure_Days` ranges from 518 to 572 across clusters — a 54-day spread on
roughly 1.5 years, about 10%. On the radar chart that renders as a full 0.00 → 1.00 sweep, visually
identical in magnitude to the income difference, which is a genuine 2× gap (35k vs 70k). The chart
manufactures the appearance of differentiation on features that barely differ. Say so before you are
asked; the honest alternative is z-scores against the population mean, or simply the boxplots.

### GRAPH 7 — Boxplots: Total Spending and Income by cluster

**What they show:** the full distribution per cluster, not just the mean — median, interquartile
range, whiskers, outliers. This is the check on whether the profile table's means are honest.

**What they produced — Total Spending:**

| Cluster | Median | IQR | Min | Max |
|---|---|---|---|---|
| 0 | 1,053 | 725–1,428 | 62 | 2,126 |
| 1 | 68 | 38–167 | 13 | 601 |
| 2 | 1,120 | 833–1,505 | 312 | 2,126 |
| 3 | 81 | 43–211 | 13 | 577 |

**Excellent separation.** The high-value boxes (median ~1,050–1,120) and low-value boxes (median
~68–81) barely touch — the low-value 75th percentile (167, 211) sits far below the high-value 25th
percentile (725, 833). This is the single strongest piece of evidence that the segmentation is real.

**What they produced — Income:**

| Cluster | Median | IQR |
|---|---|---|
| 0 | 69,109 | 60,743–77,859 |
| 1 | 34,681 | 26,495–43,629 |
| 2 | 69,901 | 61,195–78,385 |
| 3 | 37,086 | 27,743–46,098 |

Clear separation between the value tiers, but with visible overlap in the whiskers — the low-value
clusters both reach up to the 94,461 cap. **Interpretation to have ready:** there exist high-income,
low-spending customers. They are affluent people who are not buying from this retailer, which is a
distinct and commercially interesting group that a 4-cluster solution cannot isolate. That is a
better argument for a higher K than anything in the current write-up.

### The recommendation engine

Two rule-based functions convert profiles into English:

- `describe_cluster()` — tags a cluster "high income," "high spender," "campaign-responsive,"
  "lapsing / inactive," "web-heavy browser" by comparing its mean against a baseline ±15%.
- `recommend_action()` — an if/elif chain mapping profile to campaign.

**Output:**

| Cluster | Profile | Action | n |
|---|---|---|---|
| 0 | high income, high spender, campaign-responsive | VIP / loyalty programme | 514 |
| 1 | low income, low spender, web-heavy browser, lower conversion | Discount / deals-led campaign | 570 |
| 2 | high income, high spender, campaign-responsive, **lapsing** | **VIP / loyalty programme** | 535 |
| 3 | low income, low spender, **lapsing**, web-heavy browser | Win-back campaign | 594 |

**Two defects here, both verifiable from the notebook's own output:**

1. **Clusters 0 and 2 receive identical actions.** `recommend_action()` tests
   `high_value and campaign_responsive` before it tests recency, so Cluster 2 returns early on the
   VIP branch and never reaches the win-back branch. This directly contradicts the Part D
   justification for choosing K = 4 (see Part 5 below). One reordered `if` fixes it.
2. **The baseline is wrong.** `overall_avg = cluster_profile.mean()` is the unweighted mean of four
   cluster means, not the population mean. With near-equal clusters the error is small here, but it
   is not what "average customer" means and would drift badly on imbalanced clusters. The fix is
   `df[profile_cols].mean()`.

---

# PART 4 — Architecture diagram (rubric item C, 5 marks)

The ASCII flowchart in section C.0 shows seven stages: embedded CSV → complete records (2,216) →
derived features + outlier capping → StandardScaler on 4 features → K-Means with elbow/silhouette
selection and PCA projection → cluster profiling and recommendation engine → outputs (per-segment
action + validity metrics).

It accurately matches the code, labels the row count at each stage, and names the transformation on
each arrow. The one inconsistency: it states 2,216 rows going into feature engineering but the
actual model runs on 2,213 after the `Age > 90` drop, which happens inside Step 2 and isn't shown.
Minor, but know the answer.

---

# PART 5 — How testing was done

## 5.1 The fundamental problem

Clustering is unsupervised: **there is no ground truth**. Nobody labelled these 2,213 customers with
their "true" segment, so you cannot compute accuracy, precision, recall or a confusion matrix. This
is worth stating explicitly in the viva because the rubric asks you to "compare expected/correct
results with those produced by your implementation," and the correct answer is that no external
correct result exists — which is why *internal* validity indices are used instead.

## 5.2 What was actually done: three internal validity indices

| Index | What it measures | Direction | K=4 result |
|---|---|---|---|
| **Silhouette** | Own-cluster distance vs nearest-other-cluster distance, per point, averaged | Higher better (−1 to 1) | **0.318** |
| **Davies–Bouldin** | Average worst-case ratio of within-cluster scatter to between-cluster separation | Lower better | **1.202** |
| **Calinski–Harabasz** | Ratio of between-cluster to within-cluster dispersion (a variance-ratio F-statistic) | Higher better | **1657.9** |

Using three rather than one is methodologically sound: they measure different things and are not
guaranteed to agree, so consensus is stronger evidence than any single number.

## 5.3 The K comparison table

| K | Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ |
|---|---|---|---|
| 3 | **0.355** | **0.998** | **1722.0** |
| 4 | 0.318 | 1.202 | 1657.9 |
| 5 | 0.318 | 1.072 | 1548.3 |
| 6 | 0.307 | 1.076 | 1466.0 |

**K = 3 wins on all three.** The submission acknowledges this openly, which is to its credit, then
argues for K = 4 on business grounds: at K = 3 the engaged/lapsing distinction within a value tier
collapses, and a lapsing high-value customer would get the same treatment as an engaged one, even
though win-back and loyalty are materially different interventions.

**The argument is sound in principle and defeated by the code in practice** — as shown above,
Clusters 0 and 2 *do* receive the same treatment. Be first to say this. The recovery is: the
justification is right, the implementation has an ordering bug, here is the one-line fix, and with
it applied the K = 4 choice does what the write-up claims.

## 5.4 What testing is missing — and what to say instead

Three gaps, all of which the rubric would reward closing:

1. **No stability testing.** The notebook lists it under "Possible Improvements" but doesn't do it.
   **I ran it:** re-fitting with random seeds 0, 1, 7 and 99 gives Adjusted Rand Index 0.976, 1.000,
   0.995 and 0.968 against the submitted labels. The partition is highly reproducible. This is a
   genuinely useful result — the clusters are *weakly separated but highly stable*, which is a
   precise and defensible characterisation. [Certain]
2. **No alternative algorithm actually implemented.** Part B compares three techniques in prose but
   only one is coded. Running Agglomerative Clustering or a Gaussian Mixture Model on the same
   scaled features and comparing silhouette would be about ten lines and would directly serve the
   testing rubric.
3. **No holdout or bootstrap.** Fitting on 80% and assigning the remaining 20% by nearest centroid
   would test whether segment definitions generalise.

## 5.5 The real test, and how to frame it

Internal indices measure geometry, not business value. A segmentation with a mediocre silhouette
that changes campaign response rates is worth more than a beautiful one that changes nothing. The
correct validation is an **A/B trial**: run the four recommended campaigns against a random half of
each segment, hold the other half on the generic campaign, and measure the difference in response
rate, revenue per customer and unsubscribe rate over one campaign cycle. The Conclusion already says
this — lead with it, because it demonstrates you understand the difference between a model metric
and a business outcome.

---

# PART 6 — EDI and Sustainability (rubric item G, 10 marks)

## 6.1 What the submission covers

**Bias and fairness.** The argument made is the proxy-discrimination one, and it is the right
argument: income and age features are strong clustering signals *precisely because* they correlate
with protected characteristics. A cluster that appears purely behavioural — "low spenders" — can
still function as a proxy for a protected group even though no protected attribute was ever fed to
the model. The stated implication is that a deployed system needs a proxy-discrimination audit, not
just a clean input feature list.

**This is well-founded in the data.** Concretely: `Income ↔ Kidhome` is −0.529 and
`Children ↔ Total_Spending` is −0.500, so Clusters 1 and 3 — the segments routed to discount-led
campaigns — systematically contain more households with children. Family status is a protected
characteristic in UK employment law and a sensitive one in consumer contexts. Being able to name a
specific correlation in your own data rather than speaking abstractly is what separates a good EDI
answer from a generic one. [Certain — computed]

There is also a real fairness question about the *treatment*: routing low-income customers
exclusively to discount and deals campaigns while reserving early access and premium bundles for
high-income customers is a form of price and opportunity discrimination. It may be commercially
rational and it is still worth naming.

**Privacy and GDPR.** Purchase history, income and web-visit counts are personal data under GDPR.
The section states that production deployment needs a documented lawful basis, a defined retention
period for `Recency`/`Dt_Customer`-derived features, and a mechanism for a customer to be removed
from a segment. The framing that "the model already trained on it" is not a recognised exemption is
correct and is a good line to have ready.

**Explainability.** Marketers acting on a segment must be able to say in plain language why a given
customer is in it. The rule-based `describe_cluster()` function is presented as a minimal step
toward this. Worth adding verbally: K-Means is *inherently* more explainable than most alternatives
because a centroid is literally a readable customer profile — this is an accessibility-of-decisions
argument, not just a technical one.

**Sustainability.** The argument made is that K-Means is computationally cheap relative to deep
learning, training completes in seconds, and the dominant lever is **retraining cadence** — monthly
batch retraining rather than retraining on every transaction.

## 6.2 What is missing — prepare these verbally

The rubric asks for six things. The submission covers four.

**Not covered — Accessibility and Inclusion.** Nothing in Part E addresses this. Have an answer:
- Customers with no digital footprint, low transaction counts or recent enrolment cluster poorly
  and risk being systematically under-served — a cold-start exclusion problem.
- The dataset's `Marital_Status` field contains junk values ("Absurd", "YOLO", "Alone") that the
  code silently collapses into "Single." That is an inclusion issue as much as a data-quality one:
  non-standard categories are being erased rather than represented.
- Campaign outputs themselves need accessible delivery — screen-reader-compatible emails, plain
  language, no offers that depend on visual-only content.

**Not covered — quantified computational and storage figures.** Part E is qualitative. Numbers to
have ready:
- Dataset in memory: **467 KB** for 2,216 rows × 26 columns (from `df.info()`).
- Model artefact: 4 centroids × 4 features = **16 floating-point numbers**. A few hundred bytes.
  This is the striking figure — the entire deployed model is smaller than one product photograph.
- Training: sub-second on CPU for 2,213 rows, no GPU at any point.
- Scaling estimate: at 1M customers, roughly 200 MB in memory; MiniBatchKMeans or fitting on a
  sample and assigning the remainder by nearest centroid keeps it tractable. [Likely — standard
  practice, not measured]

**Deployment suggestions to reduce impact:**
1. Monthly batch retraining rather than continuous — customer segments do not shift daily.
2. Store only the 16 centroid values and a single integer cluster label per customer; do not
   persist the full scaled feature matrix.
3. Enforce a data retention window (e.g. 24 months of transaction history) — this cuts storage and
   satisfies GDPR minimisation simultaneously.
4. Use CPU inference; assigning a customer to a segment is four distance calculations.
5. Note the comparison honestly: this pipeline's total lifetime compute is negligible against a
   single fine-tuning run of a transformer model. The sustainability story for classical ML is
   genuinely strong and you should say so plainly rather than manufacturing concern.

## 6.3 The formatting problem in Part E

The section ends with a literal bracketed confidence tag:
`` `[Likely; verify against your own runtime figures on the real dataset rather than asserting this without measurement.]` ``

This is unedited assistant output left in a submitted document, and it sits alongside two other
leftovers — `FINAL_K = 4  # <-- justify this choice ... before submitting` in Step 3, and "read
these against your own judgement before writing them up" in Step 4. Against an AI Usage Statement
that says all coding and analysis is your own, these will draw attention. If asked, the only
workable answer is a direct one: they are editing oversights, here is what the section is meant to
say. Do not improvise an explanation.

---

# PART 7 — Who else has used this dataset

This is a heavily-used teaching and benchmark dataset, which is worth acknowledging openly — it
means results are comparable, and it also means novelty claims should be modest.

**Peer-reviewed / preprint work:**

- **Customer Segmentation for Targeted Marketing: Exploring DBSCAN & K-Means** (Preprints.org, 2025)
  — the closest published analogue to your project. <cite index="7-1">It uses the same 2,240-customer dataset with RFM feature engineering, K-means validated via the Elbow Method, DBSCAN, and PCA for dimensionality reduction, and applies PERMANOVA to test whether the clusters differ significantly on key variables.</cite> Its ethics section is notably more developed than yours: <cite index="7-1">it addresses anonymisation (the Kaggle data is pre-scrubbed of PII), deliberate gender-blindness to prevent gendered stereotyping, and flagging rather than removing income outliers.</cite> **This is the single most useful comparison paper for you** — it does the DBSCAN comparison your Part B only describes, and it uses a formal statistical test (PERMANOVA) where you use internal indices alone.
- **Customer personality analysis and clustering for targeted marketing** (International Journal of
  Science and Research Archive, 2024). <cite index="2-1">Applies K-means to the same Kaggle dataset for targeted marketing segmentation.</cite> Treat this one critically: <cite index="6-1">it claims to identify Big Five personality traits — extroversion, openness, conscientiousness, agreeableness, neuroticism — from the data.</cite> The dataset contains no psychometric measurements whatsoever, so that claim is not supportable from these columns. Being able to say that is a strong viva moment.
- **Customer Personality Analysis Using Machine Learning with Explainable AI** (2025). <cite index="13-1">Same 2,240 instances and 29 features; notable methodological differences from yours — it imputes missing incomes with the mean rather than dropping those rows, one-hot encodes Marital Status and Education, and uses MinMaxScaler rather than StandardScaler.</cite> A good source for defending your own preprocessing choices by contrast.

**University coursework and capstones** — useful because it shows the level this is normally pitched at:

- **Carnegie Mellon, Department of Statistics** has run capstone projects on it in at least two
  cohorts (Fall 2021 and Spring 2024). <cite index="11-1">The Spring 2024 team reclassified expenditure into sector groupings — necessities such as fish, meat and fruit forming a primary sector — to study drivers of purchasing decisions.</cite>
- **UMass Dartmouth** has a published student report applying unsupervised learning to it.

**Widely-circulated practitioner analyses:**

- A Medium walkthrough by Prasan N H that reaches a different conclusion from yours: <cite index="1-1">it uses K-Medoids at k=3 after PCA-based dimension reduction, having engineered comparable features (Days_As_Customer, Total_Purchases, Total_Amount_Spent, Age).</cite> K-Medoids is worth knowing as a robustness alternative — it uses actual data points as centres rather than means, so it is far less sensitive to outliers, which would have removed your need for the capping step.
- An analysis by Andhika Widyadwatmaja comparing <cite index="4-1">K-means against agglomerative clustering</cite> on the same data.
- <cite index="15-1">A widely-forked GitHub K-Means analysis reporting substantive findings: customers living alone spend more than those with partners, customers with no children spend more, the 40–60 age group is both the largest and the highest-spending, most incomes fall between 25,000 and 85,000, and wine and meat are the dominant categories.</cite> Every one of those is consistent with what your correlation heatmap and cluster profiles found, which is a good external validity check to mention.
- GitHub hosts a `customer-personality-analysis` topic with dozens of repositories, including ones
  combining K-Means with XGBoost and SHAP for explainable segment membership.

**How to use this in the viva.** Do not present the project as novel — it is a well-trodden dataset
and pretending otherwise is easy to expose. Present it as a *replication with an explicit business
constraint*: the differentiator in your work is not the algorithm but the argument that K should be
chosen against campaign capacity rather than purely against a validity index, plus the rule-based
translation from cluster to action. Then concede that the published DBSCAN comparison and the
PERMANOVA validation are things your submission describes but does not do.

---

# PART 8 — Honest summary of strengths and weaknesses

**Genuine strengths:**
- Fully self-contained and reproducible — embedded data, fixed random seed, no external
  dependencies. ARI 0.97–1.00 across seeds confirms reproducibility empirically.
- Preprocessing is defensible at every step, with reasons given rather than assumed.
- Balanced clusters (23–27% each), all large enough to campaign against and A/B test.
- Excellent separation on spending — the boxplots are the strongest evidence in the notebook.
- Three validity indices rather than one, and the inconvenient result (K = 3 wins, K = 2 wins on
  silhouette) is reported rather than buried. That intellectual honesty is worth defending.
- The pipeline ends in a decision, not a chart. Most student clustering projects stop at the scatter
  plot.

**Genuine weaknesses, ranked:**
1. The recommendation engine's branch ordering nullifies the stated justification for K = 4.
2. Recency is uniformly distributed 0–99 and near-zero correlated with everything, so one of the two
   segmentation axes is a slice through noise.
3. The remaining three features correlate 0.69–0.82, collapsing to a single value dimension — so the
   whole result is reproducible by hand with two median splits, and K-Means is not clearly earning
   its place on this feature set.
4. Silhouette 0.318 means substantial overlap; these are useful partitions, not discovered
   populations.
5. Campaign responsiveness drives the routing rules but was excluded from the clustering.
6. `Complain` dropped as "unused" despite being the most obvious churn signal available and one of
   the few genuinely uncorrelated behavioural columns.
7. Missing-value map plotted after `dropna()`, rendering it blank.
8. Radar chart min–max normalisation manufactures apparent differentiation on features that barely
   differ (tenure varies 10% and renders as full-scale).
9. Part E omits accessibility/inclusion and gives no quantified resource figures.
10. Editing scaffolding left in three places in the submitted notebook.
