# Viva Prep — AI-Powered Customer Segmentation (K-Means)

**Submission:** CW1_w2053223_SaadMazhar.ipynb · Domain: Marketing Analytics · Technique: K-Means (unsupervised)

---

## 1. The 60-second project summary

Take the Kaggle *Customer Personality Analysis* dataset (2,240 customers, 29 columns), clean it,
compress the raw transaction columns into RFM-style features, cluster customers into 4 groups with
K-Means, then map each group to a marketing action via a rule-based engine.

**Pipeline in one line:** embedded CSV string → drop constant/unused columns → drop 24 rows with
missing `Income` → engineer `Age`, `Total_Spending`, `Total_Purchases`, `Total_Campaigns_Accepted`,
`Children`, `Customer_Tenure_Days` → cap `Income`/`Total_Spending` at 1st/99th percentile → select
4 features (`Income`, `Total_Spending`, `Recency`, `Total_Purchases`) → `StandardScaler` →
K-Means (K=2–10 swept, K=4 chosen) → PCA 2D plot → cluster profiles → recommendation rules →
validity metrics. [Certain — read directly from the notebook]

---

## 2. Numbers you must be able to recite cold

| Item | Value |
|---|---|
| Raw rows / columns | 2,240 × 29 |
| After dropping missing `Income` | 2,216 (24 dropped, ~1%) |
| After dropping `Age > 90` | 2,213 (3 dropped) |
| Clustering features | 4 (Income, Total_Spending, Recency, Total_Purchases) |
| Final K | 4 |
| Cluster sizes | 514 / 570 / 535 / 594 |
| Silhouette (K=4) | 0.318 |
| Davies–Bouldin (K=4) | 1.202 (lower better) |
| Calinski–Harabasz (K=4) | 1657.9 (higher better) |
| Best silhouette overall | K=2, 0.433 |
| Best of K=3–6 | K=3 (0.355 / 0.998 / 1722.0) |

Cluster profile means:

| Cluster | Income | Spend | Purchases | Recency | Campaigns | n |
|---|---|---|---|---|---|---|
| 0 | 68,807 | 1,096 | 21.6 | 23.5 | 0.8 | 514 |
| 1 | 35,102 | 121 | 8.5 | 23.5 | 0.2 | 570 |
| 2 | 69,614 | 1,166 | 21.5 | 73.3 | 0.7 | 535 |
| 3 | 36,909 | 140 | 9.3 | 73.7 | 0.1 | 594 |

**Read the structure honestly:** this is a 2×2 grid. Axis 1 = value (high/low income+spend+purchases).
Axis 2 = Recency (recent ~23 days vs lapsed ~73 days). Nothing else separates the clusters. [Certain]

---

## 3. The four questions most likely to sink you

These are ordered by how badly they hurt. Prepare answers before anything else.

### Q1. "Your recommendation engine gives Cluster 0 and Cluster 2 the same action. Why?"

**This is the single biggest defect in the submission.** [Certain — verified against the notebook's own
output table]

Your Part D discussion argues K=4 was kept *over* the statistically better K=3 specifically because
K=4 separates lapsing high-value customers from engaged ones, so the lapsed group can get a win-back
offer instead of loyalty rewards. But `recommend_action()` tests `high_value and campaign_responsive`
**before** it tests recency. Cluster 2 (lapsed, 73 days) satisfies the first branch and returns early.
Output: both 0 and 2 get *"VIP / loyalty programme."*

So the code contradicts the written justification for the central design decision. The 4-cluster
solution costs you measurable cluster quality and delivers three distinct actions instead of four.

**Do not bluff this.** Have the fix ready: move the recency check above the high-value branch, or make
it a compound condition — `if high_value and lapsing: return "high-value win-back"`. Say plainly that
the ordering is a logic bug, that you found it, and that the corrected version produces four distinct
actions and vindicates the K=4 choice. A marker will respect the catch far more than a defence.

### Q2. "What does Recency actually contain?"

Recency in this dataset is **uniformly distributed 0–99** — roughly 220 customers in every decile —
and correlates with the other three features at r ≈ 0.01. [Certain — I re-ran the distribution]

PCA confirms it: PC1 (63% variance) loads almost equally on Income/Spend/Purchases; PC2 (25%) loads
**1.000 on Recency and ~0.00 on everything else.** [Certain]

The consequence is uncomfortable. The second axis of your segmentation is a variable with no cluster
structure at all — K-Means simply bisects a uniform distribution at its midpoint, which is exactly why
the two recency groups land at 23.5 and 73.3 (the midpoints of the two halves). The "engaged vs
lapsing" distinction that justifies K=4 is a slice through noise, not a discovered group.

**Defensible answer:** concede the distributional point immediately, then argue on operational grounds
— even if recency has no natural break, a 73-day-since-purchase customer genuinely does warrant
different contact than a 23-day one, so the split is business-meaningful even when it isn't
statistically natural. Add that in production you would validate it against real recency data rather
than this dataset's apparently synthetic column. Conceding is stronger than defending here.

### Q3. "Why K-Means at all? Two thresholds would give you the same four groups."

Your other three features correlate at 0.69–0.82 with each other. [Certain] They collapse to one
latent dimension: customer value. Add uncorrelated Recency and your entire 4-D feature space is
effectively 1 signal axis + 1 noise axis. Any analyst could hand-draw this 2×2 with a median split on
spend and a median split on recency, in an afternoon, with no ML.

**Honest answer:** for *this* feature set K-Means is largely rediscovering a rule an analyst could
specify. Its value is that thresholds are derived rather than assumed, it extends without redesign
when features are added, and it re-fits automatically as behaviour drifts. Then state what would make
it earn its place: adding weakly-correlated features (channel mix, category preference share, campaign
timing) so the space has genuine multi-dimensional structure. Claiming K-Means was necessary here
would be the wrong move.

### Q4. "Silhouette 0.318 is weak. Do these clusters exist?"

0.318 indicates substantial overlap; K=2 scores highest at 0.433 and the score falls monotonically as
K rises, which is the signature of a dataset with no strong natural cluster count. [Certain]

**Answer:** the correct claim is that the segmentation is a *useful partition*, not a discovery of
naturally separated populations. Real behavioural data is usually continuous. Two supporting points
you can offer that aren't in the notebook:

- **Stability:** re-fitting with random seeds 0, 1, 7, 99 gives Adjusted Rand Index 0.97–1.00 against
  the submitted labels. The partition is highly reproducible even though it is not well-separated.
  [Certain — I ran this]
- Three independent indices were used rather than one, so the K choice isn't resting on a single
  metric's quirks.

Note the "Possible Improvements" section lists stability testing as future work — you can say you
have since run it and give the ARI figures.

---

## 4. Second-tier questions

**"What is your baseline for calling a cluster 'high income'?"**
`overall_avg = cluster_profile.mean()` — the unweighted mean of the four *cluster means*, not the
population mean. With near-equal cluster sizes (514–594) the difference is small here, but it is not
the population average and would drift badly on imbalanced clusters. [Certain] Acknowledge it; the fix
is `df[profile_cols].mean()`.

**"Why hard-code 2015-01-01 for Age and Tenure?"**
The dataset was collected ~2012–2014 (`Dt_Customer` runs from 2012-07-30). Using `Timestamp.now()`
would inflate every age by a decade and make tenure meaningless. Fixing the reference date to the
collection era is the correct choice — say so confidently. [Certain]

**"Why cap outliers instead of dropping them?"**
K-Means minimises squared distance, so extreme incomes drag centroids. Capping at the 1st/99th
percentile limits that influence while preserving sample size. Winsorising is defensible; be ready to
note you did *not* cap Recency or Total_Purchases, and Total_Purchases reaches +3.8 SD. [Certain]

**"Why drop the 24 missing incomes rather than impute?"**
~1% of rows; median imputation on the strongest clustering feature would inject artificial mass at the
centre. Dropping is cleaner at this rate. Be ready to concede that it assumes missingness is random,
which was not tested. [Certain — the notebook does not test MCAR]

**"Why PCA?"**
Visualisation only — the model fits on all four scaled features, not on PCA components. PC1+PC2 retain
79% of variance. [Certain] Do not let anyone believe you clustered in PCA space.

**"How would you test this for real?"**
There is no ground truth, so internal validity indices are the only offline test. The real test is an
A/B trial: run the four recommended campaigns against segment members vs a randomised control and
measure response-rate lift. Your Conclusion already says this — lead with it.

**"How does this scale?"**
K-Means is O(n·k·i); 2,213 rows fits in well under a second. For millions of customers, MiniBatchKMeans
or fitting on a sample and assigning the rest by nearest centroid. [Likely — standard practice, not
measured by you]

---

## 5. Rubric mapping — where marks are, and where they leak

| Rubric section | Marks | Where in notebook | Risk |
|---|---|---|---|
| A. Domain review, 500–1000 words | 10 | Part A (858 words) | Within range. Slightly generic; few concrete numeric results cited from literature. |
| B. 3 techniques, 200–400 words each | 12 | B.1 K-Means (211), B.2 Hierarchical (216), B.3 DBSCAN (220) | All three clear the 200-word floor, but only just. Each covers data/setup/runtime/output as required. |
| C. Architecture diagram | 5 | C.0 ASCII flowchart | Complete and accurate; matches the code. Solid marks. |
| D. Input data + format + preprocessing | 8 | C.0 continued + Steps 1–2 | Strong. 2,240 rows >> 1,000 threshold, data embedded so it runs offline, 5 preprocessing steps documented. |
| E. Notebook prototype (data 5 / code 5 / technique 20 / output 5) | 35 | Steps 1–4 | Technique implementation is sound. **Code quality is where you lose marks** — see §6. |
| F. Testing, interpretation, strengths/limits | 20 | Part D (488 words) | Three indices + K sweep is good. No stability test, no alternative algorithm actually implemented, and the K=4 justification is undercut by the Q1 bug. |
| G. EDI + Sustainability | 10 | Part E (252 words) | **Weakest section.** See §6. |

---

## 6. Known weak points — say them before the examiner does

1. **Leftover scaffolding in the submitted notebook.** Three instances:
   - `FINAL_K = 4  # <-- justify this choice in the Testing & Evaluation discussion before submitting`
   - Step 4 markdown: *"read these against your own judgement before writing them up"*
   - Part E ends with a literal confidence tag: `` `[Likely; verify against your own runtime figures on the real dataset rather than asserting this without measurement.]` ``

   These are instructions-to-self left in a final submission. The third in particular reads as
   unedited AI-assistant output, which sits awkwardly against an AI Usage Statement claiming all
   coding and analysis is your own. [Certain — all three are in the file] If asked, the only workable
   answer is a straight one: they are editing oversights, here is what the section should say. Do not
   improvise a story about them.

2. **Part E is thin against its own rubric.** The rubric asks for bias/fairness, accessibility and
   inclusion, ethical implications (5) plus computational resource analysis, data storage
   requirements, and deployment reduction suggestions (5). The notebook covers bias, privacy,
   explainability and a qualitative sustainability paragraph — but has **no accessibility/inclusion
   content and no quantitative resource or storage figures**. Have numbers ready verbally: dataset is
   ~467 KB in memory, model artefact is 4 centroids × 4 features (trivial), training is sub-second,
   monthly batch retraining vs continuous is the main lever.

3. **The `Complain` column is dropped as "unused"** — but complaint history is arguably the most
   relevant churn signal available and would have added a genuinely uncorrelated behavioural
   dimension. Weak justification. Concede it as a feature-selection miss.

4. **Campaign responsiveness is used in the recommendation rules but not in the clustering.** So the
   clusters aren't optimised to separate responsive customers, yet the actions depend on it. This is
   internally inconsistent — either cluster on it or don't route on it. [Certain]

5. **No comparison against an actually-implemented alternative.** Part B compares three techniques in
   prose; only one is coded. Running Hierarchical or GMM on the same features and comparing silhouette
   would have been ~10 lines and directly serves the testing rubric.

---

## 7. If you have time before the viva: three cheap fixes

Each is small, each closes a question above, and each is honest to disclose as post-submission work.

1. **Reorder `recommend_action`** so lapsing is checked before the VIP branch. Kills Q1 and validates
   the entire K=4 argument.
2. **Add a stability cell** — loop over 5 random seeds, report Adjusted Rand Index. Directly answers
   Q4 and ticks the "possible improvements" box you wrote yourself.
3. **Print PCA explained variance** and plot the Recency histogram. Pre-empts Q2 by showing you know
   what that feature contains rather than being told.

---

## 8. One-paragraph answer if asked "what did you learn?"

That clustering quality metrics and business usefulness pull in different directions, and that the
gap has to be argued rather than assumed. K=3 was statistically better on all three indices; K=4 was
chosen because it separates lapsing from engaged customers within each value tier — but that argument
is only worth making if the downstream actions actually differ, and in the submitted version they
don't for the high-value tier. The wider lesson is that a segmentation is only as good as the decision
it changes, and the honest test of it is a live A/B trial, not a silhouette score.
