# 35–40 Minute Video Script — AI-Powered Customer Segmentation (K-Means)
**Student:** Saad Mazhar (W2052332 / 20221804) | **For:** CW1 video presentation

---

## HOW TO USE THIS SCRIPT
This version is written for **technical depth** — every method is explained with *why it
was chosen*, *how it works mechanically (including the actual math)*, and *what its
numbers mean*. It runs longer than a typical script (~35–40 min spoken) because depth was
prioritised over brevity, per your request. Bracketed cues like **[SHOW: ...]** tell you
what should be on screen. Read the math lines slowly — a marker will pause a recording to
re-check a formula or a number far more than they'll re-check a sentence of prose.

A note before you record: **K-Means is unsupervised** — there is no "ground truth" label
saying which segment a customer really belongs to, so classification **accuracy** (percent
correctly predicted) is not a metric that exists for this model. This script explains that
distinction explicitly in Part D, then goes deep on the three metrics that stand in for it
— Silhouette, Davies-Bouldin, Calinski-Harabasz — and on concrete, technical ways to improve
those numbers. Don't gloss past that distinction; it's one of the more sophisticated points
you can make on camera.

---

## 0:00–1:00 — Title & Framing (60 sec)

**[SHOW: Title slide/markdown cell — project title, your name, student ID, module leader]**

"Hi, I'm Saad Mazhar, student ID W2052332. This is my coursework submission on AI-powered
customer segmentation for personalised marketing, using K-Means clustering on the Customer
Personality Analysis dataset.

Over the next thirty-five to forty minutes I'll go deep on six things: the literature
behind this problem, the mathematics and trade-offs behind three competing clustering
algorithms and why I selected K-Means, exactly how the pipeline transforms raw data into
cluster assignments — including the actual formulas involved at each step — how I evaluated
the model quantitatively, an honest discussion of *why 'accuracy' isn't the right word for
an unsupervised model and what to use instead*, and how that evaluation could realistically
be improved. Then I'll close with the ethical and sustainability considerations of
deploying something like this in a real business. Let's get into it."

---

## 1:00–3:30 — Executive Summary & Motivation (2 min 30 sec)

**[SHOW: Executive Summary markdown cell]**

"So — why does this problem matter? Businesses used to run one marketing campaign and send
it to everyone. That's inefficient: a high-income customer who buys constantly and a
customer who hasn't purchased in over two months don't respond to the same email, and
treating them identically wastes marketing spend on both ends.

The core idea here is **unsupervised learning** — specifically K-Means clustering — to find
natural groups in customer data *without* being told in advance what those groups are. This
is a precise technical distinction worth stating clearly: in **supervised learning**, every
training row has a known target — think 'this email was opened: yes/no' — and the model
learns a function that maps inputs to that known output, so you can later measure accuracy
by comparing predictions to the true answer. In **unsupervised learning**, there is no
target column at all. No 'this customer is a VIP' tag exists anywhere in the raw 29 columns
of this dataset. The algorithm only sees behaviour — income, spending, recency, purchase
counts — and has to discover structure in that behaviour on its own. That single distinction
is the reason the word 'accuracy' needs to be handled carefully later in this video, and I
flag it now so it isn't a surprise in Part D.

The full pipeline follows a standard ML workflow: data collection, cleaning, feature
engineering, scaling, clustering, and then — critically — evaluation. I didn't just run
K-Means once and call it done. I validated the clustering quantitatively using three
separate internal-validation metrics: Silhouette Score, Davies-Bouldin Index, and
Calinski-Harabasz Index — I'll define exactly what each one computes, mathematically, and
show you the actual numbers, not just claim it 'worked.'

And the endpoint isn't just 'here are four clusters' — it's translating each cluster into
an actual marketing action: who gets a loyalty programme, who gets a win-back email, who
gets a discount. That's the business value.

The dataset is the Kaggle 'Customer Personality Analysis' dataset — about 2,240 real
customer records with demographics, purchase history across six product categories, and
responses to five past marketing campaigns. I'll go deep into the actual columns in a
moment."

---

## 3:30–10:00 — Literature Review (6 min 30 sec)

**[SHOW: Part A markdown cell, scroll through as you talk]**

"Before writing any code, I grounded this in existing literature — six areas.

**First, AI in marketing broadly.** Davenport, Guha, Grewal and Bressgott, 2020, argue AI
has fundamentally changed marketing because it lets businesses move past broad demographic
assumptions — 'people aged 25 to 34 like X' — into analysing actual behavioural data at
scale. Huang and Rust, 2021, make an important point I want to stress: AI here doesn't
*replace* the marketer, it supports their decision-making. My rule-based recommendation
engine later in this project is a good example — it produces a suggested action, but a
human marketer still reads and validates it.

**Second, customer segmentation itself.** Traditional segmentation used static demographic
buckets — age, gender, income bracket. Wedel and Kamakura, 2012, point out this fails to
capture how complex real purchasing behaviour actually is — two customers with identical
income can spend completely differently. That's exactly what motivates using purchasing
behaviour, not just demographics, as clustering features here.

**Third, the machine learning techniques used for this.** Xu and Wunsch, 2009, and Jain,
2010, frame segmentation as fundamentally an *unsupervised* problem — we don't have
ground-truth labels for 'which segment is this customer in,' so we can't use supervised
classification, and by extension we can't compute classification accuracy either. This is
precisely why K-Means, Hierarchical Clustering, and DBSCAN are the candidate techniques —
I'll compare all three in the next section in real mathematical depth.

**Fourth, personalisation and marketing automation.** Chaffey and Ellis-Chadwick, 2022,
describe how modern CRM systems increasingly expect automated, segment-driven
personalisation — automatically triggered emails, loyalty rewards, tailored offers — rather
than a marketer manually deciding who gets what. My recommendation engine is a small-scale
version of exactly that automation layer.

**Fifth — and I want to spend real time here because it directly shapes decisions later in
this project — ethics and challenges.** Two problems specifically. One: choosing the number
of clusters, K, is genuinely ambiguous — different choices produce materially different
business strategies, and I hit this exact issue in my own results, which I'll be honest
about in the evaluation section rather than glossing over. Two — and this is the more
serious one — features like income, location, or spending patterns can act as *proxies*
for protected characteristics even when you never explicitly feed in age, gender, or
ethnicity. A cluster that looks purely behavioural, like 'low spenders,' can still
correlate strongly with a protected group. The European Parliament and Council's GDPR,
2016, is directly relevant here because this pipeline processes income and purchase
history, which counts as personal data requiring a documented lawful basis to process.
I come back to this properly in Part E.

**Sixth, future trends** — real-time, continuously-updating segmentation rather than a
static one-off snapshot like this project, plus more explainable AI so marketers can
justify *why* a customer landed in a given segment, and generative AI eventually producing
the actual campaign content per segment automatically. That's the trajectory this kind of
project sits on."

---

## 10:00–19:00 — Part B: Comparing Three Clustering Techniques, With the Actual Math (9 min)

**[SHOW: Part B markdown cell, keep the comparison table visible for most of this]**

"Now the technique comparison — this is where I justify *why* K-Means specifically, and I'm
going to go through the mechanics of all three properly, because the *reason* to choose one
over another only makes sense once you understand what each one is actually optimising.

### K-Means — the objective function and the algorithm loop

K-Means is a **centroid-based, partitional** algorithm. Formally, it tries to choose K
centroids, mu-1 through mu-K, that minimise the **within-cluster sum of squares** — for
every point x, take the squared Euclidean distance to its assigned centroid, and sum that
across every point in every cluster. That single quantity is called **inertia**, and it's
exactly what `kmeans.inertia_` returns in scikit-learn, and exactly what feeds the elbow
plot later.

Minimising that objective exactly is NP-hard, so in practice we use **Lloyd's algorithm**, a
fast approximation that provably converges to *a* local minimum, though not necessarily the
global one — which is precisely why initialisation matters, and why I used `n_init=10`,
which I'll come back to. Lloyd's algorithm is a two-step loop, repeated until convergence:

1. **Assignment step** — every data point is assigned to its nearest centroid, measured by
   Euclidean distance.
2. **Update step** — each centroid is recalculated as the mean of every point currently
   assigned to it — that's literally where the name 'K-Means' comes from.

You repeat those two steps until centroids stop moving, or move by less than a tolerance
threshold — that's convergence. Scikit-learn also defaults to **k-means++ initialisation**
rather than pure random placement, which deliberately spreads the initial centroids apart
using a probability proportional to squared distance from existing centroids — this alone
measurably reduces the chance of a bad local minimum before `n_init` restarts even come into
play.

Its **complexity** is O of n times K times i times d — n points, K clusters, i iterations
until convergence, d dimensions — which is *linear* in the number of data points. That's the
key practical advantage: it scales comfortably to the thousands-of-customers range you'd see
in a real CRM system, and would still be tractable at millions of rows.

Its **strengths**: computationally efficient as just shown; the output is directly
interpretable, because each centroid literally *is* the coordinates of an 'average' customer
in that segment, expressed in the same units as the input features; and it has essentially
one hyperparameter, K, which — importantly — can be tuned *quantitatively*, using the elbow
method and silhouette score, rather than guessed, which I'll demonstrate properly in Part C.

Its **weaknesses**, stated precisely rather than hand-wavily: K has to be chosen in advance,
not discovered by the algorithm; the algorithm implicitly assumes clusters are roughly
spherical (technically, isotropic — equal variance in every direction) and similarly sized,
because minimising squared Euclidean distance to a single centroid is only the *statistically
correct* thing to do if the cluster is genuinely shaped like a sphere around that centroid —
real customer behaviour doesn't always respect that geometry; every single point is forced
into some cluster, so unlike DBSCAN it has no concept of 'noise' or outlier; and it's
sensitive to feature scaling, because Euclidean distance is dominated by whichever feature
has the largest numeric range, and sensitive to initial centroid placement, because Lloyd's
algorithm only guarantees a *local* minimum. That last point is exactly why I used
`n_init=10` restarts in my code — run the whole Lloyd's-algorithm loop ten times from ten
different k-means++ initialisations, and keep only the run with the lowest final inertia,
which meaningfully reduces the risk of settling on a bad local minimum.

### Hierarchical Clustering — the dendrogram and linkage criteria

**Hierarchical Clustering** takes a completely different approach: instead of committing to
K centroids upfront, agglomerative — bottom-up — hierarchical clustering starts with every
point as its own cluster, then **repeatedly merges the two closest clusters**, one merge at a
time, until everything is in a single cluster. That merge sequence is recorded as a
**dendrogram** — a tree diagram where the height of each join represents how dissimilar the
two merged clusters were. Han, Kamber and Pei, 2012, is the reference here.

The critical modelling decision is the **linkage criterion** — the rule for measuring
distance *between two clusters*, not between two points, since after the first merge a
cluster can contain many points. Single linkage uses the minimum pairwise distance between
the two clusters, which can produce long, straggly 'chained' clusters. Complete linkage uses
the maximum pairwise distance, favouring compact clusters. Average linkage takes the mean
of all pairwise distances. Ward's method — often the default choice — merges whichever pair
of clusters causes the *smallest increase* in total within-cluster variance, which makes it
conceptually closest to K-Means' own objective.

Its big advantage over K-Means is that you don't have to commit to K in advance — you cut
the dendrogram at whatever height you decide afterwards, and the tree itself is a genuinely
useful exploratory tool — it could reveal, for example, that my 'high-value' customers
actually split further into wine-focused versus general-goods high spenders. But the catch
is scalability: computing and storing the full pairwise distance matrix is O of n-squared in
both time and memory, which becomes impractical well before typical CRM row counts — at
2,213 rows that's roughly 2.4 million pairwise distances, which is fine, but a CRM system
with a few hundred thousand customers would not be. And the choice of linkage method is
itself a non-trivial modelling decision with no single correct answer.

### DBSCAN — density, epsilon, and core points

**DBSCAN** — Density-Based Spatial Clustering of Applications with Noise — takes a third,
density-based approach, controlled by two parameters: **epsilon**, the radius of a
neighbourhood around each point, and **min_samples**, the minimum number of points required
inside that radius for a point to count as a **core point**. Every point is then classified
as one of three types: a **core point**, if at least `min_samples` other points fall within
`epsilon` of it; a **border point**, if it falls within `epsilon` of a core point but doesn't
itself have enough neighbours to qualify as core; or **noise**, labelled minus-one, if
neither condition holds. Clusters are then formed by chaining together core points that are
within epsilon of each other, along with their border points.

Its real strength is that it doesn't force every point into a cluster — points in
low-density regions get explicitly labelled as noise instead of being crammed into the
nearest group, which is genuinely useful for flagging anomalous customers, maybe data-entry
errors or unusual behaviour, as a complement to segmentation. It can also find
arbitrarily-shaped clusters, not just convex, spherical ones, because it grows clusters
through connected density rather than distance-to-a-single-centroid.

But it's very sensitive to those two parameters, which are difficult to tune reliably on
customer data — there's no clean equivalent of the elbow method for epsilon, though a
k-distance plot can help. And — this is the specific problem for *this* dataset — DBSCAN
struggles when different real clusters have noticeably different densities, because a
single global epsilon can't simultaneously suit a dense cluster and a sparse one. That's
realistic here: high-value customers are naturally a sparser, more spread-out group in
feature space than average customers, so one epsilon that correctly captures the average
group would likely swallow the high-value group into noise, or vice versa. It also doesn't
guarantee a fixed number of clusters, which sounds like a strength in the abstract, but is
actually a *business* weakness in this specific context.

### So why K-Means, concretely?

The one-line justification: marketing segmentation almost always needs a small, fixed,
explainable number of groups, because the real business constraint isn't 'however many
natural clusters exist in the data' — it's 'how many distinct campaigns can we realistically
design and staff.' K-Means' centroid output — a small set of interpretable 'average
customer' profiles — maps directly onto that constraint. Its main weakness, sensitivity to
K, is the *one* thing that can be addressed directly and quantitatively — which is exactly
what Part D of this project does using the elbow method and silhouette score together.
DBSCAN's variable, data-driven cluster count and Hierarchical Clustering's O of n-squared
scalability are both a worse fit for that specific business constraint — not because
they're worse algorithms in general, but because they're worse *for this use case*."

---

## 19:00–30:00 — Part C: Implementation Walkthrough, Step by Step (11 min)

**[SHOW: switch to the actual notebook now — architecture diagram first]**

"Now let's go into the actual implementation, and I'll explain the technical reasoning
behind every individual decision, not just describe what the code does.

### Step 1 — Loading and cleaning the data (2 min)

**[SHOW: Step 1 code cell + output]**

The source is the Kaggle dataset `imakash3011/customer-personality-analysis` — a
**tab-separated** file, `marketing_campaign.csv`, not comma-separated, which matters
technically: `pandas.read_csv` defaults to a comma delimiter, so calling it without
`sep='\t'` doesn't error out — it just parses the entire row as one single garbage column,
since there are no commas to split on. That's a silent failure mode, not a crash, which
makes it worse: you'd see 2,240 rows and 1 column and might not immediately realise why.

It's 2,240 rows, one per customer, 29 columns, split into three groups: demographics — year
of birth, education, marital status, income, number of kids and teens at home; transaction
history — six spend columns, `MntWines` through `MntGoldProds`, plus purchase-channel
counts across deals, web, catalogue, and store; and campaign-response flags — five past
campaigns plus a `Response` column for the most recent one.

One specific technical gotcha: `Dt_Customer`, the enrolment date, is stored as `DD-MM-YYYY`
text. Pandas' `read_csv` does *not* auto-detect this format — its default date inference
assumes month-first, US-style ordering. If you parse it without `dayfirst=True`, any date
where the day number is 12 or below gets silently misread — day and month get swapped — and
the failure is invisible because both interpretations produce a *valid* date, just the wrong
one. That's the dangerous kind of bug: it doesn't throw an error, it quietly corrupts every
downstream tenure calculation for a subset of rows.

The code has a three-tier fallback: try the local CSV first, then `kagglehub` if it's not
there, and only as a last resort, a synthetic dataset generator that mimics the same schema
— purely so the notebook doesn't crash in an environment without the file. To be clear:
every result I'm about to show you comes from the *real* Kaggle data, loaded from the local
file — the console output literally confirms `Loaded local file: (2240, 29)`.

For missing values: `Income` had 24 missing values out of 2,240 rows, about 1.07 percent. I
made a deliberate technical choice to **drop those rows** rather than **impute** them —
worth being precise about the trade-off: imputing, say with the median, would preserve
sample size but inject 24 synthetic, statistically-guessed values directly into a feature
that gets used for clustering, silhouette scoring, and every downstream business decision.
Given the missing proportion was small — about 1 percent — dropping was the lower-risk
choice: every remaining row is a real, complete customer record. That leaves 2,216 rows.

### Step 2 — Feature engineering and preprocessing, with the actual math (3 min)

**[SHOW: Step 2 code cell + correlation heatmap + distribution plots]**

Step two engineers **RFM-style features** — Recency, Frequency, Monetary — the classic
customer-value framework, on top of the raw columns. Specifically: `Total_Spending` is the
sum of all six product-category spend columns; `Total_Purchases` sums the four
purchase-channel counts; `Total_Campaigns_Accepted` sums all five past campaign flags plus
the latest `Response`; `Children` is kids plus teens at home; `Customer_Tenure_Days` is days
since enrolment relative to a fixed reference date; and `Age` is derived as
reference-year minus year-of-birth.

I also dropped a handful of rows with implausible ages above 90 — likely data entry errors,
like someone with a birth year of 1900, which would make them over 120 years old at the
reference date. After that filter, 2,213 rows remain — three more dropped on top of the 24
from missing income.

For outliers in `Income` and `Total_Spending`, I made a specific technical choice: **cap at
the 1st and 99th percentile** rather than drop those rows entirely — a technique called
**winsorization**. The reasoning connects directly back to Part B's math: K-Means minimises
*squared* Euclidean distance to a centroid, and squaring means a point that's twice as far
away contributes four times the penalty — so a handful of extreme high-income customers can
pull a centroid's position substantially away from where the bulk of that cluster's members
actually sit. Winsorizing clips those extreme values down to the 1st/99th percentile
boundary, which limits their leverage on the centroid's position while *preserving sample
size*, which outright dropping would sacrifice.

The correlation heatmap confirms something important for feature selection: income and total
spending are strongly positively correlated, which is expected and is actually the RFM logic
at work — richer customers spend more in absolute terms.

For the actual clustering step, I deliberately used a **small feature set** — just four
features: `Income`, `Total_Spending`, `Recency`, and `Total_Purchases` — rather than every
engineered column. The technical reasoning is the **curse of dimensionality** as it applies
specifically to Euclidean-distance-based clustering: K-Means computes distance across *all*
selected features simultaneously, and as you add more dimensions — especially ones that are
correlated with each other or just noisy — the relative contrast between 'near' and 'far'
points shrinks, distances become more uniform, and clusters become both harder to separate
and harder for a human to interpret afterwards. This is also a direct trade-off I'm flagging
honestly: I lose potentially relevant signals like campaign responsiveness from the
*clustering step itself*, though I do bring them back in for profiling in step 4 — clustering
features and profiling features don't have to be the same set, and deliberately keeping them
separate is itself a design decision.

Every one of those four features then gets passed through **`StandardScaler`** —
**standardisation**, computed as z equals x minus the feature's mean, divided by the
feature's standard deviation — giving every feature a mean of zero and a standard deviation
of one. This step is not optional, for a precise reason: income is measured in tens of
thousands of dollars, while recency is measured in single or double-digit days. Without
scaling, Euclidean distance — which sums squared differences across features — would be
almost entirely dominated by whichever feature has the largest raw numeric range, which
would be income, purely because of its units, not because it's actually the most important
signal for finding natural customer groups. Standardising puts every feature on the same
scale so each contributes proportionally to its actual variance, not its arbitrary units.

### Step 3 — Choosing K and fitting the model, with the actual math (3 min)

**[SHOW: Step 3 code cell + elbow/silhouette plots + PCA scatter plot]**

Step three is model selection. I tested K from 2 to 10, and for each value recorded two
things: **inertia**, for the elbow method — exactly the within-cluster sum of squared
distances defined back in Part B — and **silhouette score**.

The **elbow method** works like this: inertia mechanically *always* decreases as K
increases — more centroids can only reduce or maintain total squared distance, never
increase it, down to the degenerate case of inertia equal to zero when K equals the number
of data points. So you're not looking for the minimum, you're looking for the point of
**diminishing returns** — where adding another centroid stops meaningfully reducing inertia.
Plotted as a curve, that shows up as a visible bend, or 'elbow.' Looking at the elbow curve
here, there's a visible bend around K equals 4.

But I didn't stop at eyeballing the elbow, because it's a qualitative, visual judgment call.
I also computed the **silhouette score** for every K, which is a formula, not a shape on a
graph. For a single point i, the silhouette value is: b of i minus a of i, divided by the
maximum of a of i and b of i — where a of i is the mean distance from point i to every other
point in its *own* cluster, capturing cohesion, and b of i is the mean distance from point i
to every point in the *nearest other* cluster, capturing separation. That ratio produces a
value between minus 1 and 1 for every single point; the overall silhouette score reported is
just the average across all points. The console output explicitly printed: *'K with highest
silhouette score: 2.'*

I want to be upfront that this creates a genuine tension I address properly in the evaluation
section: the elbow method's qualitative signal points toward K=4, but the purely statistical,
formula-driven silhouette answer says K=2 is best. Pretending these agreed would be
dishonest, and I don't.

I set `FINAL_K = 4` and fit the final model with `n_init=10` — ten independent Lloyd's
algorithm runs, each from a different k-means++ initialisation, keeping whichever run
achieved the lowest final inertia — which mitigates the sensitivity to initial centroid
placement flagged back in Part B.

To visualise four-dimensional clustering — four features means each point lives in
4-dimensional space, which can't be drawn directly — I used **PCA**, Principal Component
Analysis, to project the scaled features down to two components purely for visualisation.
Mechanically, PCA computes the **covariance matrix** of the four standardised features,
finds its **eigenvectors and eigenvalues**, and the eigenvectors — ranked by the size of
their corresponding eigenvalue, which equals the amount of variance captured along that
direction — become the new axes. The first two principal components are simply the two
directions of greatest variance in the data. I want to be precise about scope: PCA here is
*not* used for clustering itself, only for plotting the already-computed cluster
assignments afterwards, so it has zero influence on the actual cluster labels K-Means
produced.

The resulting cluster sizes are: Cluster 0, 514 customers; Cluster 1, 570; Cluster 2, 535;
Cluster 3, 594 — that's 2,213 total, matching the cleaned dataset. Notice the sizes are
reasonably balanced, roughly a quarter each, not one giant cluster with three tiny leftovers
— an even split like that is itself a sanity check, since a wildly imbalanced split, say one
cluster with 5 customers, would be a red flag for a degenerate solution, often caused by a
bad initialisation or genuinely poor separation in the data.

### Step 4 — Profiling segments and generating recommendations (3 min)

**[SHOW: Step 4 code cell + radar chart + boxplots + recommendations table]**

This is where the clusters get translated into something a marketer can actually act on. I
computed the mean of nine profiling features per cluster — a deliberately *larger* set than
the four clustering features, because interpretation benefits from more context — including
campaign responsiveness and household composition — even though those features had no
influence on which cluster a customer was assigned to.

Let me read out the actual profile numbers, because they tell a genuinely interesting
story:

- **Cluster 0** — average age 47.7, income $68,807, total spending $1,096, 21.6 total
  purchases, recency 23.5 days, meaning they bought recently.
- **Cluster 1** — age 43.4, income $35,102, spending just $121, 8.5 purchases, recency
  23.5 days — also recent, but far less valuable.
- **Cluster 2** — age 48.6, income $69,614, spending $1,166 — almost identical profile to
  Cluster 0 in income and spending — but recency is 73.3 days, three times higher.
- **Cluster 3** — age 45.0, income $36,909, spending $140, recency 73.7 days.

Here's the pattern, stated technically: the clusters effectively split along **two
independent axes** — income and spending, high for 0 and 2, low for 1 and 3; crossed with
recency, recent for 0 and 1, lapsing for 2 and 3. That two-by-two structure is a genuinely
clean, interpretable result, and it's the strongest evidence for why K equals 4 specifically
beats K equals 3 or K equals 2 *for this business problem* — collapsing to fewer clusters
would necessarily merge a 'recently active' group with a 'lapsing' group that have *very*
different marketing needs despite near-identical income and spending.

The rule-based `describe_cluster` function tags each cluster automatically by comparing it
to the overall average across the profiling features: Cluster 0 comes out as 'high income,
high spender, campaign-responsive'; Cluster 1 as 'low income, low spender, web-heavy
browser, lower conversion'; Cluster 2 as 'high income, high spender, campaign-responsive,
*lapsing / inactive*'; Cluster 3 as 'low income, low spender, lapsing / inactive, web-heavy
browser.'

And then the recommendation engine converts that into an action:

- Cluster 0 → **VIP / loyalty programme** — exclusive early access, premium bundles. 514
  customers.
- Cluster 1 → **Discount / deals-led campaign** — coupons, bundle pricing. 570 customers.
- Cluster 2 → **VIP / loyalty programme** as well — 535 customers.
- Cluster 3 → **Win-back campaign** — reactivation offer, personalised outreach. 594
  customers.

Now here's a limitation I want to be transparent about on camera rather than hide: Cluster 2
is both high-value *and* lapsing — recency of 73 days — but it still gets labelled VIP, the
*same* action as Cluster 0, rather than a win-back action. Technically, that's because of the
**order of conditions** inside `recommend_action` — it checks 'high value and
campaign-responsive' *before* it checks recency, so a high-value lapsing customer never
reaches the win-back branch; the first matching `if` wins and the function returns
immediately. In a real deployment I'd flag this as something to fix — perhaps a 'high-value,
at-risk' category that blends VIP retention *with* a win-back nudge, since Cluster 2 clearly
needs both. I think it's important to show critical thinking here rather than presenting
automated rule-based output as unquestionably correct."

---

## 30:00–36:30 — Part D: Evaluation, Why 'Accuracy' Doesn't Apply, and How to Improve It (6 min 30 sec)

**[SHOW: Part D cells — metric outputs and the K comparison table]**

"This is the section where I want to directly answer a question that comes up naturally:
*what's the accuracy of this model?*

### Why classification accuracy doesn't exist here

The honest, technically precise answer is: **there is no accuracy metric for this model,
and computing one would be a methodological error, not just an omission.** Accuracy is
defined as the proportion of predictions that match a known, correct label — correct
predictions divided by total predictions. That definition requires **ground truth**: a
column in the dataset that says the *actual, real* segment each customer belongs to, so a
prediction can be checked against it. This dataset has no such column, and fundamentally
*can't*, because customer segments aren't an observable fact about a customer the way
'did this email get opened' is — they're a modelling choice about how to group behaviour.
If somebody hands you an 'accuracy' number for a K-Means model without a labelled dataset
behind it, that number is meaningless, because there was nothing correct to compare against.

What *does* exist, and what I used instead, is **internal cluster validation** — metrics
that judge cluster quality purely from the geometry of the data itself: how tightly packed
each cluster is, and how well-separated clusters are from each other, with no external label
required at all. That's a fundamentally different question from accuracy — it's not 'did we
get the right answer,' it's 'is this a statistically coherent grouping.' I used three
independent ones, on the final model with K=4:

- **Silhouette Score: 0.318.** As defined properly in Part C, this is the mean, across every
  point, of b minus a over the max of a and b, where a is intra-cluster distance and b is
  distance to the nearest other cluster. It ranges from minus 1 to 1; higher is better;
  values near 1 mean points sit deep inside well-separated clusters, values near 0 mean
  points sit right on a cluster boundary, negative values mean points are probably in the
  *wrong* cluster entirely. 0.318 indicates real but moderate structure — clusters exist and
  are meaningfully separated, but there's measurable overlap at the boundaries, which is
  realistic for human behavioural data that exists on a spectrum rather than in hard boxes.
- **Davies-Bouldin Index: 1.202.** This is computed by taking, for every cluster, its
  *worst-case* pairing with another cluster — the ratio of the two clusters' combined
  internal scatter to the distance between their centroids — and averaging that worst-case
  ratio across all clusters. Lower is better, with zero being the theoretical best possible
  score; there's no fixed upper bound, so 1.202 is judged relative to other K values, which
  is exactly why the comparison table below matters more than the number in isolation.
- **Calinski-Harabasz Score: 1,657.9.** This is the ratio of between-cluster dispersion to
  within-cluster dispersion, each normalised by their respective degrees of freedom —
  conceptually similar to an F-statistic in ANOVA. Higher is better, and because it's a
  ratio rather than a bounded score, it's also primarily meaningful in comparison across
  different K values on the *same* dataset, rather than as an absolute number.

I also directly compared K equals 3, 4, 5, and 6 side by side, computing all three metrics
for each. Here are the actual numbers:

| K | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---|---|---|
| 3 | 0.355 | 0.998 | 1722.0 |
| 4 | 0.318 | 1.202 | 1657.9 |
| 5 | 0.318 | 1.072 | 1548.3 |
| 6 | 0.307 | 1.076 | 1466.0 |

I want to address this head-on rather than skip past it: **K=3 wins on all three metrics**
against K=4 — better, meaning higher, silhouette; lower, meaning better, Davies-Bouldin; and
higher, meaning better, Calinski-Harabasz. Purely on cluster-quality grounds, K=3 is the
statistically stronger choice, and even K=2 had the single highest silhouette score of every
K from 2 to 10 I tested.

So why did I keep `FINAL_K=4`? Because the four-way split identified in Step 4 — recent
high-value, lapsing high-value, recent low-value, lapsing low-value — maps onto four
genuinely distinct, independently-actionable campaigns: VIP retention, a win-back push for
the lapsing high-value group, a deals-led campaign for low spenders, and nurture for newer
low-value customers. Collapsing to K=3 would necessarily merge two of those groups that need
different marketing treatment, even though it scores marginally better on pure statistical
separation. This is a case where the *business* objective — a distinct, staffable campaign
per segment — outweighs picking the single highest-scoring K in isolation, which is precisely
the trade-off Part B's comparison predicted: K-Means' one addressable weakness is sensitivity
to K, and here that sensitivity produces a real tension between the statistically 'best' K
and the operationally most useful K, rather than a clean agreement.

### Concrete, technical ways this could be improved

If the goal is genuinely to improve these metrics — tighter, better-separated clusters —
rather than just accept 0.318 as final, there are several specific, technically-grounded
levers, and I'd rank them roughly by expected impact:

1. **Try Gaussian Mixture Models (GMM) instead of, or alongside, K-Means.** K-Means'
   spherical-cluster assumption, flagged in Part B, is a real constraint on this data — GMM
   models each cluster as a full multivariate Gaussian with its own covariance matrix, so
   clusters can be elongated or correlated rather than forced into circles, and it's fit via
   the **Expectation-Maximisation algorithm** rather than Lloyd's algorithm. It also produces
   *soft* assignments — a probability of membership per cluster, rather than a hard label —
   which is a more honest representation of boundary customers like the ones dragging the
   silhouette score down.

2. **Re-examine the feature set with a quantitative method, not just judgement.** I chose
   four features by domain reasoning; a more rigorous approach is to test feature subsets
   systematically — for example, drop `Recency` and re-run the silhouette comparison, or use
   **mutual information** between candidate features and a trial clustering to check whether
   a feature is actually adding separable signal versus just adding noise dimensions that
   dilute distance, per the curse-of-dimensionality point in Part C.

3. **Use `RobustScaler` instead of `StandardScaler` for the income and spending features
   specifically.** `StandardScaler`'s mean and standard deviation are themselves sensitive
   to outliers — even after winsorizing at the 1st/99th percentile, a scaler based on median
   and interquartile range would be less influenced by whatever extreme values remain within
   that 1st-to-99th window.

4. **Run a stability analysis via bootstrap resampling** — refit K-Means on repeated random
   subsamples of the data and measure how consistently the same four-cluster structure
   re-emerges, using something like the **Adjusted Rand Index** to compare cluster labels
   across resamples. High agreement would be direct evidence the clusters are a real,
   reproducible pattern rather than an artefact of this specific sample; this is also how
   you'd detect the boundary-instability limitation mentioned below before it caused a
   problem in production.

5. **Add a fourth internal metric, the Gap Statistic**, which compares the observed inertia
   curve against inertia curves computed on data with no real cluster structure at all —
   randomly generated within the same feature ranges — giving a more principled, less purely
   'eyeballed' answer to the elbow-method question than looking at a bend in a curve.

6. **If a genuine ground-truth ever became available** — for instance, if the business later
   ran an actual marketing experiment and could label customers by real response tier — *then*
   external validation metrics become possible: **Adjusted Rand Index**, **Normalised Mutual
   Information**, or **homogeneity/completeness scores**, which compare cluster assignments
   against true labels the way accuracy compares predictions against true labels in
   supervised learning. That would be the closest legitimate equivalent to 'accuracy' this
   problem could ever have — but it requires labelled data that doesn't currently exist, so
   it's a future validation step, not something achievable with the current dataset.

**Limitations, briefly and honestly, beyond what's already been said:** K-Means assumes
roughly spherical, similarly-sized clusters, which may not fully hold for real customer
behaviour, as GMM above would test directly. Cluster assignment isn't temporally stable — a
customer sitting near a boundary in feature space could flip clusters on a re-run with new
data, which the bootstrap stability check above would quantify rather than leave as a
suspicion. And the four core clustering features are a deliberate simplification that trades
some richness for interpretability and distance-metric cleanliness, as discussed in Part C."

---

## 36:30–37:45 — Conclusion (1 min 15 sec)

**[SHOW: Conclusion markdown cell]**

"To bring it together: this project showed how K-Means clustering, driven by minimising
within-cluster squared distance via Lloyd's algorithm, can turn 2,213 rows of raw
transactional data into four segments with genuinely distinct marketing needs, without ever
being told in advance what those segments should look like. Because this is unsupervised,
there's no accuracy figure to quote — instead, the internal validation metrics show the
clusters are reasonably well-separated, a silhouette of 0.318, though not perfectly
distinct, which is realistic, because real customer behaviour is inherently a spectrum, not
hard boxes, and I've laid out concrete, technical routes — GMM, feature re-selection,
robust scaling, bootstrap stability checks — for tightening that separation further.

Practically, this moves a business from one-size-fits-all marketing to four targeted
strategies: a loyalty programme for engaged high-value customers, a win-back campaign for
lapsing high-value customers, a deals-led push for low spenders, and standard nurture for
the rest. As a next step, I'd want to validate these recommended actions with real A/B
testing before trusting them in production — which would also finally provide the kind of
labelled outcome data needed for a genuine external validation metric — and re-run the
segmentation on newer data to check the segments remain stable over time."

---

## 37:45–39:15 — Part E: EDI & Sustainability, and Wrap-Up (1 min 30 sec)

**[SHOW: Part E markdown cell]**

"Finally, the ethical and sustainability side, which I think is easy to treat as an
afterthought but shouldn't be.

**Bias and fairness:** income and age are strong clustering signals precisely *because* they
correlate with protected characteristics. A cluster that reads as purely behavioural — 'low
spenders' — can still function as a proxy for a protected group, even though that attribute
was never fed into the model directly. Any real deployment needs an explicit
proxy-discrimination check — for instance, testing whether cluster membership correlates
with a protected attribute even though that attribute wasn't a clustering feature — not just
a check of the input feature list.

**Privacy and GDPR:** purchase history, income, and web-visit counts are personal data. A
production system needs a documented lawful basis for processing, a defined retention period
for recency and enrolment-date-derived features, and — critically — a working mechanism for
a customer to request removal from a segment. 'The model already trained on it' is not a
GDPR exemption.

**Explainability:** marketers acting on a segment need to explain, in plain language, why a
customer landed there. My rule-based `describe_cluster` function is a first step toward
that, but in a real system it would need to be validated against actual marketer judgement,
not just numeric thresholds — and I showed you a concrete case earlier, Cluster 2, where the
automated logic's condition ordering produced a debatable recommendation.

**Sustainability:** K-Means is computationally cheap — fitting completes in well under a
second even at this scale, since the algorithm's complexity is linear in the number of
customers. The bigger cost lever isn't the algorithm itself, it's *retraining cadence* —
retraining on every single new transaction is unnecessary computational overhead for no real
gain in segment quality; a scheduled, say monthly, batch retrain is very likely sufficient
given how slowly customer segments actually shift.

That's the full project — literature grounding, a genuine three-way technique comparison
with the underlying mathematics, a complete implementation from raw CSV to marketing action
with every technical decision justified, an honest evaluation that explains why accuracy
isn't the right lens for unsupervised clustering and what the real metrics say instead, a
set of concrete technical routes to improve those metrics further, and the ethical
considerations around deploying this in a real business. Thanks for watching."

---
---

# TIPS FOR RECORDING

## Before you hit record
1. **Fill the two open placeholders first.** The References section needs real content in
   the notebook itself before you present it on screen — don't present a notebook with
   visible `[FILL IN: ...]` text still in it. Write real Harvard-style references yourself;
   don't ask an AI tool to generate them (the notebook's own integrity note is right about
   this — fabricated citations are a distinct academic integrity issue).
2. **Re-run the whole notebook top to bottom once before recording**, with a fresh kernel,
   and confirm the printed numbers still match what's in this script (`RANDOM_STATE = 42`
   should make this deterministic, but verify — especially cluster sizes and the metric
   table). If your local CSV isn't present and it silently falls back to synthetic data,
   every number changes and this script becomes wrong — the console print `Loaded local
   file: (2240, 29)` is your confirmation it used the real data.
3. **Time yourself reading it out loud once**, even roughly. At this length you're likely to
   land closer to 40 minutes than 35 the first read-through — that's fine given the extra
   technical depth, but don't try to compensate by speeding up; cut the weakest sentence in
   each section instead. Speeding up reads as nervous, especially on formula explanations.

## Delivery
4. **Don't read in a monotone off the screen the whole time.** Glance at this script, say the
   idea in your own words, glance back. Sounding like you're explaining it to a colleague
   beats sounding like you're reciting a document — this matters even more now that there's
   real math in the script, since reciting a formula flatly is much harder to follow than
   plain narration.
5. **Slow down specifically on numbers and formulas.** "Silhouette equals b of i minus a of
   i, over the max of a of i and b of i" said too fast is unintelligible on a recording.
   Numbers and formulas are exactly what a marker will pause and rewind to check — give them
   a beat, and consider putting the formula as on-screen text while you say it.
6. **Own the 'why isn't there an accuracy number' moment and the K=3 vs K=4 tension out
   loud — don't bury either.** Markers specifically look for whether students understand the
   difference between supervised and unsupervised evaluation, and whether they notice when
   their own results complicate their chosen narrative. These are your two strongest
   "critical thinking" moments in the whole video — don't rush either one.

## Screen recording
7. **Use OBS Studio (free) or your OS's built-in recorder** for the notebook portion. Record
   in two passes if it's easier: audio + face-cam for the literature/comparison sections
   where there's no code on screen, then screen-only capture while narrating the
   implementation, and stitch in editing. Trying to nail 35-40 minutes in one continuous
   take raises your error rate a lot for no real benefit.
8. **Zoom in on the notebook.** At normal notebook font size, plots and printed tables are
   often unreadable on a compressed video export. Increase your browser/Jupyter zoom to
   ~150% before recording, or crop tightly in editing.
9. **Pre-render all outputs before recording** rather than re-running cells live on camera.
   Live execution risks a dependency error, a slow cell, or a random-seed mismatch eating
   your recording time. Show the *already-executed* notebook and scroll through it.

## Pacing across the video
10. **The three heaviest sections are Technique Comparison, Implementation, and
    Evaluation** — together now around 26-27 of your ~38 minutes, since that's where the
    technical depth you asked for actually lives. If you're running long, trim literature
    review citations to author-year only in speech (put full citations only on screen)
    rather than cutting into the math-heavy sections — those are where you demonstrate you
    actually understand the mechanics, not just that you read about them.
11. **Practice the transition sentences** between the six parts ("Now let's look at...",
    "That brings me to...") — these are what make a long video feel structured rather than
    six separate mini-videos glued together.
