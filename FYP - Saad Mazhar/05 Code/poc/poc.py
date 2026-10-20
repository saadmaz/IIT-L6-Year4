"""Preliminary proof of concept: Tea Sense pipeline on the public Tecator NIR dataset (moisture target)."""
import numpy as np, time, json
from scipy.signal import savgol_filter
from scipy.stats import wilcoxon
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
# ---- load ARFF
rows, attrs, data = [], [], False
for line in open('tecator.arff', encoding='latin-1'):
    l = line.strip()
    if not l or l.startswith('%'): continue
    if l.lower().startswith('@attribute'): attrs.append(l.split()[1]); continue
    if l.lower().startswith('@data'): data = True; continue
    if data: rows.append([float(v) for v in l.split(',')])
A = np.array(rows); idx = {a: i for i, a in enumerate(attrs)}
X = A[:, [idx[f'absorbance_{i}'] for i in range(1, 101)]]; y = A[:, idx['moisture']]
wl = np.linspace(850, 1050, 100)
print('samples', X.shape, 'moisture range', y.min(), y.max())

# ---- Kennard-Stone split 70/30
def kennard_stone(X, n):
    d = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    i, j = np.unravel_index(d.argmax(), d.shape); sel = [i, j]
    rem = [k for k in range(len(X)) if k not in sel]
    while len(sel) < n:
        mind = d[np.ix_(rem, sel)].min(1); k = rem[int(mind.argmax())]; sel.append(k); rem.remove(k)
    return np.array(sel), np.array(rem)
cal, test = kennard_stone(X, int(0.7 * len(X)))
Xc, Xt, yc, yt = X[cal], X[test], y[cal], y[test]

# ---- preprocessing
snv = lambda Z: (Z - Z.mean(1, keepdims=True)) / Z.std(1, keepdims=True)
sg = lambda Z, d=0: savgol_filter(Z, 11, 2, deriv=d, axis=1)
msc_ref = Xc.mean(0)
def msc(Z):
    out = np.empty_like(Z)
    for i, z in enumerate(Z):
        b, a = np.polyfit(msc_ref, z, 1); out[i] = (z - a) / b
    return out
PRE = {'Raw': lambda Z: Z, 'SNV': snv, 'MSC': msc, 'SG smoothing': lambda Z: sg(Z), 'SG 1st derivative': lambda Z: sg(Z, 1),
       'SNV + SG 1st derivative': lambda Z: sg(snv(Z), 1), 'SG 2nd derivative': lambda Z: sg(Z, 2)}

kf = KFold(10, shuffle=True, random_state=42)
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def pls_cv(Z, yv, maxlv=15):
    best = None
    for lv in range(1, maxlv + 1):
        p = cross_val_predict(PLSRegression(lv, scale=False), Z, yv, cv=kf).ravel(); r = rmse(p, yv)
        if best is None or r < best[1]: best = (lv, r)
    return best

pre_rows = []
for name, f in PRE.items():
    lv, r = pls_cv(f(Xc), yc)
    m = PLSRegression(lv, scale=False).fit(f(Xc), yc); pt = m.predict(f(Xt)).ravel()
    pre_rows.append([name, lv, round(r, 3), round(rmse(pt, yt), 3)])
    print('PRE', name, lv, round(r, 3), round(rmse(pt, yt), 3))
best_pre = min(pre_rows, key=lambda r: r[2])[0]; F = PRE[best_pre]
Zc, Zt = F(Xc), F(Xt)

# ---- simplified CARS
def cars(Z, yv, runs=50, lv=10):
    n, p = Z.shape; keep = np.arange(p); hist = []
    a = (p / 2) ** (1 / (runs - 1)); k = np.log(p / 2) / (runs - 1)
    for i in range(1, runs + 1):
        sub = rng.choice(n, int(0.8 * n), replace=False)
        m = PLSRegression(min(lv, len(keep)), scale=False).fit(Z[sub][:, keep], yv[sub])
        w = np.abs(m.coef_.ravel()); r = a * np.exp(-k * i); nkeep = max(2, int(round(r * p)))
        keep = keep[np.argsort(w)[::-1][:min(nkeep, len(keep))]]
        lvb, rb = pls_cv(Z[:, keep], yv, maxlv=min(lv, len(keep)))
        hist.append((rb, keep.copy(), lvb))
    rb, kb, lvb = min(hist, key=lambda h: h[0]); return np.sort(kb), lvb, rb
sel_counts = np.zeros(100)
best_cars = None
for rep in range(10):
    kb, lvb, rb = cars(Zc, yc); sel_counts[kb] += 1
    if best_cars is None or rb < best_cars[2]: best_cars = (kb, lvb, rb)
kb, lvb, rb = best_cars
mc = PLSRegression(lvb, scale=False).fit(Zc[:, kb], yc); ptc = mc.predict(Zt[:, kb]).ravel()
cars_row = [len(kb), lvb, round(rb, 3), round(rmse(ptc, yt), 3)]
print('CARS', cars_row)

# ---- models on best preprocessing (full spectrum)
lvp, _ = pls_cv(Zc, yc)
pls = PLSRegression(lvp, scale=False).fit(Zc, yc)
svr = GridSearchCV(make_pipeline(StandardScaler(), SVR()), {'svr__C': [1, 10, 100, 1000], 'svr__gamma': ['scale', 0.001, 0.01], 'svr__epsilon': [0.05, 0.2]}, cv=kf, scoring='neg_root_mean_squared_error').fit(Zc, yc)
rf = GridSearchCV(RandomForestRegressor(random_state=42), {'n_estimators': [300], 'max_depth': [None, 10], 'max_features': ['sqrt', 0.3]}, cv=kf, scoring='neg_root_mean_squared_error').fit(Zc, yc)
mean_pred = np.full_like(yt, yc.mean())
def stats(name, pc, pt):
    sd = yt.std(ddof=1); e = pt - yt
    return dict(model=name, RMSEC=round(rmse(pc, yc), 3), RMSEP=round(rmse(pt, yt), 3), R2p=round(1 - (e ** 2).sum() / ((yt - yt.mean()) ** 2).sum(), 3), RPD=round(sd / rmse(pt, yt), 2), bias=round(e.mean(), 3))
preds = {'PLSR': (pls.predict(Zc).ravel(), pls.predict(Zt).ravel()), 'SVR': (svr.predict(Zc), svr.predict(Zt)), 'Random Forest': (rf.predict(Zc), rf.predict(Zt)), 'CARS-PLSR': (mc.predict(Zc[:, kb]).ravel(), ptc)}
res = [stats('Mean-only baseline', np.full_like(yc, yc.mean()), mean_pred)] + [stats(k, *v) for k, v in preds.items()]
for r in res: print(r)
# paired test SVR vs PLSR
w = wilcoxon(np.abs(preds['SVR'][1] - yt), np.abs(preds['PLSR'][1] - yt))
print('wilcoxon SVR vs PLSR p=', w.pvalue)
# timing
t0 = time.perf_counter()
for i in range(len(Xt)): pls.predict(F(Xt[i:i + 1]))
tp = (time.perf_counter() - t0) / len(Xt) * 1000
print('ms per prediction', tp)

# ---- figures
best = min(res[1:], key=lambda r: r['RMSEP'])['model']
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200)
ax[0].plot(wl, Xc[:40].T, lw=0.6, color='#5b7fa8', alpha=.6); ax[0].set_xlabel('Wavelength (nm)'); ax[0].set_ylabel('Absorbance'); ax[0].set_title('Raw spectra (40 samples)', fontsize=10)
for s in kb: ax[0].axvspan(wl[s] - 1, wl[s] + 1, color='#e0a43a', alpha=.35, lw=0)
pb = preds[best][1]
ax[1].scatter(yt, pb, s=14, color='#2f6fb5'); lo, hi = y.min() - 2, y.max() + 2
ax[1].plot([lo, hi], [lo, hi], color='#888', lw=1); ax[1].set_xlim(lo, hi); ax[1].set_ylim(lo, hi)
ax[1].set_xlabel('Reference moisture (%)'); ax[1].set_ylabel('Predicted moisture (%)'); ax[1].set_title(f'{best}: held-out test set (n={len(yt)})', fontsize=10)
for a in ax: a.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig('poc.png'); plt.close()
json.dump(dict(n=len(X), ncal=len(cal), ntest=len(test), yrange=[float(y.min()), float(y.max())], pre=pre_rows, best_pre=best_pre, cars=cars_row, res=res, wilcoxon_p=float(w.pvalue), ms_pred=tp, best=best, svr_params=svr.best_params_, plslv=lvp), open('poc.json', 'w'), indent=1)
