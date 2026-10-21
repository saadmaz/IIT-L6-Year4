# 🥛 Synthetic Milk Adulteration Detection Dataset — Indian Brands

> ⚠️ **IMPORTANT DISCLAIMER**: This is a **100% synthetically generated** dataset created for educational and ML research purposes only. All sample values, chemical readings, adulteration rates, and test results are **entirely simulated** using statistical distributions and domain knowledge from FSSAI/BIS standards. This dataset does **not** represent real test results from any company and must **not** be used to make any claims about the actual product quality of any brand mentioned.

---

## 📋 Dataset Overview

| Property | Value |
|---|---|
| **Total Samples** | 2,500 |
| **Companies / Brands** | 25 Indian dairy brands |
| **Features (per sample)** | 128 original + 396 reference metadata columns |
| **Adulteration Rate** | ~5% (124 adulterated / 2,376 pure) |
| **Adulterant Types** | 11 types |
| **Time Period (simulated)** | Jan 2022 – Dec 2023 |
| **License** | CC0 1.0 Public Domain |

---

## 📁 Files

| File | Description |
|---|---|
| `milk_combined_full_dataset.csv` | Main dataset — 2,500 rows × 524 columns (features + reference metadata) |
| `milk_adulteration_samples.csv` | Standalone adulteration samples with 26 core features (2,160 rows) |
| `milk_adulterant_reference.csv` | Reference table — 38 chemical tests with standards, limits, instruments |
| `indian_milk_companies_samples.csv` | Company-wise samples with 42 features |

---

## 🏷️ Feature Groups

### 1. Sample Identity (11 columns)
- `Sample_ID`, `Company_Name`, `Brand_Name`, `Organization_Type`, `State`
- `Market_Tier` (Premium / Standard / Economy)
- `Market_Segment`, `Collection_Date`, `Season`, `Collection_Point`, `Milk_Type`

### 2. Compositional Parameters (7 columns)
- `Fat_percent`, `SNF_percent`, `Protein_percent`, `Lactose_percent`
- `Moisture_percent`, `Ash_percent`, `Total_Solids_percent`

### 3. Physical Parameters (7 columns)
- `pH`, `Specific_Gravity`, `Acidity_Titratable_pct`, `Viscosity_cP`
- `Freezing_Point_C`, `Conductivity_mS_cm`, `CLR_Corrected_Lactometer`

### 4. Chemical Quantitative Tests (11 columns)
- `Urea_mg_per_100mL`, `MUN_mg_per_dL`, `Melamine_ppm`, `Melamine_LCMS_ppm`
- `Formaldehyde_ppm`, `H2O2_ppm`, `Sucrose_g_per_100mL`
- `Reducing_Sugar_g_per_100mL`, `Chloride_mg_per_100mL`
- `Estimated_Added_Water_pct`, `Refractive_Index`

### 5. Microbiological Parameters (4 columns)
- `Total_Plate_Count_CFU_mL`, `Coliform_CFU_mL`
- `Somatic_Cell_Count_mL`, `MBRT_hours`

### 6. Enzyme Assays (5 columns)
- `ALP_mU_per_L` (pasteurization indicator)
- `Lactoperoxidase_Activity_U_mL`, `Urease_Activity_U_mL`
- `Lipase_Activity_U_mL`, `Protease_Activity_U_mL`

### 7. Qualitative Spot Tests — Binary 0/1 (18 columns)
| Test | Adulterant Targeted |
|---|---|
| `Iodine_Test_Starch` | Starch |
| `DMAB_Test_Urea` + `Urease_Strip_Urea` | Urea |
| `Methylene_Blue_Detergent` + `Foam_Test_Detergent` | Detergent/Soap |
| `FeSO4_H2SO4_Test_Formalin` + `Schiff_Test_Formalin` | Formalin |
| `Vanadate_Test_H2O2` + `TiO2_Strip_H2O2` | Hydrogen Peroxide |
| `Rosalic_Acid_Test_Alkali` | NaOH/Alkali |
| `CTAB_Test_NaHCO3` | Sodium Bicarbonate |
| `Baudouin_Test_VegOil` + `Halphen_Test_CottonseedOil` | Vegetable Oil |
| `Benedict_Test_Glucose` + `GOD_Strip_Glucose` | Glucose/Sugar |
| `AgNO3_Test_Salt` | Excess Salt |

### 8. Fat Quality Tests (5 columns)
- `Iodine_Value_g_I2_per_100g`, `Saponification_Value_mg_KOH_per_g`
- `Reichert_Meissl_Value`, `Polenske_Value`, `Butyro_Refractometer_40C`

### 9. HPLC Protein Profiling (3 columns)
- `Casein_HPLC_g_per_100mL`, `Whey_Protein_HPLC_g_per_100mL`, `Casein_Whey_Ratio`

### 10. GC-MS Free Fatty Acid Profile (5 columns)
- `FFA_Butyric_C4_pct`, `FFA_Caprylic_C8_pct`, `FFA_Oleic_C18_1_pct`
- `FFA_Linoleic_C18_2_pct`, `FFA_Palmitic_C16_pct`

### 11. NIR Spectroscopy (4 columns)
- `NIR_Abs_1210nm_Fat`, `NIR_Abs_1450nm_Water`
- `NIR_Abs_1730nm_Protein`, `NIR_Abs_2100nm_Lactose`

### 12. Fluorescence Spectroscopy (3 columns)
- `Fluorescence_Tryptophan_nm`, `Fluorescence_Riboflavin_nm`, `Fluorescence_Intensity_AU`

### 13. Electronic Nose Sensor Array (10 columns)
- `ENose_Sensor_S01` through `ENose_Sensor_S10`
- MOS (Metal Oxide Semiconductor) sensor readings, normalized 0–1

### 14. Labels / Target Variables (4 columns)
| Column | Type | Description |
|---|---|---|
| `Adulterant_Detected` | String | Adulterant name or "None" |
| `Is_Adulterated` | Binary (0/1) | **Primary ML target** |
| `FSSAI_Compliance` | String | PASS / FAIL |
| `Quality_Score_0_100` | Float | ⚠️ Derived — do not use as feature |

---

## 🏭 Brands Covered (25)

| Brand | Organization | State | Tier |
|---|---|---|---|
| Amul | Cooperative (GCMMF) | Gujarat | Premium |
| Mother Dairy | Government (NDDB) | Delhi | Premium |
| Nandini | Cooperative (KMF) | Karnataka | Standard |
| Heritage | Private | Telangana | Premium |
| Gowardhan | Private (Parag) | Maharashtra | Premium |
| Dodla | Private | Telangana | Standard |
| Milma | Cooperative | Kerala | Standard |
| Saras | Cooperative | Rajasthan | Standard |
| Aavin | Cooperative | Tamil Nadu | Standard |
| Verka | Cooperative | Punjab | Standard |
| Banas | Cooperative | Gujarat | Standard |
| Tirumala | Private | Andhra Pradesh | Standard |
| Sudha | Cooperative | Bihar | Economy |
| Sanchi | Cooperative | Madhya Pradesh | Economy |
| Vijaya | Cooperative | Andhra Pradesh | Standard |
| Vita | Cooperative | Haryana | Standard |
| Aanchal | Cooperative | Uttarakhand | Economy |
| Chitale | Private | Maharashtra | Premium |
| Country Delight | Private D2C | Delhi/NCR | Premium |
| Akshayakalpa | Private (Organic) | Karnataka | Premium |
| Paras | Private | Delhi/NCR | Standard |
| DMS | Government | Delhi | Economy |
| Dudhsagar | Cooperative | Gujarat | Standard |
| Mahananda | Cooperative | Maharashtra | Economy |
| Arun (Hatsun) | Private | Tamil Nadu | Standard |

---

## 🧪 Adulterant Types (11)

| Adulterant | Health Risk | Detection Method |
|---|---|---|
| Water | Low | Lactometer, Freezing Point |
| Urea | High | DMAB Test, Urease Strip |
| Detergent/Soap | High | Methylene Blue, Foam Test |
| Starch | Medium | Iodine Test |
| Formalin | Very High | FeSO4-H2SO4 Test |
| Hydrogen Peroxide | High | Vanadate/Paratoludin Test |
| Skimmed Milk Powder | Low | Nitric Acid Test, SDS-PAGE |
| Glucose/Sugar | Medium | Benedict's Test, GOD Strip |
| Sodium Bicarbonate | Medium | CTAB Test, Rosalic Acid |
| Vegetable Oil | Medium | Baudouin Test, GC |
| Melamine | Very High | LC-MS/MS, HPLC |

---

## 🤖 ML Use Cases

### Task 1: Binary Classification (Recommended Starting Point)
```python
# Target: Is_Adulterated (0 = Pure, 1 = Adulterated)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X = df[feature_cols]  # see notebook for feature list
y = df['Is_Adulterated']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
model = RandomForestClassifier(n_estimators=200, class_weight='balanced')
model.fit(X_train, y_train)
```

### Task 2: Multi-class Adulterant Identification
```python
# Target: Adulterant_Detected (12 classes including 'None')
y_multi = df['Adulterant_Detected'].fillna('None')
```

### Task 3: Anomaly Detection (Unsupervised)
```python
from sklearn.ensemble import IsolationForest
# Train only on pure samples
X_pure = X[y == 0]
iso = IsolationForest(contamination=0.05)
iso.fit(X_pure)
predictions = iso.predict(X)  # -1 = anomaly
```

---

## ⚠️ Important Notes for ML

1. **Class Imbalance**: 95% pure vs 5% adulterated. Use `class_weight='balanced'`, SMOTE, or threshold tuning. Never rely on Accuracy alone — use **F1, ROC-AUC, Recall**.

2. **Avoid Data Leakage**: Do **NOT** use these columns as features:
   - `Adulteration_Index_0_1` — derived from label
   - `FSSAI_Chemical_Score` — derived from label
   - `Quality_Score_0_100` — derived from label

3. **META Columns**: The combined dataset contains `__META_` suffix columns (reference information). Filter them out before training:
   ```python
   feature_cols = [c for c in df.columns if '__META_' not in c]
   ```

4. **Missing Values**: `Adulterant_Detected` is NaN for pure samples. Fill with `'None'` before encoding.

5. **Oversampling**: Apply SMOTE/oversampling **only to training data** after the train/test split to prevent data leakage.

---

## 📐 Reference Standards Used

| Standard | Scope |
|---|---|
| IS:1479 Part I & II (BIS) | Chemical tests for milk adulteration |
| FSSAI Food Safety Standards 2011 | Indian food safety regulations |
| FSSAI Rapid Testing Manual 2018 | Rapid adulteration detection kits |
| ISO 9622 | NIR analysis of milk |
| ISO 11816-1 | Alkaline phosphatase activity |
| IDF 182:2006 | Lactoperoxidase assay |
| AOAC 967.18 | Formaldehyde detection |
| AOCS Cd 5-40 | Halphen test for cottonseed oil |

---

## 📊 Simulation Methodology

Each feature was generated using:
- **Truncated normal distributions** parameterized by adulterant type and concentration
- **Realistic sensitivity/specificity** for spot tests (92–97% TP rate, 96–99% TN rate)
- **Seasonal variation** in composition (winter higher fat, monsoon lower SNF)
- **Brand-tier quality profiles** (premium brands: lower adulteration probability)
- **Physico-chemical coupling** (e.g. water adulteration reduces SG, fat%, SNF% simultaneously)
- **Enzyme inactivation logic** (formalin/H2O2 suppress LPO and ALP activity)

---

## 📜 License

This dataset is released under **CC0 1.0 Universal (Public Domain Dedication)**.
You can copy, modify, distribute and perform the work without asking permission.

---

## 🙏 Citation

If you use this dataset in your research or project, please cite:

```
Synthetic Milk Adulteration Detection Dataset — Indian Brands (2024)
Generated using FSSAI/BIS chemical standards and domain-based simulation.
Available at: https://www.kaggle.com/datasets/[your-username]/[dataset-slug]
```

---

*Generated for educational and ML research purposes. Not affiliated with any dairy company or government body.*
