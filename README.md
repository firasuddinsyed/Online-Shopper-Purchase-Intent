# 🛒 Online Shopper Purchase Intent Prediction

A binary classification case study predicting whether an e-commerce session
will result in a purchase — built to handle a significant class imbalance
(84.5% non-purchase vs 15.5% purchase).

**Course:** BUAN/MKT 6337 — Predictive Analytics for Data Science | UT Dallas | Spring 2025  
**Dataset:** [UCI Online Shoppers Purchasing Intention](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)

---

## 📊 Results

| Model | F1 | Recall | ROC-AUC |
|---|---|---|---|
| Logistic Regression (baseline) | 0.482 | 0.356 | 0.869 |
| Random Forest | 0.655 | 0.654 | 0.923 |
| SMOTE + Random Forest | 0.663 | 0.696 | 0.923 |
| **XGBoost (final)** | **0.666** | **0.775** | **0.924** |

**Optimal decision threshold:** 0.615 (vs default 0.5) — identified via precision-recall curve

---

## 🔑 Key Contributions

### Feature Engineering
6 new features engineered from domain knowledge — `page_efficiency` became
the **#1 most important feature** in XGBoost, outranking all 17 original features:

| Feature | Logic | Business Meaning |
|---|---|---|
| `page_efficiency` | PageValues / total pages | Commercial value per page visited |
| `engagement_score` | Product pages + duration + PageValues | Composite browsing intent signal |
| `bounce_exit_ratio` | BounceRate / ExitRate | Distinguishes quick-leave vs browse-then-leave |
| `is_holiday_season` | Nov or Dec = 1 | Holiday sessions convert at 2.5× average |
| `total_pages` | Sum of all page counts | Overall session breadth |
| `total_duration` | Sum of all durations | Total time invested in session |

### Class Imbalance Handling
Three strategies compared:
- `class_weight='balanced'` in Random Forest
- SMOTE synthetic oversampling (sampling_strategy=0.4)
- `scale_pos_weight` in XGBoost — best overall result

### Threshold Optimization
Default threshold (0.5) replaced with optimal threshold (0.615) found via
precision-recall curve — increases buyer detection rate while maintaining
acceptable precision for marketing use.

---

## 📁 Repository Structure
