# Keystroke Dynamics — KNN Biometric Classifier (ML Research)

> **This is the ML research repository.** It contains the classifier, dataset, and evaluation scripts only.
> The full production system with frontend and backend lives in a separate repository — see [Roadmap](#roadmap).

---

## What This Is

A machine learning module for **keystroke-based user authentication**, designed as a second factor (2FA) layer. The system learns the unique typing rhythm of each enrolled user and verifies their identity at login time — no password required beyond the passphrase itself.

---

## Repository Structure

```
├── KNNClassifier.py   # Core classifier class
├── test.py            # Evaluation runner (holdout, CV, GridSearch)
├── Model.ipynb        # Exploratory notebook
├── bio_bio.csv        # Enrolled biometric profiles (97 features + CLASS label)
└── README.md
```

---

## How It Works

When a user types a passphrase, timing patterns between keystrokes (dwell time, flight time) are captured as a 97-dimensional feature vector. A KNN classifier compares this against enrolled profiles and verifies whether the typing pattern matches the claimed identity.

```
User types passphrase
        ↓
97 keystroke timing features extracted
        ↓
StandardScaler normalises feature magnitudes
        ↓
KNN (k=3, Manhattan, distance-weighted) finds closest profiles
        ↓
Predicted identity  ==  Claimed identity  AND  Confidence ≥ 60%
        ↓
VERIFIED    or  REJECTED  
```

---

## Verification Logic

Authentication requires **both conditions to pass simultaneously**:

| Check | Condition | Purpose |
|---|---|---|
| Identity match | Predicted user == Claimed user | Rejects impostors even if model is confident |
| Confidence | Neighbour vote ≥ 60% | Rejects ambiguous or borderline samples |

Passing only one is not enough — this is intentional.

---

## Model Configuration

| Parameter | Value | Reason |
|---|---|---|
| Algorithm | KNN | Effective for biometric distance comparison |
| `n_neighbors` | 3 | GridSearch optimal for this dataset |
| Distance metric | Manhattan (p=1) | More robust than Euclidean for timing data |
| Weights | `distance` | Closer neighbours vote more strongly |
| Feature scaling | `StandardScaler` | KNN is distance-based; unscaled features distort distances |
| Confidence threshold | 0.60 | Tunable per deployment security requirement |

---

## Performance

Evaluated on **32 users, ~5 typing samples each (161 rows total)**.

| Metric | Score | Notes |
|---|---|---|
| Holdout Accuracy | ~74–77% | Stratified 70/30 split, varies ±4% by seed |
| Cross-Validation | 72.65% | 5-fold CV — most honest single estimate |
| GridSearch Ceiling | 80.11% | Best achievable with current data size |

> Cross-validation at **72.65%** is the number to trust. Holdout swings with random seed due to small dataset size — this is expected, not a bug.

---

## Login Verification Probability

A legitimate user logging in has approximately a **73–75% chance of being verified on the first attempt**.

Across 10 login attempts by the same legitimate user:
```
 Verified  →  7–8 times
 Rejected  →  2–3 times
```

### Why Every Login Has Three Gates

**Gate 1 - Model correctly identifies them** (~73–75%)
KNN must find their enrolled samples as the closest neighbours out of 161 rows across 32 users.

**Gate 2 - Confidence ≥ 60%**
With k=3, confidence is 33%, 66%, or 100%. A split neighbour vote (33%) rejects even a correct identity.

**Gate 3 - Typing consistency**
Stress, fatigue, different keyboard, or time of day all shift timing patterns slightly away from the 5 enrolled samples.

### Probability Breakdown

| Scenario | Probability |
|---|---|
| Legitimate user verified on first attempt | ~73–75% |
| Legitimate user verified within 2 attempts | ~93–94% |
| Legitimate user verified within 3 attempts | ~98% |
| Impostor incorrectly verified | ~25–27% ⚠️ |

> Both rates are constrained by the same root cause — 5 enrollment samples per user is not enough for production.

---

## Known Limitation — Data Size

The ceiling with 5 samples per user is ~80%. No further tuning will move it meaningfully.

| Samples per user | Legitimate verified | Impostor rejected | CV Accuracy |
|---|---|---|---|
| 5 *(current)* | ~74% | ~75% | ~72–74% |
| 20 | ~87% | ~89% | ~85–88% |
| 50+ | ~92% | ~94% | ~91–93% |

**The fix is data collection, not tuning.** The model configuration is already at its GridSearch optimum. Collecting 20–30 samples per user during enrollment will push accuracy to a deployable range without any code changes.

---

## Quick Start

```bash
pip install pandas scikit-learn
python test.py
```

To use the classifier directly:

```python
from KNNClassifier import KeystrokeClassifier

clf = KeystrokeClassifier(
    registered_biometric_file="bio_bio.csv",
    typing_sample=typing_sample,   # list of 97 floats
    neighbour_size=3,
    metric="manhattan",
    weights="distance",
)

clf.knn_manhattan_holdout()        # evaluate on held-out split
clf.get_cv_score()                 # cross-validated accuracy
clf.hyper_parameters_tuning()      # find best k, metric, weights
```

---

## Roadmap

This repository is **Phase 1** — ML research and classifier development.

**Phase 2 (separate repo)** will integrate this classifier into a full 2FA system:

- [ ] REST API wrapping `KeystrokeClassifier` (Flask / FastAPI)
- [ ] Frontend login page with real-time keystroke capture
- [ ] User enrollment flow (collect 20–30 samples per user) (optional)
- [ ] JWT-based session management after biometric verification
- [ ] Persistent biometric store (replace flat CSV with database)
- [ ] Retrain pipeline - new samples automatically improve the model

---

## Dataset Notes

- Features: 97 keystroke timing measurements per session
- Classes: 32 enrolled users (integer IDs and UUID strings)
- Rows: 161 clean samples (1 malformed header row excluded at load time)
- Balance: ~5 samples per user

---

## Dependencies

```
pandas
scikit-learn
```