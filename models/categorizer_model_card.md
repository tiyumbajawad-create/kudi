# Categorizer model card

**Trained:** synthetic Kudi household data, seeds [1, 2, 3, 4, 5, 6], 24 months each
**Algorithm:** char n-gram (2-5) TF-IDF + amount/date features -> calibrated (isotonic) logistic regression
**Evaluation split:** grouped by merchant (§5.3) -- 4 held-out merchants never seen in training, zero merchant overlap between train and test by construction

## Headline metrics (classifier zone only -- rule-covered merchants excluded)

- Macro-F1 (grouped-by-merchant split, honest): **0.246** (design doc target: >= 0.85)
- Majority-class baseline macro-F1: 0.146
- Word-token TF-IDF baseline macro-F1 (grouped split): 0.000
- Char n-gram (this model, grouped split): 0.246
- **Char n-gram, naive random row-split macro-F1: 1.000** -- shown deliberately: this is the *wrong* way to evaluate (same
merchant leaks into both train and test), and the gap between this
number and the grouped-split number above is the leakage effect
§5.3/Appendix B #3 warns about, made visible rather than just asserted.

## Chosen confidence threshold (tau)

tau = 0.7 -> coverage 0.377, precision 0.512 (design doc target: precision >= 0.9 at coverage >= 0.8)

## Coverage/precision curve

```
 threshold  coverage  precision
       0.3  0.720859   0.595745
       0.4  0.466258   0.523026
       0.5  0.461656   0.528239
       0.6  0.458589   0.531773
       0.7  0.377301   0.512195
       0.8  0.368098   0.525000
       0.9  0.365031   0.529412
```

## Full classification report

```
                              precision    recall  f1-score   support

           Food & Drink>Bars      0.000     0.000     0.000         0
    Food & Drink>Restaurants      0.995     0.715     0.832       270
              Health>Medical      0.000     0.000     0.000         0
           Shopping>Clothing      0.000     0.000     0.000         0
        Shopping>Electronics      1.000     0.805     0.892       123
Shopping>General Merchandise      0.000     0.000     0.000       107
         Transport>Rideshare      0.000     0.000     0.000       152

                    accuracy                          0.448       652
                   macro avg      0.285     0.217     0.246       652
                weighted avg      0.601     0.448     0.513       652

```

## Limitations

- **The macro-F1 target (>= 0.85) is not met on this catalog.** The
  per-class breakdown above shows why: categories with multiple
  training merchants generalize well (Restaurants F1 ~0.83, Electronics
  F1 ~0.89), but categories where the held-out test merchant's only
  same-category training example is a completely different brand name
  score near zero -- character n-grams of "TARGET" don't share enough
  with "AMAZON" to transfer the "General Merchandise" label. This is a
  genuine consequence of this demo's small (~15-merchant) classifier
  zone, not a bug: with more merchants per category (approaching the
  design doc's ~300-merchant target catalog), the model would see more
  varied within-category examples to generalize from. The naive
  row-split number above shows what happens without this honesty --
  it's substantially higher precisely because it's measuring
  memorization, not generalization.
- Evaluated on a grouped merchant split against a small (~15-merchant)
  classifier-zone catalog -- test-set merchants are entirely unseen at
  train time, which is a harder and more honest task than a random row
  split, but also means metrics are noisier with this few held-out
  merchants than they would be at production scale.
- Trained entirely on synthetic data; the noise model (processor
  prefixes, truncation, geo suffixes) reflects the generator in
  `datagen/`, not necessarily every real bank's actual formatting.
  The architecture and the user-correction feedback loop are what
  should transfer to real data, not the trained weights themselves.
- Rule-covered merchants are excluded from this evaluation by design --
  they never reach the classifier in production, so scoring them here
  would inflate the reported metric without meaning anything about the
  classifier's actual job (the long tail).
- Categories with only one merchant in the whole catalog (e.g. Bars)
  are structurally impossible to hold out for a merchant-grouped
  evaluation -- forced into train, so the model has never been tested
  on a genuinely unseen example of that category.
