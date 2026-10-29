# Member 3 — Logistic Regression: Updated Experimental Notes

## Method / design

- **Task:** Binary email classification (`0 = legitimate`, `1 = spam`) using the updated shared Enron-Spam dataset.
- **Shared data:** 21,460 training emails and 4,390 validation emails. The test set was **not** used for model fitting, hyperparameter selection or this evaluation.
- **Input representation:** A `TfidfVectorizer` with unigram word features (`ngram_range=(1,1)`), `min_df=2`, `max_features=50,000`, `sublinear_tf=False`, smoothed IDF and L2 normalization. The vocabulary and IDF weights are fitted **only on training data**. Email subject and body are retained; no stop-word removal or stemming is applied.
- **Algorithm:** Logistic Regression models the estimated probability of spam using the sigmoid function, `p(y=1|x) = 1/[1+exp(-(wᵀx+b))]`. The parameters are optimized using regularized logistic loss. `C` is the inverse of regularization strength; higher `C` means weaker regularization.
- **Fixed settings:** `solver='liblinear'`, `max_iter=2000`, prediction threshold `0.5`, `random_state=42`.
- **Tuning:** Six settings: `C ∈ {0.1, 1, 10}` × `class_weight ∈ {None, 'balanced'}`. Model selection used the highest validation spam F1-score, with precision and experiment ID as tie-breakers.

## Updated validation results (new dataset)

The best run was **Experiment #5**, with **`C=10` and `class_weight=None`**. Training converged in **8 solver iterations**. Its measured validation results were:

| Metric | Value |
|---|---:|
| Accuracy | 99.09% |
| Precision (spam) | 98.58% |
| Recall (spam) | 99.52% |
| F1-score (spam) | 99.05% |
| ROC AUC | 0.99905 |
| True negatives (TN) | 2,264 |
| False positives (FP) | 30 |
| False negatives (FN) | 10 |
| True positives (TP) | 2,086 |
| Training time for selected classifier (excluding TF-IDF fitting) | 0.83652 s |
| Validation prediction time for selected classifier | 0.00462 s |

The model classified **4,350 of 4,390** validation emails correctly. Training and inference timings refer to a single local run and should not be treated as a cross-model speed comparison unless measured under identical hardware and software conditions.

### Hyperparameter experiments

| Experiment | C | Class weight | Precision | Recall | F1-score | FP | FN |
|---|---:|---|---:|---:|---:|---:|---:|
| 1 | 0.1 | none | 94.05% | 99.52% | 96.71% | 132 | 10 |
| 2 | 0.1 | balanced | 93.63% | 99.52% | 96.48% | 142 | 10 |
| 3 | 1 | none | 97.34% | 99.57% | 98.44% | 57 | 9 |
| 4 | 1 | balanced | 97.03% | 99.62% | 98.31% | 64 | 8 |
| 5 | 10 | none | 98.58% | 99.52% | 99.05% | 30 | 10 |
| 6 | 10 | balanced | 98.44% | 99.57% | 99.00% | 33 | 9 |

Increasing `C` from 0.1 to 10 improved F1-score for both class-weight settings within the tested range. `class_weight='balanced'` did not improve F1-score in these six runs. This does not establish that larger `C` is always preferable or that class weighting is never useful.

## Error analysis — updated validation split

**False positives (30):** Legitimate messages sometimes resembled bulk or promotional email. Illustrative cases from `validation_errors.csv`:

| Actual label | Email subject / description | Estimated spam probability | Model prediction |
|---|---|---:|---|
| Legitimate | Welcome to the next wave in digital music | 99.58% | Spam |
| Legitimate | Registration confirmation from Spinner.com | 97.47% | Spam |
| Legitimate | Dell order confirmation | 83.30% | Spam |
| Legitimate | Marathon proposal for our meeting | 59.91% | Spam |

The first three contain features associated with legitimate commercial messages, subscriptions or confirmation workflows. Their formatting and vocabulary may resemble spam, though keyword overlap has **not** been verified as the exact cause of these individual errors.

**False negatives (10):** Some spam resembled ordinary event invitations, business messages or newsletters. Examples:

| Actual label | Email subject / description | Estimated spam probability | Model prediction |
|---|---|---:|---|
| Spam | Important information about your house | 48.89% | Legitimate |
| Spam | Powerball results | 25.32% | Legitimate |
| Spam | Recruitment service advertisement to executives / HR managers | 25.70% | Legitimate |
| Spam | When will the real estate bubble burst? (event invitation) | 17.78% | Legitimate |
| Spam | Peter G tour dates | 10.29% | Legitimate |

The mortgage example contains character substitutions such as `m?rtgage applicat?on`. Such word obfuscation may reduce the usefulness of word-based TF-IDF features; however, the precise contribution of each token was not tested. The real-estate invitation appears in two similar error examples, which suggests that duplicate or near-duplicate content within a split merits further checking.

These cases illustrate **lexical overlap**, **unusual/obfuscated spellings**, **short or conversational messages**, and **context ambiguity** as plausible explanations, not demonstrated causal mechanisms.

## Feature interpretation — updated model

`influential_features.csv` ranks individual TF-IDF features by the fitted Logistic Regression coefficient. For a fixed set of other features, a larger positive coefficient shifts the log-odds toward spam, while a negative coefficient shifts them toward legitimate email.

| Higher spam association | Coefficient | Higher legitimate association | Coefficient |
|---|---:|---|---:|
| `2004` | +7.584 | `enron` | -17.984 |
| `2005` | +7.464 | `vince` | -11.107 |
| `money` | +6.310 | `thanks` | -10.141 |
| `your` | +6.136 | `louise` | -9.565 |
| `http` | +6.056 | `2001` | -8.813 |

The prominence of years, Enron-specific names and other corpus-related tokens is consistent with possible dataset-specific learning. It is **not proof** of the underlying source of those associations. Coefficients are not probabilities or causal effects, and single tokens cannot determine a message's final prediction by themselves.

## Validity and limitations

- The **train–validation normalized exact-text overlap check returned 0** for the updated split. This check does not rule out near-duplicates or within-split duplication, and the group should maintain a consistent dataset policy.
- The **validation set** was used for hyperparameter selection; therefore validation scores are not a final unbiased estimate of performance. Member 5 should assess all frozen models on the same **untouched test set**.
- The same TF-IDF settings as the earlier Member 2 approach were used, but **no updated Naïve Bayes results have been supplied here**. Do **not** carry forward the previous Naïve Bayes comparison or claims that Logistic Regression reduced false negatives by a particular amount; recompute those comparisons when Member 2 provides the new results on the same split.
- The trained model may be affected by distribution shift, adversarial obfuscation and ambiguous promotional/legitimate content.
- Predicted spam probabilities are uncalibrated model estimates; a high estimate does not guarantee correctness.
- Differences between the old and new validation percentages cannot establish improved generalisation because the validation data changed.

## References to verify and cite in the group report

1. Jáñez-Martino, F. et al. (2023). *A review of spam email detection: analysis of spammer strategies and the dataset shift problem*. Artificial Intelligence Review 56, 1145–1173. https://doi.org/10.1007/s10462-022-10195-4
2. scikit-learn, `LogisticRegression`: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
3. scikit-learn, `TfidfVectorizer`: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

**Note on submission:** These are evidence-backed working notes, not a finished report. Write the final analysis in your own words, verify references, and disclose any generative-AI assistance as required by the project instructions.
