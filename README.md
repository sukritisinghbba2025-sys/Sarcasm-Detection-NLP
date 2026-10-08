# Sarcasm Detection in Text

**CA3 NLP Mini Project** | Sukriti Singh | PRN: 25030422121

## Problem Statement
Sarcasm relies on subtext, cultural context, and irony, causing traditional keyword-based NLP pipelines to misclassify negative or absurd statements as positive. Failing to detect sarcasm corrupts downstream analytics, automated content moderation, and customer sentiment tracking.

## Objectives
To build an interpretable Machine Learning pipeline capable of distinguishing between legitimate news and sarcastic text, auditing it across text subgroups (length, style, numbers), and extracting the exact vocabulary features driving the model's predictions both globally and locally.

## Dataset
* **Name:** News Headlines Dataset for Sarcasm Detection v2
* **Source:** The Onion (sarcastic) and HuffPost (real)
* **Author:** Rishabh Misra
* **License:** CC BY 4.0
* **Size:** 28,503 unique headlines, 47.5% sarcastic
* **Link:** [Kaggle Dataset](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)

> **Important:** the label records which *outlet* wrote the headline, not whether a human judged it sarcastic. This shapes every limitation below.

## Proposed Method & Tech Stack
* **Algorithm:** Text Lemmatization + TF-IDF Vectorization + Logistic Regression / Multinomial Naive Bayes
* **Libraries:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, NLTK

---

## Project Pipeline & Business Impact

### Business Impact
In a business context, failing to detect sarcasm poisons automated sentiment pipelines. A customer tweeting, "Brilliant, my software crashed again," will be incorrectly tagged as highly positive due to the word "brilliant." Building an initial classification layer to isolate sarcasm protects data integrity, ensuring angry customers are routed to human agents rather than receiving tone-deaf automated responses.

### Preprocessing & Approach
1. All text is lowercased, and punctuation is stripped via Regex.
2. Each word is reduced to its base dictionary form using **NLTK's WordNet Lemmatizer**.
3. A **TF-IDF Vectorizer** capped at 10,000 features captures single words and two-word phrases (unigrams and bigrams) and removes English stop words. The vocabulary is learned on the training set only, so there is no data leakage.
4. A **Logistic Regression classifier** (C=1.5, max iterations 1000) makes the primary prediction. A **Multinomial Naive Bayes** model serves as the baseline.
5. Split: 80% train / 20% test with a fixed seed (42).

*Note: Logistic Regression was selected over a neural network because each word receives one readable weight, so the model's behaviour can be explained exactly, with no approximation.*

## Final Test Performance Metrics

| Metric (sarcasm class) | Logistic Regression | Naive Bayes |
| :--- | :--- | :--- |
| **Overall Accuracy** | 79.62% | 79.51% |
| **Precision** | 79.60% | 79.48% |
| **Recall** | 75.57% | 75.46% |
| **F1** | 0.775 | 0.774 |
| **AUC** | 0.878 | 0.881 |

> Evaluated on 5,701 unseen test headlines. The two models are almost tied, so Logistic Regression is the final model because its weights can be audited directly. Precision (0.80) and recall (0.76) are balanced, so the model is not simply favouring one class.

---

## Model Card & Audit

### Model Details & Intended Use
* **Architecture:** Logistic Regression on TF-IDF sparse matrices (unigrams and bigrams).
* **Intended Use:** A pre-filtering layer that flags possibly sarcastic text for manual review before data is passed to literal sentiment analysis models. For learning and demonstration; not for decisions about people.
* **Out-of-Scope:** Multi-paragraph documents, spoken audio, and non-English text.

### Subgroup Audit Results
The test set was split along three independent axes. False-positive rate (FPR) is the share of *real* headlines wrongly flagged, which equals wasted human review effort.

| Axis | Subgroup | n | Accuracy | Sarcasm Recall | FPR |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Length** | Short (< 10 words) | 2,489 | 78.1% | 73.4% | 17.6% |
| | Long (>= 10 words) | 3,212 | 80.8% | 77.4% | 16.4% |
| **Style** | Question | 165 | 77.6% | 56.5% | 19.0% |
| | Statement | 5,536 | 79.7% | 75.7% | 16.8% |
| **Numbers** | Has a number | 811 | 81.6% | 75.3% | 12.6% |
| | No number | 4,890 | 79.3% | 75.6% | 17.6% |

**Is each accuracy gap real?** 1,000 bootstrap resamples of the test set give a 95% interval for each gap:

| Axis | Gap (pts) | 95% CI | Verdict |
| :--- | :--- | :--- | :--- |
| Length (long - short) | +2.76 | 0.59 to 4.90 | Real gap |
| Style (question - statement) | -2.10 | -8.37 to 4.31 | Could be noise |
| Numbers (has - none) | +2.34 | -0.47 to 5.01 | Could be noise |

**Audit Insight:** Short headlines are the confirmed weak spot. TF-IDF relies on cumulative word weights, so short sentences offer fewer signals to overcome the decision threshold. Question headlines show the lowest sarcasm recall (56.5%), but on only 165 headlines, so that result needs caution.

### Top Feature Explanations
Because Logistic Regression assigns a concrete coefficient to every word, a headline's score is `intercept + sum(tfidf value x word weight)`, an exact explanation.
* **🔴 Sarcasm Signals:** The highest positive coefficients belong to *nation, area, man, report, local*. *The Onion* frequently relies on the trope "Area Man Does X," making these generic nouns strong satire indicators.
* **🟢 Real News Signals:** The lowest negative coefficients belong to *trump, donald, donald trump, queer, muslim*. *HuffPost* heavily covers political figures and social topics.
* **Local example:** "Area man wins argument with thermostat" scores 0.99 sarcastic, driven by *area* (+2.50), *man* (+1.87) and *area man* (+0.85).

### Evidence the Model Leans on Outlet Style
Retraining after removing the words behind the 25 strongest weights on each side (49 words) lowers accuracy from 79.62% to 75.81% (**-3.81 pts**). Removing 49 *random* words changes it by only **-0.04 pts**. A small set of outlet-style cues carries a disproportionate share of the signal.

### Stress Test (19 hand-written texts)

| Category | Accuracy |
| :--- | :--- |
| Onion-style sarcasm | 100% |
| Explicit markers ("yeah right", "/s") | 100% |
| Real news | 80% |
| Sincere negative | 50% |
| Dry sarcasm | 25% |
| Sincere positive | 0% |
| **Overall** | **63%** |

The sample is tiny, so treat this as indicative only.

### Calibration
Predicted probabilities are reasonably calibrated around 0.5 (observed sarcasm rate 49.8% for predictions in 0.4 to 0.6), and slightly under-confident at the extremes (the 0.8 to 1.0 bin predicts 89.6% but observes 94.4%).

## Model Limitations & Caveats
1. **The labels measure the outlet, not sarcasm.** The model has largely learned publication style and structural formulas (e.g., "Area Man...") rather than semantic irony, so it may fail on sarcasm from other sources such as tweets and reviews.
2. **Dry or context-dependent sarcasm is missed.** On the hand-written set it catches only 1 of 4 examples such as "I absolutely love being stuck in traffic", and it wrongly flags sincere, casual statements as sarcastic, because their vocabulary looks unlike HuffPost headlines.
3. **Uneven subgroup performance.** Short headlines are weaker (confirmed), and question headlines have low recall on a small sample.
4. **No context or tone.** Bag-of-words sees word order only through bigrams.
5. **Narrow data.** English, US news headlines from a fixed time period.

**Recommendation:** re-test on text from a different source before any real use, and consider a transformer model (such as BERT) for context-aware detection.
