# Sarcasm Detection in Text

**CA3 NLP Mini Project** | Sukriti Singh | PRN: 25030422121

## Problem Statement
Sarcasm relies on subtext, cultural context, and irony, causing traditional keyword-based NLP pipelines to misclassify negative or absurd statements as positive. Failing to detect sarcasm corrupts downstream analytics, automated content moderation, and customer sentiment tracking.

## Objectives
Build an interpretable ML pipeline that separates legitimate news from sarcastic text, audit it across text subgroups (length, style, numbers), and extract the exact vocabulary driving its predictions both globally and locally.

## Dataset
* **Name:** News Headlines Dataset for Sarcasm Detection v2
* **Source:** The Onion (sarcastic) and HuffPost (real)
* **Author:** Rishabh Misra | **License:** CC BY 4.0
* **Size:** 28,503 unique headlines, 47.5% sarcastic
* **Link:** [Kaggle Dataset](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)

> **Important:** the label records which *outlet* wrote the headline, not whether a human judged it sarcastic.

## Proposed Method & Tech Stack
* **Algorithm:** Lemmatization + TF-IDF + Logistic Regression (baseline: Multinomial Naive Bayes)
* **Libraries:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, NLTK

---

## Project Pipeline & Business Impact

### Business Impact
A customer tweeting "Brilliant, my software crashed again" is tagged positive because of the word "brilliant". A first-pass sarcasm filter protects sentiment data and routes angry customers to human agents instead of tone-deaf automated replies.

### Preprocessing & Approach
1. Lowercase and strip punctuation with a regex.
2. Reduce words longer than 3 letters to their base form with NLTK's WordNet Lemmatizer (short words are skipped because WordNet turns "has" into "ha").
3. TF-IDF, 10,000 features, unigrams and bigrams, English stop words removed. The vocabulary is learned on training data only (no leakage).
4. Logistic Regression with C=3, chosen by 5-fold cross-validation on the training set; Naive Bayes as baseline. 80/20 split, seed 42.

*Logistic Regression was chosen over a neural network because each word gets one readable weight, so predictions can be explained exactly.*

## Final Test Performance Metrics

| Metric (sarcasm class) | Logistic Regression | Naive Bayes |
| :--- | :--- | :--- |
| **Accuracy** | 79.72% | 79.60% |
| **Precision** | 79.00% | 79.42% |
| **Recall** | 76.86% | 75.80% |
| **F1** | 0.779 | 0.776 |
| **AUC** | 0.878 | 0.880 |

> Evaluated once on 5,701 unseen test headlines. The models are nearly tied, so Logistic Regression is final because its weights can be audited directly. Precision and recall are balanced.

---

## Model Card & Audit

### Model Details & Intended Use
* **Architecture:** Logistic Regression on TF-IDF sparse matrices.
* **Intended Use:** a pre-filter that flags possibly sarcastic text for manual review before literal sentiment analysis. For learning and demonstration; not for decisions about people.
* **Out-of-Scope:** multi-paragraph documents, spoken audio, non-English text.

### Subgroup Audit Results
FPR = share of *real* headlines wrongly flagged (wasted human review).

| Axis | Subgroup | n | Accuracy | Sarcasm Recall | FPR |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Length** | Short (< 10 words) | 2,489 | 77.8% | 74.1% | 18.7% |
| | Long (10+ words) | 3,212 | 81.2% | 79.2% | 17.1% |
| **Style** | Question | 165 | 79.4% | 60.9% | 17.6% |
| | Statement | 5,536 | 79.7% | 77.0% | 17.8% |
| **Numbers** | Has a number | 811 | 82.0% | 78.9% | 15.2% |
| | No number | 4,890 | 79.3% | 76.5% | 18.2% |

**Are the gaps real?** Gap in points (first minus second group) with a 95% bootstrap interval from 1,000 resamples:

| Axis | Accuracy gap | Sarcasm-recall gap |
| :--- | :--- | :--- |
| Long - short | +3.37 (1.22 to 5.54), real gap | +5.08 (2.02 to 8.33), real gap |
| Question - statement | -0.34 (-6.7 to 5.83), could be noise | -16.13 (-36.02 to 2.35), could be noise |
| Has number - none | +2.65 (-0.31 to 5.56), could be noise | +2.40 (-2.07 to 6.83), could be noise |

**Audit Insight:** long headlines beat short ones by 3.37 accuracy points (real gap); short text gives TF-IDF fewer signals to work with. The lowest sarcasm recall is for *Question* (60.9%, n=165); bootstrap verdict: **could be noise**, so treat it with caution.

### Top Feature Explanations
A headline's score is `intercept + sum(tfidf value x word weight)`, an exact explanation.
* **🔴 Sarcasm Signals:** *area, nation, report, man, clearly*. The Onion often uses the "Area Man Does X" formula.
* **🟢 Real News Signals:** *queer, trump, allegedly, trans, donald*. HuffPost covers political figures and social topics.

### Reliance on a Few Outlet-Style Words
Removing the words behind the 25 strongest weights per side (49 words) and retraining changes accuracy by **-3.53 pts**. Removing 49 random words changes it by **-0.18 pts**. A small set of outlet-style cues carries a disproportionate share of the signal (evidence, not proof).

### Stress Test (24 hand-written texts, 4 per category)

| Category | Accuracy |
| :--- | :--- |
| Onion-style sarcasm | 100% |
| Explicit markers | 75% |
| Real news | 75% |
| Dry sarcasm | 25% |
| Sincere negative | 25% |
| Sincere positive | 25% |
| **Overall** | **54%** |

Mostly non-headline text, so part of any failure is distribution shift. Indicative only.

### Calibration
In the middle probability bin the model says 0.50 and the observed sarcasm rate is 0.51; the largest gap in any bin is 0.03.

## Model Limitations & Caveats
1. **Labels measure the outlet, not sarcasm.** The model leans on publication style and formulas such as "Area Man...", so it may fail on sarcasm from other sources (tweets, reviews).
2. **Dry, context-dependent sarcasm is missed** (25% correct on the hand-written set), and sincere casual text is often wrongly flagged (75% of sincere positive examples).
3. **Uneven subgroup performance** (table above); small groups need caution.
4. **No context or tone.** Bag-of-words sees word order only through bigrams.
5. **Narrow data:** English, US news headlines from a fixed period.

**Recommendation:** re-test on text from another source before any real use, and consider a transformer such as BERT for context-aware detection.
