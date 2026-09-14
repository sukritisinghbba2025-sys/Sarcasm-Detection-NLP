# Project Title: SARCASM DETECTION IN TEXT

**Problem Statement:** 
Sarcasm relies on subtext, cultural context, and irony, causing traditional keyword-based NLP pipelines to misclassify negative or absurd statements as positive. Failing to detect sarcasm corrupts downstream analytics, automated content moderation, and customer sentiment tracking.

**Objectives / Expected Outcome:** 
To build an interpretable Machine Learning pipeline capable of distinguishing between legitimate news and sarcastic text, auditing it for length-based biases, and extracting the exact vocabulary features driving the model's predictions both globally and locally.

**Dataset Name and Link:** 
News Headlines Dataset for Sarcasm Detection (Rishabh Misra) | [Kaggle Link](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)

**Dataset Source / License:** 
Sourced from The Onion and HuffPost. Available on Kaggle for research and academic use (Requires citation to author Rishabh Misra).

**Proposed Method / Tools / Libraries:** 
* Algorithm: TF-IDF Vectorization + Logistic Regression
* Libraries: Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

---

## 1. Project Write-up & Pipeline

**Business Impact**
In a business context, failing to detect sarcasm poisons automated sentiment pipelines. A customer tweeting, "Brilliant, my software crashed again," will be incorrectly tagged as highly positive due to the word "brilliant." Building an initial classification layer to isolate sarcasm protects data integrity, ensuring angry customers are routed to human agents rather than receiving tone-deaf automated responses.

**Preprocessing & Approach**
Text preprocessing is kept minimal but essential: all text is lowercased, and punctuation/URLs are stripped via Regex. The core engine is a Term Frequency-Inverse Document Frequency (TF-IDF) Vectorizer capped at 10,000 features, mapping the text into numerical vectors while severely penalizing uninformative stop words. A Logistic Regression classifier handles the prediction. This specific algorithm was selected over a neural network to guarantee 100% mathematical explainability, allowing us to audit exactly which vocabulary terms influence the decision threshold.

**Final Test Performance Metrics**
* **Overall Accuracy:** 79.84%
* **False Positives:** 497
* **False Negatives:** 657
The model effectively parsed the data, returning a strong true positive recognition rate with balanced Precision and Recall scores, confirming it did not simply overfit to the majority class.

---

## 2. Model Card & Audit

**Model Details & Intended Use**
* **Architecture:** Logistic Regression optimized on TF-IDF sparse matrices. 
* **Intended Use:** To serve as a pre-filtering layer that flags sarcastic text for manual review before data is passed to literal sentiment analysis models.
* **Out-of-Scope:** This model is not intended for analyzing multi-paragraph documents, processing spoken audio, or evaluating non-English text.

**Subgroup Audit Results**
The test set was partitioned by vocabulary length to test for linguistic density bias.

| Subgroup | Condition | Accuracy |
| :--- | :--- | :--- |
| **Short Headlines** | < 10 Words | 78.40% |
| **Long Headlines** | >= 10 Words | 81.03% |

*Audit Insight:* The 2.6% accuracy gap indicates the model struggles with brevity. Because TF-IDF relies on cumulative word weights, shorter sentences offer fewer mathematical signals to overcome the baseline decision threshold. 

**Top Feature Explanations**
Because Logistic Regression assigns a concrete mathematical coefficient to every word, we can perfectly explain its behavior:
* **Sarcasm Signals:** The highest positive coefficients belong to *area, report, man, nation, local*. *The Onion* frequently relies on the trope "Area Man Does X," establishing these generic nouns as heavy satire indicators.
* **Real News Signals:** The lowest negative coefficients belong to *donald, trump, california, watch, heres*. *HuffPost* heavily utilizes political entities, state names, and multimedia tags in its standard reporting.

**Model Limitations & Caveats**
The primary failure mode is that the model has not learned abstract human sarcasm; it has learned publication style. It relies heavily on structural formulas (e.g., "Area Man...") rather than semantic irony. Consequently, it fails completely on "dry" or context-dependent sarcasm (e.g., "I absolutely love being stuck in traffic") where the individual vocabulary is standard, but the premise is absurd to a human reader.
