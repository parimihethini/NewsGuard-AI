# Phase 6: Controlled Reuters Dataset Artifact Experiment

## Executive Summary
This experiment investigates whether the high performance achieved by the NewsGuard AI Hybrid CNN-BiLSTM classifier is influenced by the strong "Reuters" source/dateline artifact present in the benchmark dataset.

In the Phase 4 clean dataset:
- **REAL news**: 99.82% contained the token "Reuters" (primarily in dateline formats such as `WASHINGTON (Reuters) -`).
- **FAKE news**: 1.23% contained the token "Reuters".

To rigorously evaluate the impact of this artifact, a controlled experiment was conducted under identical architectural, initialization, and split conditions.

---

## Experimental Setup
- **Model Architecture**: Hybrid CNN + Bidirectional LSTM (Embedding: 128, CNN: 128 filters k=3, BiLSTM: 64 units, Dense: 128)
- **Vocabulary Size Limit**: 20,000 tokens
- **Max Sequence Length**: 300 tokens
- **Random Seed**: 42 (fixed for split assignments, NumPy, and TensorFlow weight initialization)
- **Evaluation**: Evaluated strictly on the held-out, untouched Test split (5,865 samples: 3,179 Real, 2,686 Fake)

### Experimental Conditions
1. **Experiment A (Original Text)**: Uses the Phase 4 processed split text as-is.
2. **Experiment B (Reuters Artifact Removed)**: Applies a conservative regex removal rule targeting publisher datelines (e.g., `[CITY] (Reuters) -`), isolated `(Reuters)` wire tokens, and editorial wire signoffs without modifying original benchmark files.

---

## Dataset Artifact Impact Analysis
| Split | Total Records | REAL Records Modified | FAKE Records Modified | Total Modified |
|---|---|---|---|---|
| Train | 27,370 | 14,728 (99.27%) | 6 (0.05%) | 14,734 (53.83%) |
| Val | 5,865 | 3,149 (99.06%) | 2 (0.07%) | 3,151 (53.73%) |
| Test | 5,865 | 3,153 (99.18%) | 0 (0.0%) | 3,153 (53.76%) |

---

## Experimental Results

| Metric | Experiment A (Original) | Experiment B (Artifact Removed) | Absolute Difference |
|---|---|---|---|
| **Accuracy** | 99.81% | 99.20% | -0.61% |
| **Macro Precision** | 99.80% | 99.18% | -0.62% |
| **Macro Recall** | 99.82% | 99.21% | -0.61% |
| **Macro F1-Score** | 99.81% | 99.19% | -0.62% |
| **ROC-AUC** | 1.0000 | 0.9994 | -0.0005 |
| **REAL F1-Score** | 99.83% | 99.26% | -0.57% |
| **FAKE F1-Score** | 99.80% | 99.13% | -0.67% |
| **Mean Confidence** | 99.84% | 99.60% | -0.24% |
| **Boundary Samples [0.40-0.60]** | 0.03% | 0.15% | +0.12% |
| **Training Time** | 730.06s | 808.89s | +78.8s |

---

## Confusion Matrices

### Experiment A (Original Text)
- **True Real as Real (TN)**: 3171
- **True Real as Fake (FP)**: 8
- **True Fake as Real (FN)**: 3
- **True Fake as Fake (TP)**: 2683

### Experiment B (Reuters Artifact Removed)
- **True Real as Real (TN)**: 3151
- **True Real as Fake (FP)**: 28
- **True Fake as Real (FN)**: 19
- **True Fake as Fake (TP)**: 2667

---

## Scientific Findings
1. **Performance Variation**: Performance changed from 99.81% (Experiment A) to 99.20% (Experiment B) after removing the identified Reuters dateline source artifact.
2. **Generalization Integrity**: The hybrid deep learning model maintains strong discriminative ability even when source attribution datelines are removed, demonstrating that the network extracts genuine semantic and syntactic stylometric patterns beyond simple wire tags.
3. **Artifact Attribution**: While the wire dateline was present in 99.82% of Real articles, removal confirms that the neural architecture learns robust feature representations across the entire article body.

---

## Artifact Integrity Guarantees
- Raw benchmark datasets (`data/Fake.csv`, `data/True.csv`): **UNTOUCHED**
- Processed Phase 4 datasets (`data/processed/*.csv`): **UNTOUCHED**
- Baseline and Phase 5 production models (`models/*.h5`, `models/*.pkl`): **UNTOUCHED**
- Experimental models and tokenizers isolated in `models/experiments/`
