# CNN Model - Final Evaluation Report

**Date:** January 3, 2026  
**Model:** CNN Drowsiness Classifier  
**Model Path:** `backend/models/cnn_drowsiness.h5`

---

## Executive Summary

This report presents a comprehensive evaluation of the CNN-based drowsiness detection model, including both same-dataset and cross-dataset performance analysis. The evaluation reveals important insights about model generalization and dataset characteristics.

---

## Evaluation Methodology

### Training Configuration
- **Training Datasets:** DDD dataset (5,000 alert + 5,000 drowsy images)
- **Total Training Images:** 10,000 images
- **Model Architecture:** CNN with 4 convolutional blocks
- **Input Size:** 224×224 pixels
- **Training Split:** 70% train, 15% validation, 15% test

### Evaluation Approaches
1. **Same-Dataset Evaluation:** Test on held-out portion of DDD dataset
2. **Cross-Dataset Evaluation:** Test on completely different NTHUDDD dataset

---

## Table 1: Same-Dataset Performance (DDD Test Set)

**Test Set:** 1,500 images from DDD dataset (750 alert + 750 drowsy)

| Metric      | Value  | Percentage |
|-------------|--------|------------|
| **Accuracy**    | 1.0000 | 100.00%    |
| **Precision**   | 1.0000 | 100.00%    |
| **Recall**      | 1.0000 | 100.00%    |
| **F1-Score**    | 1.0000 | 100.00%    |

### Confusion Matrix (Same-Dataset)

| Metric                    | Count |
|---------------------------|-------|
| **True Negatives (TN)**   | 750   |
| **False Positives (FP)**  | 0     |
| **False Negatives (FN)**  | 0     |
| **True Positives (TP)**   | 750   |

---

## Table 2: Error Rates - Same Dataset (DDD)

| Metric                          | Value  | Percentage |
|---------------------------------|--------|------------|
| **FAR** (at threshold 0.5)      | 0.0000 | 0.00%      |
| **FRR** (at threshold 0.5)      | 0.0000 | 0.00%      |
| **EER** (Equal Error Rate)      | 0.0000 | 0.00%      |
| **AUC** (Area Under Curve)      | 1.0000 | 100.00%    |

---

## Table 3: Cross-Dataset Performance (NTHUDDD Test Set)

**Test Set:** 4,000 images from NTHUDDD dataset (2,000 alert + 2,000 drowsy)  
**Note:** NTHUDDD dataset was NOT used during training

| Metric      | Value  | Percentage |
|-------------|--------|------------|
| **Accuracy**    | 0.5000 | 50.00%     |
| **Precision**   | 0.5000 | 50.00%     |
| **Recall**      | 1.0000 | 100.00%    |
| **F1-Score**    | 0.6667 | 66.67%     |

### Confusion Matrix (Cross-Dataset)

| Metric                    | Count |
|---------------------------|-------|
| **True Negatives (TN)**   | 0     |
| **False Positives (FP)**  | 2,000 |
| **False Negatives (FN)**  | 0     |
| **True Positives (TP)**   | 2,000 |

---

## Table 4: Error Rates - Cross Dataset (NTHUDDD)

| Metric                          | Value  | Percentage |
|---------------------------------|--------|------------|
| **FAR** (at threshold 0.5)      | 1.0000 | 100.00%    |
| **FRR** (at threshold 0.5)      | 0.0000 | 0.00%      |
| **EER** (Equal Error Rate)      | 0.8260 | 82.60%     |
| **AUC** (Area Under Curve)      | 0.5870 | 58.70%     |

---

## Analysis and Discussion

### Same-Dataset Performance

The model achieved **perfect 100% accuracy** on the DDD test set, demonstrating:
- Excellent learning of DDD dataset characteristics
- No false positives (FAR = 0%)
- No false negatives (FRR = 0%)
- Perfect class separation (AUC = 100%)

**Interpretation:** The model successfully learned to distinguish between alert and drowsy states within the DDD dataset's specific conditions (lighting, camera angles, subject demographics).

### Cross-Dataset Performance

The model achieved only **50% accuracy** on the NTHUDDD dataset, revealing:
- Poor generalization to unseen data source
- High false acceptance rate (FAR = 100%)
- Model classifies all alert images as drowsy
- Performance equivalent to random guessing

**Interpretation:** The model overfitted to DDD-specific features rather than learning general drowsiness patterns. This indicates:
1. Dataset-specific characteristics (lighting, camera setup, subject demographics)
2. Limited diversity in training data
3. Need for multi-dataset training approach

### Key Findings

#### Strengths
✓ Excellent performance on same-source data  
✓ Robust feature extraction within known conditions  
✓ Zero false negatives on training dataset source  

#### Limitations
✗ Poor generalization across datasets  
✗ High false positive rate on unseen data  
✗ Dataset-specific feature learning  
✗ Limited real-world applicability without retraining  

---

## Metric Definitions

### Performance Metrics
- **Accuracy**: Overall correctness (correct predictions / total predictions)
- **Precision**: Of all drowsy predictions, how many were correct (TP / (TP + FP))
- **Recall**: Of all actual drowsy cases, how many were detected (TP / (TP + FN))
- **F1-Score**: Harmonic mean of precision and recall

### Error Rate Metrics
- **FAR (False Acceptance Rate)**: Rate of alert states incorrectly classified as drowsy
  - Formula: FP / (FP + TN)
  - Lower is better
  
- **FRR (False Rejection Rate)**: Rate of drowsy states incorrectly classified as alert
  - Formula: FN / (FN + TP)
  - Lower is better
  
- **EER (Equal Error Rate)**: Point where FAR = FRR
  - Optimal operating point
  - Lower is better
  
- **AUC (Area Under ROC Curve)**: Overall discrimination ability
  - Range: 0.5 (random) to 1.0 (perfect)
  - Higher is better

---

## Recommendations

### For Academic Reporting

1. **Report Both Results**: Present both same-dataset and cross-dataset performance
2. **Acknowledge Limitations**: Discuss generalization challenges
3. **Explain Findings**: Dataset-specific learning vs. general feature learning
4. **Future Work**: Multi-dataset training approach

### For Improved Performance

1. **Multi-Dataset Training**
   - Train on mixed datasets (DDD + NTHUDDD + yawing)
   - Ensure balanced representation from each source
   - Expected improvement: 85-95% cross-dataset accuracy

2. **Data Augmentation**
   - Add lighting variations
   - Include different camera angles
   - Simulate real-world conditions

3. **Transfer Learning**
   - Use pre-trained face recognition models
   - Fine-tune on drowsiness detection
   - Better generalization expected

4. **Real-World Validation**
   - Test with live camera feed
   - Validate in actual driving conditions
   - Collect edge case scenarios

---

## Conclusion

The CNN model demonstrates **excellent performance on same-source data (100% accuracy)** but **poor generalization to unseen datasets (50% accuracy)**. This is a common challenge in machine learning and highlights the importance of:

1. **Diverse training data** from multiple sources
2. **Cross-dataset validation** to assess generalization
3. **Proper evaluation methodology** beyond single-dataset testing

The current model is suitable for:
- ✓ Proof of concept demonstrations
- ✓ DDD dataset-specific applications
- ✓ Understanding model capabilities and limitations

The model requires retraining for:
- ✗ Production deployment
- ✗ Real-world driving scenarios
- ✗ Cross-dataset generalization

---

## Next Steps

### Immediate Actions
1. Retrain model with all three datasets mixed together
2. Implement proper train/test split across all datasets
3. Re-evaluate cross-dataset performance

### Long-Term Improvements
1. Collect more diverse training data
2. Test with real-time camera feed
3. Implement ensemble methods
4. Add domain adaptation techniques

---

## Evaluation Scripts

### Run Same-Dataset Evaluation
```bash
cd backend
python scripts/proper_evaluation.py --total-per-class 5000
```

### Run Cross-Dataset Evaluation
```bash
cd backend
python scripts/cross_dataset_evaluation.py --test-size 2000
```

### Run Quick Test
```bash
cd backend
python scripts/test_trained_model.py
```

---

## References

**Datasets Used:**
- DDD (Driver Drowsiness Dataset): 41,793 images
- NTHUDDD: 66,521 images  
- yawing: 5,119 images

**Model Details:**
- Architecture: Custom CNN (4 conv blocks + 2 dense layers)
- Framework: TensorFlow/Keras
- Input: 224×224×3 RGB images
- Output: Binary classification (alert/drowsy)

---

**Report Generated:** January 3, 2026  
**Model Version:** 1.0.0  
**Evaluation Status:** Complete
