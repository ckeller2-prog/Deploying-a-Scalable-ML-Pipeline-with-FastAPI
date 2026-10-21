# Model Card

## Model Details
- **Developed By:** Machine Learning DevOps Student
- **Model Date:** October 2026
- **Model Type:** Random Forest Classifier
- **Version:** 1.0
- **License:** MIT

## Intended Use
- **Primary Intended Uses:** Predicting whether an individual's income exceeds 0,000 per year based on demographic and employment census attributes.
- **Primary Intended Users:** Students and researchers studying machine learning pipelines.
- **Out-of-Scope Use:** Should not be used for automated high-stakes financial decisions without bias auditing.

## Training Data
- **Dataset:** Census Income Dataset (Adult dataset) from the UCI Machine Learning Repository.
- **Data Split:** 80% training data and 20% test data, stratified by the target label.

## Evaluation Data
- **Dataset:** The 20% held-out test split from the census dataset.

## Metrics
- **Performance Measures:** Precision, Recall, and F1-Score (beta=1).
- **Overall Model Performance:**
  - **Precision:** 0.7353
  - **Recall:** 0.6378
  - **F1-Score:** 0.6831

## Ethical Considerations
- The dataset contains demographic attributes (such as race, sex, and native-country) which can reflect historical societal biases. Slice-based performance evaluations were conducted to monitor subgroup metrics.

## Caveats and Recommendations
- Performance varies across demographic slices due to representation disparities in the underlying census data. Further fairness auditing is recommended.
