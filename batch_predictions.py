import numpy as np

def predict_svm_in_batches(model, df, features, batch_size=10000):
    """Predict SVM decision scores in batches without dropping records."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    scores = []

    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]

        X = (
            batch[features]
            .replace([np.inf, -np.inf], np.nan)
            .to_numpy(dtype=float)
        )

        scores.append(model.decision_function(X))

    if not scores:
        return np.empty(0, dtype=float)

    return np.concatenate(scores)
