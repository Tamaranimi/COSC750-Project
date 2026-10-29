from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, GroupKFold

def build_svm_search():
    """Group-safe SVM hyperparameter search with fold-specific preprocessing."""

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            class_weight={0: 1, 1: 100}
        ))
    ])

    search = GridSearchCV(
        estimator=pipeline,
        param_grid={
            "svm__C": [1, 10],
            "svm__gamma": ["scale"]
        },
        scoring="average_precision",
        cv=GroupKFold(n_splits=3),
        n_jobs=1,
        refit=True
    )

    return search
