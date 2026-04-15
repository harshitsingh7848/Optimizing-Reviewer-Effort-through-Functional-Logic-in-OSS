from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

def compute_rf_baseline(valid: list, gt_labels: list):
    """
    Random Forest baseline trained on structural + semantic features.
    Uses 80/20 train/test split with stratification to preserve class balance.

    Features used:
      - cyclomatic_delta: functional complexity change
      - max_nesting:      structural depth
      - logic_density:    semantic logic ratio
      - total_size:       additions + deletions (size signal)
    """
    X = np.array([[
        r.get("cyclomatic_delta", 0),
        r.get("max_nesting", 0),
        r.get("logic_density", 0.0),
        r.get("additions", 0) + r.get("deletions", 0),
    ] for r in valid])

    y = np.array(gt_labels)

    indices = list(range(len(valid)))
    idx_train, idx_test = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    return rf_preds, y_test, idx_test