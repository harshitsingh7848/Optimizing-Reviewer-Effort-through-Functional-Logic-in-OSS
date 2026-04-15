from scipy.stats import spearmanr

def check_circularity(valid, effort_values):
    circularity_results = {}
    features = {
        "additions":        [r.get("additions", 0) for r in valid],
        "deletions":        [r.get("deletions", 0) for r in valid],
        "cyclomatic_delta": [r.get("cyclomatic_delta", 0) for r in valid],
        "max_nesting":      [r.get("max_nesting", 0) for r in valid],
        "logic_density":    [r.get("logic_density", 0.0) for r in valid],
    }
    print("\n Circularity Check: Feature Correlation with Effort")
    for name, vals in features.items():
        rho, p = spearmanr(vals, effort_values)
        circularity_results[name] = {
            "spearman_rho": round(rho, 4),
            "p_value":      round(p, 4),
            "significant":  bool(p < 0.05),
        }
    return circularity_results