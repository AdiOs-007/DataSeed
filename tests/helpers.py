"""Small local fixtures shared by the tests (no downloaded data, nothing outside the repo)."""
import numpy as np
import pandas as pd

from dataseed import schema_from_dict


def schema(*cols, relationships=None, constraints=None, **kw):
    d = {"dataset_name": "t", "columns": list(cols), "relationships": relationships or []}
    if constraints:
        d["constraints"] = constraints
    d.update(kw)
    return schema_from_dict(d)


def student_sample(n=60, seed=3, tmp_csv=None):
    """Correlated, rule-based student sample (marks ~ attendance; pass_status derived)."""
    rng = np.random.default_rng(seed)
    att = np.clip(rng.normal(74, 13, n), 35, 100)
    marks = np.clip(0.9 * (att - 74) + rng.normal(58, 14, n), 10, 99)
    df = pd.DataFrame({
        "student_id": [f"STU-{i:04d}" for i in range(1, n + 1)],
        "age": rng.integers(18, 26, n),
        "gender": rng.choice(["Male", "Female"], n),
        "attendance": att.round(1), "marks": marks.round(1)})
    df["pass_status"] = np.where(df.marks >= 40, "Pass", "Fail")
    if tmp_csv is not None:
        df.to_csv(tmp_csv, index=False)
    return df
