import torch
import ssl

_original_create_default_context = ssl.create_default_context

def _create_unverified_context(*args, **kwargs):
    ctx = _original_create_default_context(*args, **kwargs)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx

ssl.create_default_context = _create_unverified_context
ssl._create_default_https_context = _create_unverified_context

from ucimlrepo import fetch_ucirepo
import pandas as pd
from model._preproc import one_hot_with_cap, split_data, scale_dataframes, to_tensors

# NOTE: obesity (544) has categorical target but UCI lists it as Classification/Regression/Clustering.
# We use it as ordinal regression (category codes 0-6).
# NOTE: seoulBike (560) has wrong target in UCI metadata ("Functioning Day" instead of "Rented Bike Count").
# We fix this via DATASETS_WITH_TARGET_FIX.
datasets = [
    ("student_perf", 320, "regression"),
    ("abalone", 1, "regression"),
    ("obesity", 544, "regression"),
    ("bike", 275, "regression"),
    ("realstate", 477, "regression"),
    ("energy_efficiency", 242, "regression"),
    ("concrete", 165, "regression"),
    ("ai4i", 601, "regression"),
    ("appliances", 374, "regression"),
    ("popularity", 332, "regression"),
    ("iris", 53, "classification"),
    ("hearth", 45, "classification"),
    ("winequalityc", 186, "classification"),
    ("breast", 17, "classification"),
    ("adult", 2, "classification"),
    ("bank", 222, "classification"),
    ("wine", 109, "classification"),
    ("car_evaluation", 19, "classification"),
    ("student_dropout", 697, "classification"),
    ("mushrooms", 73, "classification"),
    ("seoulBike", 560, "regression"),
]

DATASETS_WITH_TARGET_FIX = {
    560: "Rented Bike Count",
}


def get_ucidata(dataset_id, task, device="cuda", cap=50):
    dataset = fetch_ucirepo(id=dataset_id)

    X = dataset.data.features
    y = dataset.data.targets

    if dataset_id in DATASETS_WITH_TARGET_FIX:
        target_col = DATASETS_WITH_TARGET_FIX[dataset_id]
        y = X[[target_col]]
        X = X.drop(columns=[target_col])

    X = X.dropna(axis=1)

    X_all, orig_num_cols, dummy_cols = one_hot_with_cap(X, cap=cap)

    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    if y.dtype.name in ["object", "str", "string", "category"]:
        y = y.astype("category").cat.codes

    if task == "classification":
        class_dict = {old: new for new, old in enumerate(sorted(y.unique()))}
        y = y.map(class_dict)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_all, y)

    X_train, X_val, X_test = scale_dataframes(
        X_train, X_val, X_test, num_cols=orig_num_cols
    )

    X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t = to_tensors(
        X_train, X_val, X_test, y_train, y_val, y_test, task, device=device
    )

    return X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t
