import bisect
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
import optuna
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(42)


def load_data(train_path="train.csv", test_path="test.csv", sample_path="sample_submission.csv"):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sample_path)
    return train, test, sample_sub


def base_clean(df):
    df = df.copy()
    df["product_category"] = df["product_category"].str.strip().str.title()
    df["product_category"] = df["product_category"].replace(
        {
            "Fruits And Vegetables": "Fruits and Vegetables",
            "Health And Hygiene": "Health and Hygiene",
        }
    )
    df["fat_content"] = df["fat_content"].str.strip().str.title()
    return df


def run_eda(train_c, show=True):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(train_c["total_sales"], bins=40, color="steelblue", edgecolor="white")
    axes[0].set_title("Distribution of total_sales")

    cat_sales = train_c.groupby("product_category")["total_sales"].mean().sort_values()
    axes[1].barh(cat_sales.index, cat_sales.values, color="darkorange")
    axes[1].set_title("Average sales by product category")
    plt.tight_layout()
    if show:
        plt.savefig("eda_overview.png", dpi=100)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter(train_c["product_price"], train_c["total_sales"], alpha=0.3, s=8)
    axes[0].set_title("Price vs Sales")

    store_sales = train_c.groupby("store_code")["total_sales"].mean().sort_values()
    axes[1].bar(store_sales.index, store_sales.values, color="seagreen")
    axes[1].set_title("Average sales by store")
    axes[1].tick_params(axis="x", rotation=90)

    format_sales = train_c.groupby("store_format")["total_sales"].mean().sort_values()
    axes[2].barh(format_sales.index, format_sales.values, color="indianred")
    axes[2].set_title("Average sales by store format")
    plt.tight_layout()
    if show:
        plt.savefig("eda_detail.png", dpi=100)
    plt.close(fig)


def build_features(train_c, test_c, n_folds=5, seed=42, smooth_m_product=15, smooth_m_store=50):
    full = pd.concat([train_c.drop(columns=["total_sales"]), test_c], ignore_index=True)

    prod_weight_map = full.groupby("product_code")["product_weight_kg"].mean()
    full["product_weight_kg"] = full["product_weight_kg"].fillna(full["product_code"].map(prod_weight_map))
    cat_weight_map = full.groupby("product_category")["product_weight_kg"].median()
    full["product_weight_kg"] = full["product_weight_kg"].fillna(full["product_category"].map(cat_weight_map))
    full["product_weight_kg"] = full["product_weight_kg"].fillna(full["product_weight_kg"].median())

    full["store_size"] = full["store_size"].fillna("Unknown")
    full["price_per_kg"] = full["product_price"] / full["product_weight_kg"]
    full["store_age_bucket"] = pd.cut(
        full["store_age_years"], bins=[0, 25, 30, 40, 50], labels=["<=25", "26-30", "31-40", "41-50"]
    )

    n_train = len(train_c)
    tr = full.iloc[:n_train].copy()
    te = full.iloc[n_train:].copy()
    tr["total_sales"] = train_c["total_sales"].values
    global_mean = tr["total_sales"].mean()

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for col, m in [("product_code", smooth_m_product), ("store_code", smooth_m_store)]:
        oof = np.zeros(len(tr))
        for tr_idx, va_idx in kf.split(tr):
            fold_tr = tr.iloc[tr_idx]
            stats = fold_tr.groupby(col)["total_sales"].agg(["mean", "count"])
            smoothed = (stats["mean"] * stats["count"] + global_mean * m) / (stats["count"] + m)
            oof[va_idx] = tr.iloc[va_idx][col].map(smoothed).fillna(global_mean).values
        tr[f"{col}_te"] = oof
        stats_full = tr.groupby(col)["total_sales"].agg(["mean", "count"])
        smoothed_full = (stats_full["mean"] * stats_full["count"] + global_mean * m) / (stats_full["count"] + m)
        te[f"{col}_te"] = te[col].map(smoothed_full).fillna(global_mean)

    cat_stats = tr.groupby("product_category")["total_sales"].mean()
    tr["category_avg_sales"] = tr["product_category"].map(cat_stats)
    te["category_avg_sales"] = te["product_category"].map(cat_stats).fillna(global_mean)

    store_stats = tr.groupby("store_code")["total_sales"].mean()
    tr["store_avg_sales"] = tr["store_code"].map(store_stats)
    te["store_avg_sales"] = te["store_code"].map(store_stats).fillna(global_mean)

    tr["price_rank_in_category"] = tr.groupby("product_category")["product_price"].rank(pct=True)
    cat_price_arrays = {c: np.sort(g["product_price"].values) for c, g in tr.groupby("product_category")}

    def rank_lookup(row):
        arr = cat_price_arrays.get(row["product_category"])
        return bisect.bisect_left(arr, row["product_price"]) / len(arr) if arr is not None and len(arr) else 0.5

    te["price_rank_in_category"] = te.apply(rank_lookup, axis=1)

    cat_cols = [
        "fat_content", "product_category", "store_size", "store_location_tier",
        "store_format", "store_age_bucket",
    ]
    for c in cat_cols:
        tr[c] = tr[c].astype("category")
        te[c] = pd.Categorical(te[c], categories=tr[c].cat.categories)

    feature_cols = [
        "product_weight_kg", "fat_content", "shelf_visibility", "product_category",
        "product_price", "store_age_years", "store_size", "store_location_tier",
        "store_format", "price_per_kg", "store_age_bucket", "store_avg_sales",
        "category_avg_sales", "price_rank_in_category", "product_code_te", "store_code_te",
    ]
    return tr, te, feature_cols, cat_cols


def rmse(a, b):
    return mean_squared_error(a, b) ** 0.5


def baseline_cv_score(train_full, kf):
    scores = []
    for tr_idx, va_idx in kf.split(train_full):
        tr_fold, va_fold = train_full.iloc[tr_idx], train_full.iloc[va_idx]
        grp_mean = tr_fold.groupby(["store_code", "product_category"])["total_sales"].mean()
        global_mean = tr_fold["total_sales"].mean()
        pred = va_fold.set_index(["store_code", "product_category"]).index.map(grp_mean).to_numpy()
        pred = np.where(pd.isna(pred), global_mean, pred)
        scores.append(rmse(va_fold["total_sales"], pred))
    return np.mean(scores)


def tune_models(X, y, X_xgb_all, cat_cols, kf, n_trials=25):
    folds = list(kf.split(X))

    def lgb_cv_score(params):
        scores = []
        for tr_idx, va_idx in folds:
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            m = lgb.LGBMRegressor(n_estimators=3000, random_state=42, verbosity=-1, **params)
            m.fit(X_tr, y_tr, categorical_feature=cat_cols, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(80, verbose=False)])
            scores.append(rmse(y_va, m.predict(X_va)))
        return np.mean(scores)

    def xgb_cv_score(params):
        scores = []
        for tr_idx, va_idx in folds:
            X_tr, X_va = X_xgb_all.iloc[tr_idx], X_xgb_all.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            m = xgb.XGBRegressor(n_estimators=3000, random_state=42, eval_metric="rmse",
                                  early_stopping_rounds=80, **params)
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            scores.append(rmse(y_va, m.predict(X_va)))
        return np.mean(scores)

    def lgb_objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
        return lgb_cv_score(params)

    def xgb_objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
        return xgb_cv_score(params)

    study_lgb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study_lgb.optimize(lgb_objective, n_trials=n_trials)

    study_xgb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study_xgb.optimize(xgb_objective, n_trials=n_trials)

    return study_lgb.best_params, study_lgb.best_value, study_xgb.best_params, study_xgb.best_value


def find_blend_weight(X, y, X_xgb_all, cat_cols, lgb_params, xgb_params, kf):
    oof_lgb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        m_lgb = lgb.LGBMRegressor(n_estimators=3000, random_state=42, verbosity=-1, **lgb_params)
        m_lgb.fit(X_tr, y_tr, categorical_feature=cat_cols, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(80, verbose=False)])
        oof_lgb[va_idx] = m_lgb.predict(X_va)

        X_tr_x, X_va_x = X_xgb_all.iloc[tr_idx], X_xgb_all.iloc[va_idx]
        m_xgb = xgb.XGBRegressor(n_estimators=3000, random_state=42, eval_metric="rmse",
                                  early_stopping_rounds=80, **xgb_params)
        m_xgb.fit(X_tr_x, y_tr, eval_set=[(X_va_x, y_va)], verbose=False)
        oof_xgb[va_idx] = m_xgb.predict(X_va_x)

    best_w, best_score = 0.5, float("inf")
    for w in np.arange(0, 1.01, 0.05):
        score = rmse(y, w * oof_lgb + (1 - w) * oof_xgb)
        if score < best_score:
            best_score, best_w = score, w
    return best_w, best_score


def train_final_and_predict(X, y, X_test, X_xgb_all, X_test_xgb, cat_cols, lgb_params, xgb_params, blend_w_lgb):
    X_tr_h, X_ho_h, y_tr_h, y_ho_h = train_test_split(X, y, test_size=0.08, random_state=7)
    X_tr_hx, X_ho_hx = X_xgb_all.loc[X_tr_h.index], X_xgb_all.loc[X_ho_h.index]

    m_lgb_h = lgb.LGBMRegressor(n_estimators=3000, random_state=42, verbosity=-1, **lgb_params)
    m_lgb_h.fit(X_tr_h, y_tr_h, categorical_feature=cat_cols, eval_set=[(X_ho_h, y_ho_h)],
                callbacks=[lgb.early_stopping(80, verbose=False)])
    best_lgb_n = m_lgb_h.best_iteration_

    m_xgb_h = xgb.XGBRegressor(n_estimators=3000, random_state=42, eval_metric="rmse",
                                early_stopping_rounds=80, **xgb_params)
    m_xgb_h.fit(X_tr_hx, y_tr_h, eval_set=[(X_ho_hx, y_ho_h)], verbose=False)
    best_xgb_n = m_xgb_h.best_iteration

    final_lgb = lgb.LGBMRegressor(n_estimators=best_lgb_n, random_state=42, verbosity=-1, **lgb_params)
    final_lgb.fit(X, y, categorical_feature=cat_cols)

    final_xgb = xgb.XGBRegressor(n_estimators=best_xgb_n, random_state=42, **xgb_params)
    final_xgb.fit(X_xgb_all, y)

    pred_lgb_test = final_lgb.predict(X_test)
    pred_xgb_test = final_xgb.predict(X_test_xgb)
    pred_blend = blend_w_lgb * pred_lgb_test + (1 - blend_w_lgb) * pred_xgb_test
    pred_blend = np.clip(pred_blend, 0, None)  

    return pred_blend, final_lgb


def main(run_tuning=True, n_trials=25):
    print("Loading data...")
    train, test, sample_sub = load_data()

    print("Cleaning data...")
    train_c = base_clean(train)
    test_c = base_clean(test)

    print("Running EDA (saving eda_overview.png, eda_detail.png)...")
    run_eda(train_c)

    print("Building features (leak-safe target encoding)...")
    train_full, test_full, feature_cols, cat_cols = build_features(train_c, test_c)

    X = train_full[feature_cols]
    y = train_full["total_sales"]
    X_test = test_full[feature_cols]

    X_xgb_all = X.copy()
    X_test_xgb = X_test.copy()
    for c in cat_cols:
        cats = X[c].cat.categories
        X_xgb_all[c] = X_xgb_all[c].cat.codes
        X_test_xgb[c] = pd.Categorical(X_test_xgb[c], categories=cats).codes

    kf = KFold(n_splits=5, shuffle=True, random_state=1)

    print("Baseline RMSE:", round(baseline_cv_score(train_full, kf), 2))

    if run_tuning:
        print(f"Tuning hyperparameters with Optuna ({n_trials} trials each)...")
        lgb_params, lgb_rmse, xgb_params, xgb_rmse = tune_models(X, y, X_xgb_all, cat_cols, kf, n_trials)
        print("Best LightGBM CV RMSE:", round(lgb_rmse, 2), lgb_params)
        print("Best XGBoost CV RMSE :", round(xgb_rmse, 2), xgb_params)
    else:
        lgb_params = {
            "learning_rate": 0.023018771752866554, "num_leaves": 15, "min_child_samples": 19,
            "subsample": 0.662099453989276, "colsample_bytree": 0.6220889248943965,
            "reg_alpha": 1.9273255357762828, "reg_lambda": 8.879648068510113,
        }
        xgb_params = {
            "learning_rate": 0.010524529328229023, "max_depth": 4, "min_child_weight": 14,
            "subsample": 0.7738289957189959, "colsample_bytree": 0.927635009398278,
            "reg_alpha": 0.004686895213520136, "reg_lambda": 0.9537636278761616,
        }

    print("Finding best blend weight...")
    blend_w_lgb, blend_rmse = find_blend_weight(X, y, X_xgb_all, cat_cols, lgb_params, xgb_params, kf)
    print(f"Best blend: {blend_w_lgb:.2f} LightGBM / {1 - blend_w_lgb:.2f} XGBoost -> RMSE {blend_rmse:.2f}")

    print("Training final model on full data and predicting test set...")
    pred_blend, final_lgb = train_final_and_predict(
        X, y, X_test, X_xgb_all, X_test_xgb, cat_cols, lgb_params, xgb_params, blend_w_lgb
    )

    submission = pd.DataFrame({"id": test_full["id"], "total_sales": pred_blend})

    assert list(submission.columns) == list(sample_sub.columns)
    assert len(submission) == len(sample_sub)
    assert set(submission["id"]) == set(sample_sub["id"])
    assert submission["total_sales"].isna().sum() == 0
    assert (submission["total_sales"] >= 0).all()

    submission.to_csv("submission.csv", index=False)
    print("submission.csv written:", submission.shape)


if __name__ == "__main__":
    main(run_tuning=True, n_trials=25)
