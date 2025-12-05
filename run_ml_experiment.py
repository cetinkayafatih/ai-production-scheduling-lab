from src.simulation import generate_meta_dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


def main():
    # 1) Meta-dataset üret: her satır = 1 senaryo
    print("📦 Generating meta-dataset...")
    records = generate_meta_dataset(n_instances=600, seed=42)
    df = pd.DataFrame(records)

    feature_cols = [
        "n_jobs",
        "total_pt",
        "avg_pt",
        "std_pt",
        "avg_dd",
        "std_dd",
        "avg_weight",
        "std_weight",
        "due_tightness",
    ]

    X = df[feature_cols]
    y = df["best_policy"]

    # 2) Train / test böl
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.25, random_state=42, stratify=y
    )

    # 3) Modeli eğit
    print("🧠 Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # 4) Sınıflandırma performansı
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\n✅ Classification accuracy:", round(acc, 4))
    print("\n📋 Classification report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 5) TWT açısından politika karşılaştırması
    # Statik kuralların ortalama TWT'si
    fifo_mean = df_test["fifo_twt"].mean()
    spt_mean = df_test["spt_twt"].mean()
    edd_mean = df_test["edd_twt"].mean()

    # ML policy: modelin seçtiği kuralın TWT'si
    def twt_of_policy_row(row, policy_name: str) -> float:
        if policy_name == "FIFO":
            return row["fifo_twt"]
        elif policy_name == "SPT":
            return row["spt_twt"]
        elif policy_name == "EDD":
            return row["edd_twt"]
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

    ml_twt_values = []
    oracle_twt_values = []

    # df_test'in index'leri orijinal df'den geldiği için (örneğin 507, 893 vs.),
    # y_pred ise 0..len-1 aralığında. Bu nedenle enumerate + reset_index kullanıyoruz.
    df_test_reset = df_test.reset_index(drop=True)
    for i, row in df_test_reset.iterrows():
        pred_policy = y_pred[i]  # aynı sıradaki tahmin
        ml_twt_values.append(twt_of_policy_row(row, pred_policy))

        # Oracle: üçü arasından minimum TWT
        oracle_twt_values.append(
            min(row["fifo_twt"], row["spt_twt"], row["edd_twt"])
        )

    ml_mean = sum(ml_twt_values) / len(ml_twt_values)
    oracle_mean = sum(oracle_twt_values) / len(oracle_twt_values)

    print("\n📊 Average Total Weighted Tardiness (TWT) on test set:")
    print(f"- FIFO (static):  {fifo_mean:.2f}")
    print(f"- SPT  (static):  {spt_mean:.2f}")
    print(f"- EDD  (static):  {edd_mean:.2f}")
    print(f"- ML meta-policy: {ml_mean:.2f}")
    print(f"- Oracle (best of 3): {oracle_mean:.2f}")


if __name__ == "__main__":
    main()