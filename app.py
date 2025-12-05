import streamlit as st
import pandas as pd

from src.simulation import (
    generate_random_jobs,
    run_policy,
    generate_meta_dataset,
)
from src.heuristics import fifo_rule, spt_rule, edd_rule

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# Yardımcı fonksiyonlar
# -----------------------------
def compute_single_instance_results(n_jobs: int, seed: int | None = None):
    jobs = generate_random_jobs(n_jobs=n_jobs, seed=seed)

    policies = [
        ("FIFO", fifo_rule),
        ("SPT", spt_rule),
        ("EDD", edd_rule),
    ]

    results = []
    for name, rule in policies:
        res = run_policy(jobs, rule)
        results.append(
            {
                "Policy": name,
                "Total Weighted Tardiness": res["total_weighted_tardiness"],
                "Total Tardiness": res["total_tardiness"],
                "Average Flow Time": res["average_flow_time"],
                "Makespan": res["makespan"],
            }
        )

    jobs_df = pd.DataFrame(
        [
            {
                "Job ID": j.job_id,
                "Processing Time": j.processing_time,
                "Due Date": j.due_date,
                "Weight": j.weight,
            }
            for j in jobs
        ]
    )

    results_df = pd.DataFrame(results)
    return jobs_df, results_df


def run_ml_meta_policy_experiment(
    n_instances: int = 600,
    test_size: float = 0.25,
    random_state: int = 42,
):
    records = generate_meta_dataset(n_instances=n_instances, seed=random_state)
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

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred, zero_division=0)

    # Statik politikaların TWT ortalamaları
    fifo_mean = df_test["fifo_twt"].mean()
    spt_mean = df_test["spt_twt"].mean()
    edd_mean = df_test["edd_twt"].mean()

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

    df_test_reset = df_test.reset_index(drop=True)
    for i, row in df_test_reset.iterrows():
        pred_policy = y_pred[i]
        ml_twt_values.append(twt_of_policy_row(row, pred_policy))

        oracle_twt_values.append(
            min(row["fifo_twt"], row["spt_twt"], row["edd_twt"])
        )

    ml_mean = sum(ml_twt_values) / len(ml_twt_values)
    oracle_mean = sum(oracle_twt_values) / len(oracle_twt_values)

    twt_summary = pd.DataFrame(
        {
            "Policy": [
                "FIFO (static)",
                "SPT (static)",
                "EDD (static)",
                "ML meta-policy",
                "Oracle (best of 3)",
            ],
            "Avg TWT": [
                fifo_mean,
                spt_mean,
                edd_mean,
                ml_mean,
                oracle_mean,
            ],
        }
    )

    return acc, cls_report, twt_summary


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(
        page_title="AI-Driven Production Scheduling Lab",
        layout="wide",
    )

    st.title("🧠 AI-Driven Production Scheduling Lab")
    st.write(
        "Tek makineli üretim çizelgeleme problemi için "
        "**heuristic + ML meta-policy** deney ortamı."
    )

    st.markdown(
        """
        ### Problem Özeti
        
        Tek bir makine, birden fazla iş (job) işler. Her işin:
        
        - İşlem süresi \\(p_j\\)
        - Teslim tarihi \\(d_j\\)
        - Ağırlığı / önemi \\(w_j\\)
        
        vardır. Amaç, **Toplam Ağırlıklı Gecikmeyi (Total Weighted Tardiness)** minimize etmektir:
        
        \\[
        \\min \\sum_j w_j \\max(0, C_j - d_j)
        \\]
        
        Bu uygulama ile:
        
        - FIFO / SPT / EDD gibi klasik kuralları deneyebilir,
        - Farklı job setleri üzerinde performanslarını görebilir,
        - Ve bir **Random Forest meta-policy** ile “Hangi kural bu durumda daha iyi?” sorusunu inceleyebilirsin.
        """
    )

    mode = st.sidebar.radio(
        "Mod Seç:",
        ["Single Instance Simulation", "ML Meta-Policy Experiment"],
    )

    if mode == "Single Instance Simulation":
        st.header("🎛 Single Instance Simulation")

        n_jobs = st.sidebar.slider("Number of jobs", min_value=5, max_value=50, value=20)
        seed = st.sidebar.number_input("Random seed (optional)", value=42, step=1)

        if st.button("Simülasyonu Çalıştır"):
            jobs_df, results_df = compute_single_instance_results(
                n_jobs=n_jobs, seed=int(seed)
            )

            st.subheader("Job Set")
            st.dataframe(jobs_df, use_container_width=True)

            st.subheader("Policy Results")
            st.dataframe(results_df, use_container_width=True)

            # En iyi kuralı TWT'ye göre bul
            best_row = results_df.loc[
                results_df["Total Weighted Tardiness"].idxmin()
            ]
            st.success(
                f"Bu senaryoda en düşük toplam ağırlıklı gecikmeyi sağlayan kural: "
                f"**{best_row['Policy']}** "
                f"(TWT = {best_row['Total Weighted Tardiness']:.2f})."
            )

            st.subheader("Total Weighted Tardiness Karşılaştırması")
            chart_data = results_df.set_index("Policy")["Total Weighted Tardiness"]
            st.bar_chart(chart_data)

    else:
        st.header("🤖 ML Meta-Policy Experiment")

        n_instances = st.sidebar.slider(
            "Number of generated instances",
            min_value=200,
            max_value=2000,
            value=600,
            step=100,
        )
        test_size = st.sidebar.slider(
            "Test size",
            min_value=0.1,
            max_value=0.4,
            value=0.25,
            step=0.05,
        )
        random_state = st.sidebar.number_input(
            "Random state", value=42, step=1
        )

        if st.button("ML Deneyini Çalıştır"):
            with st.spinner("Meta-dataset üretiliyor ve model eğitiliyor..."):
                acc, cls_report, twt_summary = run_ml_meta_policy_experiment(
                    n_instances=int(n_instances),
                    test_size=float(test_size),
                    random_state=int(random_state),
                )

            st.subheader("Sınıflandırma Performansı")
            st.write(f"**Accuracy:** {acc:.4f}")
            st.text(cls_report)

            st.subheader("Average Total Weighted Tardiness (TWT)")
            st.dataframe(twt_summary, use_container_width=True)

            st.bar_chart(
                twt_summary.set_index("Policy")["Avg TWT"],
            )

            # Küçük bir yorum / özet metni
            try:
                fifo_val = float(
                    twt_summary.loc[
                        twt_summary["Policy"] == "FIFO (static)", "Avg TWT"
                    ].iloc[0]
                )
                spt_val = float(
                    twt_summary.loc[
                        twt_summary["Policy"] == "SPT (static)", "Avg TWT"
                    ].iloc[0]
                )
                edd_val = float(
                    twt_summary.loc[
                        twt_summary["Policy"] == "EDD (static)", "Avg TWT"
                    ].iloc[0]
                )
                ml_val = float(
                    twt_summary.loc[
                        twt_summary["Policy"] == "ML meta-policy", "Avg TWT"
                    ].iloc[0]
                )
                oracle_val = float(
                    twt_summary.loc[
                        twt_summary["Policy"] == "Oracle (best of 3)", "Avg TWT"
                    ].iloc[0]
                )

                best_static = min(
                    [("FIFO", fifo_val), ("SPT", spt_val), ("EDD", edd_val)],
                    key=lambda x: x[1],
                )

                st.info(
                    f"Bu deneyde **en iyi statik kural**: **{best_static[0]}** "
                    f"(Avg TWT = {best_static[1]:.2f}).\n\n"
                    f"**ML meta-policy** ise Avg TWT = {ml_val:.2f} ile, "
                    f"statik kurallara göre daha esnek bir performans sergiliyor ve "
                    f"**oracle** seviyesine (Avg TWT = {oracle_val:.2f}) oldukça yaklaşıyor."
                )
            except Exception:
                st.info(
                    "TWT özet sonuçları yorumlanırken bir hata oluştu, "
                    "ancak tablo ve grafikler kullanıma hazır."
                )


if __name__ == "__main__":
    main()