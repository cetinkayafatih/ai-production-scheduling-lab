from src.simulation import generate_random_jobs, run_policy
from src.heuristics import fifo_rule, spt_rule, edd_rule


def main():
    # 20 job'lu rastgele bir instance üret
    jobs = generate_random_jobs(n_jobs=20, seed=42)

    # Her kuralı sırayla çalıştır
    policies = [
        ("FIFO", fifo_rule),
        ("SPT", spt_rule),
        ("EDD", edd_rule),
    ]

    for name, rule in policies:
        res = run_policy(jobs, rule)
        print(f"{name} → TWT = {res['total_weighted_tardiness']:.2f}, "
              f"Flow Time = {res['average_flow_time']:.2f}, "
              f"Makespan = {res['makespan']:.2f}")


if __name__ == "__main__":
    main()