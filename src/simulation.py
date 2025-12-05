import random
from heuristics import fifo_rule, spt_rule, edd_rule
from typing import List, Callable, Dict

from env_single_machine import Job, SingleMachineEnv
from metrics import (
    total_weighted_tardiness,
    total_tardiness,
    average_flow_time,
    makespan,
)


def generate_random_jobs(
    n_jobs: int,
    processing_time_range=(1, 10),
    due_date_range=(10, 50),
    weight_range=(1, 3),
    seed: int | None = None,
) -> List[Job]:
    if seed is not None:
        random.seed(seed)

    jobs: List[Job] = []
    for i in range(n_jobs):
        pt = random.randint(*processing_time_range)
        dd = random.randint(*due_date_range)
        w = random.randint(*weight_range)
        jobs.append(
            Job(
                job_id=i,
                processing_time=pt,
                due_date=dd,
                weight=w,
            )
        )
    return jobs


def run_policy(
    jobs: List[Job],
    policy_fn: Callable[[List[Job]], int],
) -> Dict:
    """
    Verilen job listesi ve politika fonksiyonu için simülasyon çalıştırır.
    policy_fn: remaining_jobs listesi verildiğinde hangi job_id'yi seçeceğini söyler.
    """
    env = SingleMachineEnv(jobs)
    total_cost = 0.0

    while not env.is_done():
        job_id = policy_fn(env.remaining_jobs)
        cost = env.step(job_id)
        total_cost += cost

    schedule = env.completed_jobs

    results = {
        "total_weighted_tardiness": total_weighted_tardiness(schedule),
        "total_tardiness": total_tardiness(schedule),
        "average_flow_time": average_flow_time(schedule),
        "makespan": makespan(schedule),
        "total_cost": total_cost,
        "schedule": schedule,
    }
    return results


def summarize_instance(jobs: List[Job]) -> Dict:
    """
    Bir job setini tek satır feature'lara özetler.
    """
    n_jobs = len(jobs)
    total_pt = sum(j.processing_time for j in jobs)
    avg_pt = total_pt / n_jobs
    avg_dd = sum(j.due_date for j in jobs) / n_jobs
    avg_weight = sum(j.weight for j in jobs) / n_jobs

    # Basit çeşitlilik ölçüleri
    pt_values = [j.processing_time for j in jobs]
    dd_values = [j.due_date for j in jobs]
    w_values = [j.weight for j in jobs]

    def _std(vals):
        m = sum(vals) / len(vals)
        return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5

    std_pt = _std(pt_values)
    std_dd = _std(dd_values)
    std_weight = _std(w_values)

    # Ortalama due date, ortalama completion süresine göre sıkı mı gevşek mi?
    # (çok negatifse due date'ler sıkı demektir)
    avg_nominal_completion = total_pt / n_jobs
    due_tightness = avg_dd - avg_nominal_completion

    return {
        "n_jobs": n_jobs,
        "total_pt": total_pt,
        "avg_pt": avg_pt,
        "std_pt": std_pt,
        "avg_dd": avg_dd,
        "std_dd": std_dd,
        "avg_weight": avg_weight,
        "std_weight": std_weight,
        "due_tightness": due_tightness,
    }


def evaluate_all_policies(jobs: List[Job]) -> Dict[str, float]:
    """
    FIFO, SPT, EDD için total weighted tardiness değerlerini döner.
    """
    res_fifo = run_policy(jobs, fifo_rule)
    res_spt = run_policy(jobs, spt_rule)
    res_edd = run_policy(jobs, edd_rule)

    return {
        "FIFO": res_fifo["total_weighted_tardiness"],
        "SPT": res_spt["total_weighted_tardiness"],
        "EDD": res_edd["total_weighted_tardiness"],
    }


def build_meta_record(jobs: List[Job]) -> Dict:
    """
    Tek bir job seti için:
    - feature'ları hesaplar
    - 3 heuristiği de dener
    - en iyi heuristiği label olarak koyar
    """
    features = summarize_instance(jobs)
    scores = evaluate_all_policies(jobs)

    best_policy = min(scores, key=scores.get)

    record = {
        **features,
        "fifo_twt": scores["FIFO"],
        "spt_twt": scores["SPT"],
        "edd_twt": scores["EDD"],
        "best_policy": best_policy,
    }
    return record


def generate_meta_dataset(
    n_instances: int = 500,
    n_jobs_range=(10, 40),
    processing_time_range=(1, 10),
    due_date_range=(15, 60),
    weight_range=(1, 3),
    seed: int | None = None,
) -> List[Dict]:
    """
    ML için dataset üretir:
    Her satır bir job setini ve onun en iyi heuristic label'ını temsil eder.
    """
    if seed is not None:
        random.seed(seed)

    records: List[Dict] = []

    for _ in range(n_instances):
        n_jobs = random.randint(*n_jobs_range)
        jobs = generate_random_jobs(
            n_jobs=n_jobs,
            processing_time_range=processing_time_range,
            due_date_range=due_date_range,
            weight_range=weight_range,
            seed=None,  # global random state kullanıyoruz
        )
        rec = build_meta_record(jobs)
        records.append(rec)

    return records