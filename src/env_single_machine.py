from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Job:
    job_id: int
    processing_time: float
    due_date: float
    weight: float = 1.0
    release_time: float = 0.0  # istersek ileride kullanırız


class SingleMachineEnv:
    """
    Basit tek makine çizelgeleme ortamı.
    Her adımda bir job seçiyorsun, makine onu işliyor, zaman ilerliyor.
    """

    def __init__(self, jobs: List[Job]):
        self.initial_jobs: List[Job] = jobs
        self.reset()

    def reset(self):
        self.time: float = 0.0
        self.completed_jobs: List[Dict] = []
        self.remaining_jobs: List[Job] = list(self.initial_jobs)

    def is_done(self) -> bool:
        return len(self.remaining_jobs) == 0

    def step(self, job_id: int) -> float:
        """
        Seçilen job_id'yi işle, zamanı ilerlet, tardiness hesapla.
        Geriye "weighted_tardiness" (cost) döner.
        """
        job = next(j for j in self.remaining_jobs if j.job_id == job_id)
        self.remaining_jobs = [j for j in self.remaining_jobs if j.job_id != job_id]

        start_time = self.time
        completion_time = start_time + job.processing_time
        self.time = completion_time

        tardiness = max(0.0, completion_time - job.due_date)
        weighted_tardiness = job.weight * tardiness

        self.completed_jobs.append(
            {
                "job_id": job.job_id,
                "start_time": start_time,
                "processing_time": job.processing_time,
                "completion_time": completion_time,
                "due_date": job.due_date,
                "weight": job.weight,
                "tardiness": tardiness,
                "weighted_tardiness": weighted_tardiness,
            }
        )

        return weighted_tardiness