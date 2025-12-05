from typing import List
from env_single_machine import Job


def fifo_rule(jobs: List[Job]) -> int:
    """
    First In First Out:
    Burada basitçe job_id sırasına göre seçiyoruz
    (gerçekte arrival time'a göre de düzenleyebilirsin).
    """
    return sorted(jobs, key=lambda j: j.job_id)[0].job_id


def spt_rule(jobs: List[Job]) -> int:
    """Shortest Processing Time."""
    return sorted(jobs, key=lambda j: j.processing_time)[0].job_id


def edd_rule(jobs: List[Job]) -> int:
    """Earliest Due Date."""
    return sorted(jobs, key=lambda j: j.due_date)[0].job_id