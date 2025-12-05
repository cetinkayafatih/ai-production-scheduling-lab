from typing import List, Dict


def total_weighted_tardiness(schedule: List[Dict]) -> float:
    return sum(entry["weighted_tardiness"] for entry in schedule)


def total_tardiness(schedule: List[Dict]) -> float:
    return sum(entry["tardiness"] for entry in schedule)


def average_flow_time(schedule: List[Dict]) -> float:
    """
    Flow time = completion_time - start_time
    """
    if not schedule:
        return 0.0
    total_flow = sum(
        entry["completion_time"] - entry["start_time"] for entry in schedule
    )
    return total_flow / len(schedule)


def makespan(schedule: List[Dict]) -> float:
    if not schedule:
        return 0.0
    return max(entry["completion_time"] for entry in schedule)