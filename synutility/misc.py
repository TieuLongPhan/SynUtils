import random
from typing import Dict, List, Optional
from datetime import datetime


def stratified_random_sample(
    data: List[Dict[str, any]],
    property_key: str,
    samples_per_class: int = 1,
    seed: Optional[int] = 42,
    bypass: bool = False,
) -> List[Dict[str, any]]:
    """
    Stratifies and samples data from a list of dictionaries based on a
    specified property key.

    Parameters:
    - data (List[Dict[str, any]]): The data to sample from, a list of dictionaries.
    - property_key (str): The key in the dictionaries to stratify by.
    - samples_per_class (int): The number of samples to take from each class.
    Defaults to 1.
    - seed (Optional[int], optional): The seed for the random number generator
    for reproducibility. Defaults to 42.
    - bypass (bool, optional): If True, classes with fewer than
    `samples_per_class` entries will be skipped without raising an error.
    Defaults to False.

    Returns:
    - List[Dict[str, any]]: A list of sampled dictionaries, where each entry corresponds
    to a sampled item.

    Raises:
    - ValueError: If a class has fewer than `samples_per_class` entries
    and `bypass` is False.
    """

    if seed is not None:
        random.seed(seed)

    stratified_data = {}
    for item in data:
        key = item.get(property_key)
        if key is None:
            continue  # Exclude data items where the specified key is not present
        if key in stratified_data:
            stratified_data[key].append(item)
        else:
            stratified_data[key] = [item]

    sampled_data = []
    for key, items in stratified_data.items():
        class_size = len(items)
        if class_size >= samples_per_class:
            sampled_data.extend(random.sample(items, samples_per_class))
        elif bypass:
            continue  # Skip this group entirely if not enough data and bypass is True
        else:
            raise ValueError(
                f"Not enough data to sample {samples_per_class} items for class '{key}', "
                f"only {class_size} available."
            )

    return sampled_data


def calculate_processing_time(start_time_str: str, end_time_str: str) -> float:
    """
    Calculates the processing time in seconds between two timestamps.

    Parameters:
    - start_time_str (str): A string representing the start time in the format
    'YYYY-MM-DD HH:MM:SS,fff'.
    - end_time_str (str): A string representing the end time in the same format as
    start_time_str.

    Returns:
    - float: The duration between the start and end time in seconds.

    Raises:
    - ValueError: If the input strings do not match the expected format.
    """
    datetime_format = "%Y-%m-%d %H:%M:%S,%f"

    start_time = datetime.strptime(start_time_str, datetime_format)
    end_time = datetime.strptime(end_time_str, datetime_format)

    duration = end_time - start_time

    return duration.total_seconds()
