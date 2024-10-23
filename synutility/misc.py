import random
from typing import Dict, List


def stratified_random_sample(
    data: List[Dict], property_key: str, samples_per_class: int, seed: int = None
) -> List[Dict]:
    """
    Stratifies and samples data from a list of dictionaries based on a specified property.

    Parameters:
    - data (List[Dict]): The data to sample from, a list of dictionaries.
    - property_key (str): The key in the dictionaries to stratify by.
    - samples_per_class (int): The number of samples to take from each class.
    - seed (int): The seed for the random number generator for reproducibility.

    Returns:
    - List[Dict]: A list of sampled dictionaries.
    """

    if seed is not None:
        random.seed(seed)

    # Group data by the specified property
    stratified_data = {}
    for item in data:
        key = item.get(property_key)
        if key in stratified_data:
            stratified_data[key].append(item)
        else:
            stratified_data[key] = [item]

    # Sample data from each group
    sampled_data = []
    for key, items in stratified_data.items():
        if len(items) >= samples_per_class:
            sampled_data.extend(random.sample(items, samples_per_class))
        else:
            raise ValueError(
                f"Not enough data to sample {samples_per_class} items for class {key}"
            )

    return sampled_data
