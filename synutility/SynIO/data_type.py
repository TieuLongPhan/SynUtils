import json
import pickle
import numpy as np
from numpy import ndarray
from joblib import dump, load
from typing import List, Dict, Any
from synutility.SynIO.debug import setup_logging

logger = setup_logging()


def save_database(database: list[dict], pathname: str = "./Data/database.json") -> None:
    """
    Save a database (a list of dictionaries) to a JSON file.

    Args:
        database: The database to be saved.
        pathname: The path where the database will be saved.
                    Defaults to './Data/database.json'.

    Raises:
        TypeError: If the database is not a list of dictionaries.
        ValueError: If there is an error writing the file.
    """
    if not all(isinstance(item, dict) for item in database):
        raise TypeError("Database should be a list of dictionaries.")

    try:
        with open(pathname, "w") as f:
            json.dump(database, f)
    except IOError as e:
        raise ValueError(f"Error writing to file {pathname}: {e}")


def load_database(pathname: str = "./Data/database.json") -> List[Dict]:
    """
    Load a database (a list of dictionaries) from a JSON file.

    Args:
        pathname: The path from where the database will be loaded.
                    Defaults to './Data/database.json'.

    Returns:
        The loaded database.

    Raises:
        ValueError: If there is an error reading the file.
    """
    try:
        with open(pathname, "r") as f:
            database = json.load(f)  # Load the JSON data from the file
        return database
    except IOError as e:
        raise ValueError(f"Error reading to file {pathname}: {e}")


def save_to_pickle(data: List[Dict[str, Any]], filename: str) -> None:
    """
    Save a list of dictionaries to a pickle file.

    Parameters:
    data (List[Dict[str, Any]]): A list of dictionaries to be saved.
    filename (str): The name of the file where the data will be saved.
    """
    with open(filename, "wb") as file:
        pickle.dump(data, file)


def load_from_pickle(filename: str) -> List[Any]:
    """
    Load data from a pickle file.

    Parameters:
    filename (str): The name of the pickle file to load data from.

    Returns:
    List[Any]: The data loaded from the pickle file.
    """
    with open(filename, "rb") as file:
        return pickle.load(file)


def load_gml_as_text(gml_file_path):
    """
    Load the contents of a GML file as a text string.

    Parameters:
    - gml_file_path (str): The file path to the GML file.

    Returns:
    - str: The text content of the GML file.
    """
    try:
        with open(gml_file_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        print(f"File not found: {gml_file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def save_compressed(array: ndarray, filename: str) -> None:
    """
    Saves a NumPy array in a compressed format using .npz extension.

    Parameters:
    - array (ndarray): The NumPy array to be saved.
    - filename (str): The file path or name to save the array to,
    with a '.npz' extension.

    Returns:
    - None: This function does not return any value.
    """
    np.savez_compressed(filename, array=array)


def load_compressed(filename: str) -> ndarray:
    """
    Loads a NumPy array from a compressed .npz file.

    Parameters:
    - filename (str): The path of the .npz file to load.

    Returns:
    - ndarray: The loaded NumPy array.

    Raises:
    - KeyError: If the .npz file does not contain an array with the key 'array'.
    """
    with np.load(filename) as data:
        if "array" in data:
            return data["array"]
        else:
            raise KeyError(
                "The .npz file does not contain" + " an array with the key 'array'."
            )


def save_model(model: Any, filename: str) -> None:
    """
    Save a machine learning model to a file using joblib.

    Parameters:
    - model (Any): The machine learning model to save.
    - filename (str): The path to the file where the model will be saved.
    """
    dump(model, filename)
    logger.info(f"Model saved successfully to {filename}")


def load_model(filename: str) -> Any:
    """
    Load a machine learning model from a file using joblib.

    Parameters:
    - filename (str): The path to the file from which the model will be loaded.

    Returns:
    - Any: The loaded machine learning model.
    """
    model = load(filename)
    logger.info(f"Model loaded successfully from {filename}")
    return model


def save_dict_to_json(data: dict, file_path: str) -> None:
    """
    Save a dictionary to a JSON file.

    Parameters:
    -----------
    data : dict
        The dictionary to be saved.

    file_path : str
        The path to the file where the dictionary should be saved.
        Make sure the file has a .json extension.

    Returns:
    --------
    None
    """
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    logger.info(f"Dictionary successfully saved to {file_path}")


def load_dict_from_json(file_path: str) -> dict:
    """
    Load a dictionary from a JSON file.

    Parameters:
    -----------
    file_path : str
        The path to the JSON file from which to load the dictionary.
        Make sure the file has a .json extension.

    Returns:
    --------
    dict
        The dictionary loaded from the JSON file.
    """
    try:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        logger.info(f"Dictionary successfully loaded from {file_path}")
        return data
    except Exception as e:
        logger.error(e)
        return None
