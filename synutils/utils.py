import os
import json
import logging
import warnings
import numpy as np
from numpy import ndarray
from rdkit import rdBase
from joblib import dump, load
from typing import Any


def setup_logging(log_level: str = "INFO", log_filename: str = None) -> logging.Logger:
    """
    Configures the logging for an application. It sets up logging to either the console
    or a file based on whether a log filename is provided. The function adjusts the
    logging level dynamically according to the specified log level.

    Parameters
    ----------
    log_level : str, optional
        Specifies the logging level. Accepted values are 'DEBUG', 'INFO', 'WARNING',
        'ERROR', 'CRITICAL'. Default is 'INFO'.
    log_filename : str, optional
        Specifies the filename of the log file. If provided, logs will be written
        to this file. If None, logs will be written to the console. Default is None.

    Returns
    -------
    logging.Logger
        The configured logger object.

    Raises
    ------
    ValueError
        If the specified log_level is not recognized as a valid logging level.
    """
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    if log_filename:
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        logging.basicConfig(
            level=numeric_level, format=log_format, filename=log_filename, filemode="w"
        )
    else:
        logging.basicConfig(level=numeric_level, format=log_format)

    return logger


def configure_warnings_and_logs(
    ignore_warnings: bool = False, disable_rdkit_logs: bool = False
) -> None:
    """
    Configures warning and logging behaviors for Python and RDKit. This function allows
    selective suppression of Python warnings and disabling of RDKit error and warning logs
    based on the parameters provided.

    Parameters
    ----------
    ignore_warnings : bool, optional
        If True, suppresses all Python warnings. If False, normal operation of warnings.
        Default is False.
    disable_rdkit_logs : bool, optional
        If True, disables RDKit error and warning logs. If False, logs operate normally.
        Default is False.

    Usage
    -----
    This function should be used at the start of scripts where control over warning and
    logging verbosity is needed. It is particularly useful in production or testing phases
    to reduce output clutter but should be used cautiously during development to
    avoid overlooking important warnings or errors.
    """
    if ignore_warnings:
        # Suppress all Python warnings (e.g., DeprecationWarning, RuntimeWarning)
        warnings.filterwarnings("ignore")
    else:
        # Reset the warnings to default behavior (i.e., printing all warnings)
        warnings.resetwarnings()

    if disable_rdkit_logs:
        # Disable RDKit error and warning logs
        rdBase.DisableLog("rdApp.error")
        rdBase.DisableLog("rdApp.warning")
    else:
        # Enable RDKit error and warning logs if they were previously disabled
        rdBase.EnableLog("rdApp.error")
        rdBase.EnableLog("rdApp.warning")


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
    logging.info(f"Model saved successfully to {filename}")


def load_model(filename: str) -> Any:
    """
    Load a machine learning model from a file using joblib.

    Parameters:
    - filename (str): The path to the file from which the model will be loaded.

    Returns:
    - Any: The loaded machine learning model.
    """
    model = load(filename)
    logging.info(f"Model loaded successfully from {filename}")
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

    logging.info(f"Dictionary successfully saved to {file_path}")


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
        logging.info(f"Dictionary successfully loaded from {file_path}")
        return data
    except Exception as e:
        logging.error(e)
        return None
