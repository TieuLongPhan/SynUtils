import os
import logging
import warnings
from rdkit import rdBase


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
            level=numeric_level, format=log_format, filename=log_filename, filemode="a"
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
