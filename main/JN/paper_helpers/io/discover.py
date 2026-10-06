import os
from typing import List, Dict

import pandas as pd


def find_data_obj_files(start_dir: str, target_filename: str = "data_obj.npy") -> List[str]:
    """
    Recursively find all files with the specified filename under a root directory.

    The function walks the directory tree starting at ``start_dir`` and collects
    the full paths to any files named ``target_filename``.

    Parameters
    ----------
    start_dir : str
        Root directory to start the search from.
    target_filename : str, optional
        Name of the file to search for (default is ``"data_obj.npy"``).

    Returns
    -------
    List[str]
        A list of absolute or relative paths (depending on ``start_dir``)
        to files matching ``target_filename``.

    Raises
    ------
    FileNotFoundError
        If ``start_dir`` does not exist. ``os.walk`` would otherwise return no
        matches silently, which looks identical to "the directory is empty".
    """
    if not os.path.isdir(start_dir):
        raise FileNotFoundError(
            f"Search root does not exist: {start_dir!r} "
            f"(resolved to {os.path.abspath(start_dir)}). "
            "Generated artifacts are missing -- see main/README.md for the commands "
            "that create dataObj_v2/ and binn_v2_models/."
        )

    matches: List[str] = []

    for root, dirs, files in os.walk(start_dir):
        if target_filename in files:
            matches.append(os.path.join(root, target_filename))

    return matches


def paths_to_df(paths: List[str]) -> pd.DataFrame:
    """
    Convert a list of file paths into a pandas DataFrame using X_Y directory names as metadata.

    For each path, the function:
    - Stores the full path under the column ``"full_path"``.
    - Parses each component of the path; if the component contains an underscore,
      it is split into ``X`` and ``Y`` at the first underscore, and added as
      a column/value pair ``{X: Y}`` in the resulting DataFrame row.

    Parameters
    ----------
    paths : List[str]
        List of paths to ``data_obj.npy`` (or similar) files.

    Returns
    -------
    pd.DataFrame
        DataFrame where each row represents a file. The columns include:
        - ``"full_path"``: full file path.
        - Additional columns defined by directory components of the form ``X_Y``.
    """
    records = []

    for path in paths:
        row: Dict[str, str] = {"full_path": path}
        parts = os.path.normpath(path).split(os.sep)

        for part in parts:
            if "_" in part:
                try:
                    x, y = part.split("_", 1)
                    row[x] = y
                except ValueError:
                    # Skip components that do not match the expected X_Y format
                    continue

        records.append(row)

    return pd.DataFrame(records)


def condense_df(df: pd.DataFrame, filters: Dict[str, object]) -> pd.DataFrame:
    """
    Filter a DataFrame by matching specified values for given columns.

    This is typically used with DataFrames created by :func:`paths_to_df`.
    Each key–value pair in ``filters`` is applied as an equality filter on the
    corresponding column. Values are compared as strings.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame to filter.
    filters : Dict[str, object]
        Dictionary where keys are column names and values are the desired
        values to match. Values are cast to strings before comparison.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with the index reset.
    """
    filtered_df = df.copy()

    for key, value in filters.items():
        filtered_df = filtered_df[filtered_df[key] == str(value)]

    return filtered_df.reset_index(drop=True)
