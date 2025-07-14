# metadata_loader.py

import pandas as pd
import os

REQUIRED_METADATA_FILES = {
    "layout": "metadata/filelayout.xlsx",
    "questions": "metadata/Survey Questions.xlsx",
    "themes": "metadata/Survey Themes.xlsx",
    "scales": "metadata/Survey Scales.xlsx",
    "demographics": "metadata/Demographics.xlsx",
    "mapping": "metadata/Unified_Column_Mapping.xlsx"
}

def load_required_metadata():
    metadata = {}
    for key, path in REQUIRED_METADATA_FILES.items():
        if not os.path.exists(path):
            raise RuntimeError(f"Missing required metadata file: {path}")
        try:
            df = pd.read_excel(path)
            df.columns = [c.upper().strip() for c in df.columns]
            metadata[key] = df
        except Exception as e:
            raise RuntimeError(f"Error loading metadata file `{path}`: {e}")
    return metadata

def validate_dataset(dataset_path, mapping_df):
    if not os.path.exists(dataset_path):
        raise RuntimeError(f"Missing main dataset file: {dataset_path}")

    try:
        sample = pd.read_csv(dataset_path, nrows=5)
        normalized_headers = [c.upper().strip() for c in sample.columns]
    except Exception as e:
        raise RuntimeError(f"Failed to read dataset headers: {e}")

    required_logical = ["QUESTION", "SURVEYR", "DEMCODE", "ANSWER1"]
    for logical_name in required_logical:
        if logical_name not in mapping_df["LOGICAL_NAME"].str.upper().values:
            raise RuntimeError(f"Missing required logical column in Unified_Column_Mapping: {logical_name}")

        mapped_column = mapping_df.loc[
            mapping_df["LOGICAL_NAME"].str.upper() == logical_name,
            "IN_DATASET"
        ].values[0].upper().strip()

        if mapped_column not in normalized_headers:
            raise RuntimeError(f"Mapped column `{mapped_column}` for `{logical_name}` not found in dataset.")

    return dataset_path
