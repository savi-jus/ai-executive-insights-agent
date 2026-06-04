"""Dataset profiling helpers used before AI analysis."""


def profile_data(df):
    """Build a lightweight summary of shape, types, and missing values."""
    profile = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.astype(str).to_dict(),
    }

    return profile
