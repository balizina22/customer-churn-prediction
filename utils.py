import pandas as pd

def build_input_from_dict(data, feature_names):
    df = pd.DataFrame([data])
    
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    return df[feature_names]