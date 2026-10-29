import pandas as pd

def blocks(df, minutes=10):
    candidates=[c for c in df if 'timestamp' in ''.join(ch for ch in str(c).lower() if ch.isalnum())]
    if not candidates:
        raise ValueError('No Timestamp column: cannot verify temporal burst groups. No group-safe claim is made.')
    t=pd.to_datetime(df[candidates[0]],errors='coerce',format='mixed',dayfirst=True)
    if t.isna().any():
        raise ValueError(f'Cannot parse all timestamps in {candidates[0]}: {t.isna().sum()} invalid')
    return t.dt.floor(f'{minutes}min').astype(str).to_numpy()

