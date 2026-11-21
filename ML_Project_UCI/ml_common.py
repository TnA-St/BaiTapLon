"""Cau hinh du lieu UCI dung chung giua train va Streamlit."""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'data' / 'student-por.csv'
MODEL_DIR = ROOT / 'models'
REPORT_DIR = ROOT / 'reports'
SOURCE_URL = 'https://archive.ics.uci.edu/dataset/320/student+performance'

NUMERIC_FEATURES = ['studytime', 'failures', 'absences', 'G1', 'G2']
CATEGORY_FEATURES = ['internet', 'activities', 'schoolsup', 'famsup']
FEATURES = NUMERIC_FEATURES + CATEGORY_FEATURES
TARGET_GRADE = 'G3'
PASS_THRESHOLD = 10


def read_data():
    """Doc CSV UCI dung dau phan cach ';', kiem tra cau truc."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f'Khong tim thay {DATA_PATH}. Hay kiem tra file data/student-por.csv trong thu muc du an.'
        )
    df = pd.read_csv(DATA_PATH, sep=';')
    required = FEATURES + [TARGET_GRADE]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f'Thieu cot: {missing}. Hay dung student-por.csv goc cua UCI '
            '(phan tach bang dau cham phay).'
        )
    for col in NUMERIC_FEATURES + [TARGET_GRADE]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if df[TARGET_GRADE].isna().any():
        raise ValueError('Cot G3 co gia tri rong/khong phai so.')
    df['pass_status'] = df[TARGET_GRADE].ge(PASS_THRESHOLD).map({True:'Pass',False:'Fail'})
    return df


def make_preprocessor():
    numeric = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('standardize', StandardScaler()),
    ])
    categories = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer([
        ('numeric', numeric, NUMERIC_FEATURES),
        ('category', categories, CATEGORY_FEATURES),
    ], sparse_threshold=0.0)
