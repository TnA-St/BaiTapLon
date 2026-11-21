"""Huan luyen 7 mo hinh giam sat + K-Means; ghi nhat ky Epoch/Mini-batch cua MLP.

Chay:
    python train_models.py
    python train_models.py --max-epochs 180 --batch-size 200 --patience 15

Dataset student-por.csv da duoc dong goi trong data/, su dung offline.
"""
import argparse
import hashlib
from copy import deepcopy
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score,
    silhouette_score
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ml_common import (
    ROOT, MODEL_DIR, REPORT_DIR, DATA_PATH, SOURCE_URL,
    FEATURES, NUMERIC_FEATURES, PASS_THRESHOLD, make_preprocessor, read_data
)

SEED = 42


def classification_scores(y_true, predictions, name):
    return {
        'Model': name,
        'Accuracy': round(float(accuracy_score(y_true, predictions)), 4),
        'Precision': round(float(precision_score(y_true, predictions, pos_label='Pass', zero_division=0)), 4),
        'Recall': round(float(recall_score(y_true, predictions, pos_label='Pass', zero_division=0)), 4),
        'F1': round(float(f1_score(y_true, predictions, pos_label='Pass', zero_division=0)), 4),
    }


def train_mlp_epochs(X_train, y_train, X_val, y_val, max_epochs, batch_size, patience):
    """Dung partial_fit de theo doi chinh xac tung epoch va cac mini-batch.

    Preprocessing CHI fit tren train_no_val de tranh leakage validation.
    Model tot nhat duoc chon bang validation accuracy, khong dung test set.
    """
    pre = make_preprocessor()
    X_tr = np.asarray(pre.fit_transform(X_train), dtype=np.float64)
    X_v = np.asarray(pre.transform(X_val), dtype=np.float64)
    y_tr = np.asarray(y_train)
    y_v = np.asarray(y_val)

    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16), activation='relu', solver='adam',
        learning_rate_init=0.001, random_state=SEED
    )
    rng = np.random.default_rng(SEED)
    best_accuracy = -1.0
    best_epoch = 0
    best_model = None
    stale = 0
    history = []
    classes = np.array(['Fail', 'Pass'])

    for epoch in range(1, max_epochs + 1):
        idx = rng.permutation(len(y_tr))
        batch_losses = []
        batch_sizes = []
        for start in range(0, len(idx), batch_size):
            batch_indices = idx[start: start + batch_size]
            # MLPClassifier.partial_fit cap nhat tham so sau tung lo nho.
            if epoch == 1 and start == 0:
                mlp.partial_fit(X_tr[batch_indices], y_tr[batch_indices], classes=classes)
            else:
                mlp.partial_fit(X_tr[batch_indices], y_tr[batch_indices])
            batch_losses.append(float(mlp.loss_))
            batch_sizes.append(len(batch_indices))

        train_loss = float(np.average(batch_losses, weights=batch_sizes))
        train_accuracy = float(accuracy_score(y_tr, mlp.predict(X_tr)))
        val_accuracy = float(accuracy_score(y_v, mlp.predict(X_v)))
        improved = val_accuracy > best_accuracy + 1e-4
        if improved:
            best_accuracy = val_accuracy
            best_epoch = epoch
            best_model = deepcopy(mlp)
            stale = 0
        else:
            stale += 1

        history.append({
            'Epoch': epoch,
            'Mini_batches': len(batch_sizes),
            'Samples_seen': len(y_tr),
            'Train_loss': round(train_loss, 6),
            'Train_accuracy': round(train_accuracy, 6),
            'Val_accuracy': round(val_accuracy, 6),
            'Best_val_accuracy': round(best_accuracy, 6),
            'No_improvement_epochs': stale,
        })
        if epoch == 1 or epoch % 10 == 0 or stale >= patience:
            print(f'  MLP epoch {epoch:3d} | loss {train_loss:.4f} | val accuracy {val_accuracy:.3f} | patience {stale}/{patience}')
        if stale >= patience:
            break

    fitted = Pipeline([
        ('preprocess', pre),
        ('model', best_model),
    ])
    return fitted, pd.DataFrame(history), {
        'best_epoch': int(best_epoch),
        'epochs_run': len(history),
        'best_val_accuracy': round(float(best_accuracy), 6),
        'stopped_early': len(history) < max_epochs,
        'train_size': int(len(y_tr)),
        'validation_size': int(len(y_v)),
        'batch_size': int(batch_size),
        'max_epochs': int(max_epochs),
        'patience': int(patience),
        'mini_batches_per_epoch': int((len(y_tr) + batch_size - 1) // batch_size),
    }


def main():
    parser = argparse.ArgumentParser(description='UCI Student Performance - huan luyen Machine Learning')
    parser.add_argument('--max-epochs', type=int, default=180)
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--patience', type=int, default=15)
    args = parser.parse_args()
    if min(args.max_epochs, args.batch_size, args.patience) < 1:
        parser.error('--max-epochs, --batch-size va --patience phai la so nguyen duong')

    df = read_data()
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    REPORT_DIR.mkdir(exist_ok=True, parents=True)
    X = df[FEATURES]
    y_grade = df['G3']
    y_class = df['pass_status']

    # 20% la tap TEST, chi danh gia cuoi cung, tuyet doi khong de early stopping nhin thay.
    tr_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=SEED, stratify=y_class
    )
    X_train, X_test = X.iloc[tr_idx], X.iloc[test_idx]
    y_cls_train, y_cls_test = y_class.iloc[tr_idx], y_class.iloc[test_idx]
    y_grade_train, y_grade_test = y_grade.iloc[tr_idx], y_grade.iloc[test_idx]

    reg = Pipeline([
        ('preprocess', make_preprocessor()),
        ('model', LinearRegression()),
    ])
    reg.fit(X_train, y_grade_train)
    reg_pred = reg.predict(X_test)
    regression_scores = {
        'MAE': round(float(mean_absolute_error(y_grade_test, reg_pred)), 4),
        'RMSE': round(float(np.sqrt(mean_squared_error(y_grade_test, reg_pred))), 4),
        'R2': round(float(r2_score(y_grade_test, reg_pred)), 4),
    }
    joblib.dump(reg, MODEL_DIR / 'linear_regression.joblib')
    (REPORT_DIR / 'regression_metrics.json').write_text(
        json.dumps(regression_scores, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    algorithms = {
        'Logistic Regression': LogisticRegression(max_iter=2000, random_state=SEED),
        'KNN': KNeighborsClassifier(n_neighbors=7),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=SEED),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=8, random_state=SEED),
        'SVM': SVC(C=1.0, kernel='rbf', random_state=SEED),
    }
    rows = []
    for name, estimator in algorithms.items():
        model = Pipeline([
            ('preprocess', make_preprocessor()),
            ('model', estimator),
        ])
        model.fit(X_train, y_cls_train)
        prediction = model.predict(X_test)
        rows.append(classification_scores(y_cls_test, prediction, name))
        joblib.dump(model, MODEL_DIR / f'{name.lower().replace(" ", "_")}.joblib')
        if name == 'Random Forest':
            classes = ['Fail', 'Pass']
            cm = confusion_matrix(y_cls_test, prediction, labels=classes)
            (REPORT_DIR / 'confusion_matrix.json').write_text(
                json.dumps({'labels': classes, 'matrix': cm.tolist()}, indent=2), encoding='utf-8'
            )

    # Validation la 10% cua TRAIN; chi MLP dung de theo doi early stopping.
    mtrain_idx, val_idx = train_test_split(
        np.arange(len(X_train)), test_size=0.10,
        random_state=SEED, stratify=y_cls_train
    )
    print('\n=== Huan luyen MLP theo Epoch / Mini-batch ===')
    mlp, history, mlp_info = train_mlp_epochs(
        X_train.iloc[mtrain_idx], y_cls_train.iloc[mtrain_idx],
        X_train.iloc[val_idx], y_cls_train.iloc[val_idx],
        max_epochs=args.max_epochs, batch_size=args.batch_size, patience=args.patience
    )
    mlp_pred = mlp.predict(X_test)
    rows.append(classification_scores(y_cls_test, mlp_pred, 'MLP Neural Network'))
    joblib.dump(mlp, MODEL_DIR / 'mlp_neural_network.joblib')
    history.to_csv(REPORT_DIR / 'mlp_training_history.csv', index=False)
    (REPORT_DIR / 'mlp_training_summary.json').write_text(
        json.dumps(mlp_info, indent=2), encoding='utf-8'
    )

    metrics = pd.DataFrame(rows).sort_values('F1', ascending=False)
    metrics.to_csv(REPORT_DIR / 'classification_metrics.csv', index=False)

    # K-Means khong dung G3 hay pass_status lam bien dau vao.
    cluster_cols = NUMERIC_FEATURES
    cluster_prep = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    Z = cluster_prep.fit_transform(df[cluster_cols])
    kmeans = KMeans(n_clusters=3, random_state=SEED, n_init=10)
    labels = kmeans.fit_predict(Z)
    joblib.dump({'preprocess': cluster_prep, 'model': kmeans, 'columns': cluster_cols},
                MODEL_DIR / 'kmeans_clusters.joblib')
    grouped = df.assign(cluster=labels).groupby('cluster')[cluster_cols + ['G3']].mean().round(3)
    grouped.to_csv(REPORT_DIR / 'cluster_summary.csv')
    (REPORT_DIR / 'clustering_metrics.json').write_text(
        json.dumps({'silhouette_score': round(float(silhouette_score(Z, labels)), 4),
                    'n_clusters': 3}, indent=2), encoding='utf-8'
    )

    manifest = {
        'dataset': DATA_PATH.name,
        'dataset_sha256': hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        'source': SOURCE_URL,
        'records': int(len(df)),
        'features': FEATURES,
        'test_size': int(len(test_idx)),
        'train_size': int(len(tr_idx)),
        'random_seed': SEED,
        'pass_threshold': PASS_THRESHOLD,
        'trained_at_utc': datetime.now(timezone.utc).isoformat(),
        'mlp': mlp_info,
    }
    (REPORT_DIR / 'training_manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print('\n=== Linear Regression ===')
    print(regression_scores)
    print('\n=== Classification (test) ===')
    print(metrics.to_string(index=False))
    print('\n=== MLP ===')
    print(mlp_info)
    print(f'\nDa luu models/ va reports/. Chay ung dung: python -m streamlit run app.py')


if __name__ == '__main__':
    main()
