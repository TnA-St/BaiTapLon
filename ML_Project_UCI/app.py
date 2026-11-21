"""Student Performance | Streamlit toi gian, 4 tab, bang so va bieu do tron den trang.

Du lieu UCI duoc dong goi trong thu muc data/; khong can tai tu web.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ml_common import (
    DATA_PATH,
    FEATURES,
    MODEL_DIR,
    NUMERIC_FEATURES,
    REPORT_DIR,
    ROOT,
    SOURCE_URL,
    read_data,
)

st.set_page_config(page_title="Student Performance", page_icon="⬛", layout="wide")

# Khong mau me, khong hinh minh hoa, khong sidebar; giu du lieu va bieu do de doc.
st.markdown(
    """
    <style>
      :root { color-scheme: light; }
      .stApp, [data-testid="stAppViewContainer"] { background: #FFFFFF; color: #161616; }
      [data-testid="stHeader"] { background: rgba(255,255,255,.95); }
      [data-testid="stSidebar"] { display: none; }
      h1, h2, h3, p, label, [data-testid="stMetricLabel"] { color: #161616 !important; }
      h1 { font-size: 2rem !important; font-weight: 650 !important; letter-spacing: -.04em; }
      h3 { font-size: 1.12rem !important; }
      div[data-testid="stMetric"] { background:#FFFFFF; border:1px solid #D5D5D5;
          border-radius:5px; padding:15px 16px; }
      div[data-testid="stMetricValue"] { color:#161616; }
      /* Tat ca nut thao tac: NEN DEN - CHU TRANG (ke ca noi dung p/span do Streamlit tao) */
      div[data-testid="stButton"] button,
      div[data-testid="stFormSubmitButton"] button,
      div[data-testid="stDownloadButton"] button {
          background:#111111 !important;
          color:#FFFFFF !important;
          border:1px solid #111111 !important;
          border-radius:5px !important;
          box-shadow:none !important;
      }
      div[data-testid="stButton"] button:hover,
      div[data-testid="stFormSubmitButton"] button:hover,
      div[data-testid="stDownloadButton"] button:hover,
      div[data-testid="stButton"] button:focus,
      div[data-testid="stFormSubmitButton"] button:focus,
      div[data-testid="stDownloadButton"] button:focus,
      div[data-testid="stButton"] button:active,
      div[data-testid="stFormSubmitButton"] button:active,
      div[data-testid="stDownloadButton"] button:active {
          background:#000000 !important;
          color:#FFFFFF !important;
          border-color:#000000 !important;
      }
      div[data-testid="stButton"] button:disabled,
      div[data-testid="stFormSubmitButton"] button:disabled,
      div[data-testid="stDownloadButton"] button:disabled {
          background:#555555 !important;
          color:#FFFFFF !important;
          border-color:#555555 !important;
          opacity:.8;
      }
      div[data-testid="stButton"] button *,
      div[data-testid="stFormSubmitButton"] button *,
      div[data-testid="stDownloadButton"] button *,
      div[data-testid="stButton"] button:hover *,
      div[data-testid="stFormSubmitButton"] button:hover *,
      div[data-testid="stDownloadButton"] button:hover *,
      div[data-testid="stButton"] button:disabled *,
      div[data-testid="stFormSubmitButton"] button:disabled *,
      div[data-testid="stDownloadButton"] button:disabled * {
          color:#FFFFFF !important;
          fill:currentColor !important;
          stroke:currentColor !important;
          -webkit-text-fill-color:#FFFFFF !important;
      }
      button[data-baseweb="tab"] { color:#363636 !important; }
      button[data-baseweb="tab"][aria-selected="true"] {
          color:#0A0A0A !important; border-bottom:2px solid #111111 !important; }
      [data-testid="stFileUploader"], [data-testid="stForm"], [data-testid="stExpander"] {
          border-color:#DADADA !important; }
      hr { border-color: #E5E5E5; }
      .block-container { max-width: 1190px; padding-top:2.1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

CLASSIFIERS = {
    "Logistic Regression": "logistic_regression.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "SVM": "svm.joblib",
    "MLP Neural Network": "mlp_neural_network.joblib",
}
ARTIFACTS = [
    MODEL_DIR / "linear_regression.joblib",
    MODEL_DIR / "kmeans_clusters.joblib",
    REPORT_DIR / "classification_metrics.csv",
    REPORT_DIR / "regression_metrics.json",
    REPORT_DIR / "confusion_matrix.json",
    REPORT_DIR / "cluster_summary.csv",
    REPORT_DIR / "clustering_metrics.json",
    REPORT_DIR / "mlp_training_history.csv",
    REPORT_DIR / "mlp_training_summary.json",
    REPORT_DIR / "training_manifest.json",
] + [MODEL_DIR / filename for filename in CLASSIFIERS.values()]

GRAYS = ["#232323", "#9A9A9A", "#D4D4D4", "#656565", "#E8E8E8"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@st.cache_data(show_spinner=False)
def read_cached(digest: str) -> pd.DataFrame:
    return read_data()


@st.cache_resource(show_spinner=False)
def load_model(filename: str):
    # Chi nap cac mo hinh duoc train_models.py cua chinh du an tao.
    return joblib.load(MODEL_DIR / filename)


@st.cache_data(show_spinner=False)
def load_report(filename: str) -> pd.DataFrame:
    return pd.read_csv(REPORT_DIR / filename)


def load_json(filename: str) -> dict:
    return json.loads((REPORT_DIR / filename).read_text(encoding="utf-8"))


def current_data():
    if not DATA_PATH.is_file():
        return None, None, None
    try:
        digest = sha256(DATA_PATH)
        return read_cached(digest), digest, None
    except (ValueError, OSError, pd.errors.ParserError) as error:
        return None, None, str(error)


def model_ready(digest: str | None) -> bool:
    if digest is None or any(not path.is_file() for path in ARTIFACTS):
        return False
    try:
        return load_json("training_manifest.json").get("dataset_sha256") == digest
    except (ValueError, OSError, KeyError):
        return False


def pie_chart(labels, values, title: str, height: int = 330):
    values = [int(value) for value in values]
    if not values or sum(values) == 0:
        st.info("Không có dữ liệu để vẽ biểu đồ.")
        return
    fig = go.Figure(go.Pie(
        labels=[str(item) for item in labels],
        values=values,
        hole=0,
        sort=False,
        marker={"colors": GRAYS[:len(values)], "line": {"color": "#FFFFFF", "width": 2}},
        textinfo="label+percent",
        textposition="outside",
        automargin=True,
        hovertemplate="%{label}: %{value} mẫu (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        title={"text": title, "font": {"size": 16, "color": "#111111"}},
        template="plotly_white", height=height,
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font={"family": "Arial", "size": 12, "color": "#161616"},
        margin={"l": 25, "r": 25, "t": 55, "b": 20},
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def mlp_learning_curves(history: pd.DataFrame) -> None:
    """Ve hai duong hoc TU CSV thuc nghiem, khong dung hinh minh hoa co dinh."""
    required = ["Epoch", "Train_loss", "Val_accuracy"]
    if any(column not in history.columns for column in required):
        st.warning("File mlp_training_history.csv thieu cot can thiet de ve bieu do.")
        return
    plot_data = history[required].copy()
    for column in required:
        plot_data[column] = pd.to_numeric(plot_data[column], errors="coerce")
    plot_data = plot_data.dropna().sort_values("Epoch")
    if plot_data.empty:
        st.info("CSV huan luyen MLP chua co du lieu Epoch.")
        return

    def line_figure(title: str, y_col: str, y_label: str, y_range=None):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_data["Epoch"], y=plot_data[y_col],
            mode="lines+markers" if len(plot_data) <= 8 else "lines",
            line={"color": "#151515", "width": 2.5},
            marker={"size": 5, "color": "#151515"},
            hovertemplate="Epoch %{x}<br>" + y_label + ": %{y:.4f}<extra></extra>",
            name=y_label,
        ))
        fig.update_layout(
            title={"text": title, "x": 0.5, "font": {"size": 16, "color": "#111111"}},
            template="plotly_white",
            height=345,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            showlegend=False,
            font={"family": "Arial", "size": 12, "color": "#151515"},
            margin={"l": 38, "r": 15, "t": 60, "b": 45},
        )
        fig.update_xaxes(title_text="Vòng học (Epoch)", gridcolor="#E6E6E6", zeroline=False)
        fig.update_yaxes(title_text=y_label, gridcolor="#E6E6E6", zeroline=False, range=y_range)
        return fig

    first, second = st.columns(2)
    with first:
        st.plotly_chart(
            line_figure("Mất mát trên tập học", "Train_loss", "Loss"),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with second:
        st.plotly_chart(
            line_figure("Accuracy trên tập kiểm định", "Val_accuracy", "Accuracy", [0, 1]),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    st.caption(
        "Đồ thị được vẽ trực tiếp từ reports/mlp_training_history.csv do mô hình MLP "
        "huấn luyện trên data/student-por.csv tạo ra, không phải hình minh họa cố định."
    )


df, digest, error = current_data()
trained = model_ready(digest)

st.title("Student Performance")
st.caption("Phân tích và dự đoán kết quả học tập · UCI Student Performance · Thang điểm G3: 0–20")

# Dung CHINH XAC 4 tab. Tat ca ky nang goc duoc gop vao cac tab nay.
tab_data, tab_predict, tab_compare, tab_charts = st.tabs([
    "1. Dữ liệu & huấn luyện",
    "2. Dự đoán",
    "3. So sánh mô hình",
    "4. Biểu đồ",
])

with tab_data:
    st.subheader("Dữ liệu đầu vào")
    c1, c2, c3 = st.columns(3)
    c1.metric("Bản ghi", len(df) if df is not None else "—")
    c2.metric("Điểm G3 trung bình", f"{df['G3'].mean():.2f}" if df is not None else "—")
    c3.metric("Mô hình", "Đã huấn luyện" if trained else "Chưa huấn luyện")
    if error:
        st.error(f"Lỗi đọc dữ liệu: {error}")
    st.write("Dữ liệu có sẵn trong dự án: `data/student-por.csv` (649 bản ghi; dấu phân cách `;`).")
    st.markdown(f"**Link gốc dữ liệu UCI:** [{SOURCE_URL}]({SOURCE_URL})")
    st.caption("Bản ZIP đã kèm sẵn file CSV, vì vậy không cần tải lại hoặc thay dữ liệu trong giao diện.")

    if df is not None:
        st.markdown("**Bảng thống kê**")
        stats = pd.DataFrame({
            "Chỉ tiêu": ["Tổng bản ghi", "Đạt (Pass)", "Không đạt (Fail)", "G3 trung bình", "G3 thấp nhất", "G3 cao nhất"],
            "Giá trị": [len(df), int(df['pass_status'].eq('Pass').sum()),
                        int(df['pass_status'].eq('Fail').sum()),
                        round(float(df['G3'].mean()), 2),
                        int(df['G3'].min()), int(df['G3'].max())],
        })
        st.dataframe(stats, hide_index=True, use_container_width=True)
        with st.expander("Xem dữ liệu CSV (30 dòng đầu)"):
            st.dataframe(df.head(30), hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Huấn luyện các mô hình")
    st.caption("Dữ liệu UCI đã có sẵn. Nhấn Huấn luyện lần đầu để tạo mô hình và báo cáo trên máy (không cần Internet).")
    with st.expander("Thông số MLP", expanded=False):
        a, b, c = st.columns(3)
        with a:
            max_epochs = st.number_input("Số Epoch tối đa", min_value=1, max_value=1000, value=180)
        with b:
            batch_size = st.number_input("Kích thước Mini-batch", min_value=1, max_value=2000, value=200)
        with c:
            patience = st.number_input("Early Stopping patience", min_value=1, max_value=100, value=15)
    if st.button("Huấn luyện / Huấn luyện lại", type="primary", disabled=df is None):
        cmd = [sys.executable, str(ROOT / "train_models.py"),
               "--max-epochs", str(int(max_epochs)),
               "--batch-size", str(int(batch_size)),
               "--patience", str(int(patience))]
        with st.spinner("Đang huấn luyện mô hình từ dữ liệu CSV..."):
            try:
                proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                                      timeout=360, check=False)
            except subprocess.TimeoutExpired:
                st.error("Huấn luyện vượt quá giới hạn 6 phút. Hãy giảm số Epoch.")
            except OSError as exc:
                st.error(f"Không chạy được chương trình huấn luyện: {exc}")
            else:
                if proc.returncode:
                    st.error("Huấn luyện chưa thành công.")
                    st.code((proc.stderr or proc.stdout)[-5000:], language="text")
                else:
                    load_model.clear()
                    load_report.clear()
                    st.session_state.pop("last_prediction", None)
                    st.rerun()
    if trained:
        st.success("Mô hình đã khớp với dữ liệu hiện tại.")

with tab_predict:
    st.subheader("Dự đoán kết quả học tập")
    if not trained:
        st.info("Vào tab 1 và huấn luyện mô hình trước khi dự đoán.")
    else:
        with st.form("prediction_form"):
            left, middle, right = st.columns(3)
            with left:
                studytime = st.selectbox("Thời gian học mỗi tuần", [1, 2, 3, 4],
                    format_func=lambda x: {1:"Dưới 2 giờ", 2:"2–5 giờ", 3:"5–10 giờ", 4:"Trên 10 giờ"}[x])
                failures = st.number_input("Số lần không đạt trước đây", 0, 4, 0)
                absences = st.number_input("Số buổi vắng", 0, 93, 4)
            with middle:
                g1 = st.number_input("Điểm G1", 0, 20, 12)
                g2 = st.number_input("Điểm G2", 0, 20, 14)
                algorithm = st.selectbox("Mô hình phân loại", list(CLASSIFIERS), index=4)
            with right:
                internet = st.selectbox("Có Internet tại nhà", ["yes", "no"], format_func=lambda x:"Có" if x=="yes" else "Không")
                activities = st.selectbox("Có tham gia ngoại khóa", ["yes", "no"], format_func=lambda x:"Có" if x=="yes" else "Không")
                schoolsup = st.selectbox("Có hỗ trợ từ trường", ["no", "yes"], format_func=lambda x:"Có" if x=="yes" else "Không")
                famsup = st.selectbox("Có hỗ trợ từ gia đình", ["yes", "no"], format_func=lambda x:"Có" if x=="yes" else "Không")
            do_predict = st.form_submit_button("Dự đoán", type="primary")
        if do_predict:
            sample = pd.DataFrame([{
                "studytime":studytime,"failures":int(failures),"absences":int(absences),
                "G1":int(g1),"G2":int(g2),"internet":internet,"activities":activities,
                "schoolsup":schoolsup,"famsup":famsup,
            }])
            try:
                score = float(np.clip(load_model("linear_regression.joblib").predict(sample)[0],0,20))
                prediction = str(load_model(CLASSIFIERS[algorithm]).predict(sample)[0])
                st.session_state["last_prediction"] = {
                    "Điểm G3 dự đoán":round(score,2),
                    "Phân loại":"Đạt" if prediction=="Pass" else "Không đạt",
                    "Mô hình phân loại":algorithm,
                }
            except (ValueError, OSError) as exc:
                st.error(f"Không thể dự đoán: {exc}. Hãy thử huấn luyện lại.")
        if "last_prediction" in st.session_state:
            st.markdown("**Kết quả**")
            result = st.session_state["last_prediction"]
            st.dataframe(pd.DataFrame([{"Chỉ tiêu":key,"Giá trị":str(value)}
                                      for key, value in result.items()]),
                         hide_index=True, use_container_width=True)
            st.caption("G3 và nhãn Đạt/Không đạt do hai mô hình độc lập tạo ra; chỉ dùng để tham khảo.")

with tab_compare:
    st.subheader("So sánh hiệu năng mô hình")
    if not trained:
        st.info("Chưa có kết quả đánh giá. Vui lòng huấn luyện tại tab 1.")
    else:
        metrics = load_report("classification_metrics.csv")
        st.markdown("**7 mô hình phân loại (Test)**")
        st.dataframe(metrics, hide_index=True, use_container_width=True)
        st.download_button("Xuất bảng so sánh (.csv)",
                           metrics.to_csv(index=False).encode("utf-8-sig"),
                           file_name="classification_metrics.csv", mime="text/csv")
        st.markdown("**Linear Regression – dự đoán G3**")
        regression = load_json("regression_metrics.json")
        st.dataframe(pd.DataFrame([{"MAE":regression["MAE"], "RMSE":regression["RMSE"],
                                   "R²":regression["R2"]}]),
                     hide_index=True, use_container_width=True)
        st.caption("MAE/RMSE đo sai số điểm trên thang 0–20; R² không phải độ chính xác phân loại.")
        st.markdown("**Ma trận nhầm lẫn – Random Forest**")
        cm = load_json("confusion_matrix.json")
        matrix = pd.DataFrame(cm["matrix"], columns=[f"Dự đoán {name}" for name in cm["labels"]],
                              index=[f"Thực tế {name}" for name in cm["labels"]])
        st.dataframe(matrix, use_container_width=True)
        correct = int(np.trace(np.asarray(cm["matrix"])))
        total = int(np.asarray(cm["matrix"]).sum())
        pie_chart(["Đúng", "Sai"], [correct, total-correct], "Tỷ lệ dự đoán đúng/sai – Random Forest")

with tab_charts:
    st.subheader("Biểu đồ tròn và bảng số liệu")
    if df is None:
        st.info("Không tìm thấy dữ liệu trong thư mục data/. Hãy kiểm tra file student-por.csv.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            counts = df["pass_status"].value_counts().reindex(["Pass", "Fail"], fill_value=0)
            pie_chart(["Đạt", "Không đạt"], counts.tolist(), "Tỷ lệ Đạt / Không đạt")
        with col_b:
            study = df["studytime"].value_counts().reindex([1,2,3,4], fill_value=0)
            pie_chart(["< 2 giờ", "2–5 giờ", "5–10 giờ", "> 10 giờ"],
                      study.tolist(), "Mức thời gian học mỗi tuần")
        a, b, c = st.columns(3)
        a.metric("Bản ghi",len(df))
        b.metric("G3 trung bình",f"{df['G3'].mean():.2f}")
        c.metric("Tỷ lệ đạt",f"{df['pass_status'].eq('Pass').mean():.1%}")
        if trained:
            st.divider()
            st.markdown("**K-Means: 3 cụm học sinh**")
            bundle = load_model("kmeans_clusters.joblib")
            transformed = bundle["preprocess"].transform(df[bundle["columns"]])
            groups = bundle["model"].predict(transformed)
            values = pd.Series(groups).value_counts().reindex([0,1,2],fill_value=0)
            left, right = st.columns([1, 1])
            with left:
                pie_chart(["Cụm 0", "Cụm 1", "Cụm 2"], values.tolist(), "Số mẫu trong từng cụm")
            with right:
                st.markdown("Bảng trung bình theo cụm")
                st.dataframe(load_report("cluster_summary.csv"),
                             hide_index=True, use_container_width=True)
                clustering = load_json("clustering_metrics.json")
                st.metric("Silhouette Score", f"{clustering['silhouette_score']:.3f}")
            st.caption("Các cụm được tạo từ studytime, failures, absences, G1, G2; không sử dụng G3 khi học.")
            st.divider()
            st.markdown("**MLP: thông tin huấn luyện**")
            info = load_json("mlp_training_summary.json")
            x, y, z = st.columns(3)
            x.metric("Epoch đã chạy", info["epochs_run"])
            y.metric("Epoch tốt nhất",info["best_epoch"])
            z.metric("Validation Accuracy tốt nhất",f"{info['best_val_accuracy']:.1%}")
            sizes = [min(info["batch_size"], info["train_size"]-start)
                     for start in range(0, info["train_size"], info["batch_size"])]
            pie_chart([f"Lô {i+1}" for i in range(len(sizes))], sizes,
                      "Phân chia Mini-batch trong một Epoch")
            st.divider()
            st.markdown("**Biểu đồ quá trình học của MLP (theo CSV huấn luyện)**")
            history = load_report("mlp_training_history.csv")
            mlp_learning_curves(history)
            with st.expander("Xem bảng Loss và Accuracy theo Epoch"):
                st.dataframe(history, hide_index=True, use_container_width=True)
                st.download_button("Tải lịch sử MLP", history.to_csv(index=False).encode("utf-8-sig"),
                                   file_name="mlp_training_history.csv", mime="text/csv")
        else:
            st.divider()
            st.info("Huấn luyện mô hình ở tab 1 để xem hai biểu đồ Loss và Validation Accuracy từ CSV thực nghiệm.")

st.divider()
st.caption(f"Nguồn: UCI Student Performance ({SOURCE_URL}) · Dự đoán chỉ mang tính tham khảo, không thay thế đánh giá của giáo viên.")
