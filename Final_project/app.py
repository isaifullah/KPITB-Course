import streamlit as st
import pandas as pd
import numpy as np
import openai
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import re
import hashlib
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = "gpt-4o-mini"

st.set_page_config(page_title="Talk2Data – AI Data Analyst", page_icon="🎙️", layout="wide")

# ========================== FUNCTIONS (copied from notebook – ensure they are below) ==========================
# (Paste all notebook functions here, but I will include the full code inline for clarity)

def analyze_dataframe(df: pd.DataFrame) -> Dict:
    if df is None:
        return {
            'rows': 0, 'cols': 0, 'missing_total': 0,
            'missing_by_column': {}, 'duplicates': 0, 'memory_mb': 0,
            'numeric_cols': [], 'categorical_cols': [], 'boolean_cols': [], 'other_cols': []
        }
    return {
        'rows': len(df),
        'cols': len(df.columns),
        'missing_total': df.isnull().sum().sum(),
        'missing_by_column': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'numeric_cols': df.select_dtypes(include=['number']).columns.tolist(),
        'categorical_cols': [c for c in df.select_dtypes(include=['object']).columns if df[c].nunique() < 20],
        'boolean_cols': df.select_dtypes(include=['bool']).columns.tolist(),
        'other_cols': [c for c in df.columns if c not in df.select_dtypes(include=['number','object','bool']).columns],
    }

def get_column_stats_structured(df: pd.DataFrame, col: str) -> str:
    if df is None or col not in df.columns:
        return "Column not found."
    data = df[col]
    stats = {
        "Column": col,
        "Type": str(data.dtype),
        "Non‑null": f"{data.count()} / {len(df)}",
        "Null": data.isnull().sum(),
        "Unique": data.nunique(),
    }
    if col in df.select_dtypes(include=['number']).columns:
        stats["Mean"] = f"{data.mean():.2f}"
        stats["Std"] = f"{data.std():.2f}"
        stats["Min"] = f"{data.min():.2f}"
        stats["25%"] = f"{data.quantile(0.25):.2f}"
        stats["50%"] = f"{data.median():.2f}"
        stats["75%"] = f"{data.quantile(0.75):.2f}"
        stats["Max"] = f"{data.max():.2f}"
    elif col in df.select_dtypes(include=['bool']).columns:
        true_cnt = data.sum()
        stats["True"] = true_cnt
        stats["False"] = len(df) - true_cnt
    elif col in df.select_dtypes(include=['object']).columns:
        top5 = data.value_counts().head(5)
        stats["Top 5"] = ", ".join(f"{v}({c})" for v, c in top5.items())
    else:
        stats["Sample"] = data.head(5).tolist()
    table = "| Attribute | Value |\n| --- | --- |\n"
    for k, v in stats.items():
        table += f"| {k} | {v} |\n"
    return table

def is_id_column(col_name: str) -> bool:
    lower = col_name.lower()
    return any(id_word in lower for id_word in ['id', 'passengerid', 'customerid', 'rowid'])

def detect_outliers_all_numeric(df: pd.DataFrame, skip_ids: bool = True) -> str:
    if df is None:
        return "No data loaded."
    numeric = df.select_dtypes(include=['number']).columns.tolist()
    if skip_ids:
        numeric = [c for c in numeric if not is_id_column(c)]
    if not numeric:
        return "No numeric columns found for outlier detection."
    result = []
    for col in numeric:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)].shape[0]
        result.append(f"- **{col}**: {outliers} outliers (outside [{lower:.2f}, {upper:.2f}])")
    return "\n".join(result)

def rename_column(df: pd.DataFrame, old_name: str, new_name: str) -> pd.DataFrame:
    df = df.copy()
    return df.rename(columns={old_name: new_name})

def change_column_type(df: pd.DataFrame, col: str, new_type: str) -> pd.DataFrame:
    df = df.copy()
    if new_type == 'numeric':
        df[col] = pd.to_numeric(df[col], errors='coerce')
    elif new_type == 'category':
        df[col] = df[col].astype('category')
    elif new_type == 'datetime':
        df[col] = pd.to_datetime(df[col], errors='coerce')
    elif new_type == 'string':
        df[col] = df[col].astype(str)
    else:
        raise ValueError(f"Unsupported type: {new_type}")
    return df

def merge_two_columns(df: pd.DataFrame, col1: str, col2: str, new_name: str, how: str = 'concat') -> pd.DataFrame:
    df = df.copy()
    if how == 'concat':
        df[new_name] = df[col1].astype(str) + "_" + df[col2].astype(str)
    elif how == 'add':
        if col1 in df.select_dtypes(include=['number']).columns and col2 in df.select_dtypes(include=['number']).columns:
            df[new_name] = df[col1] + df[col2]
        else:
            raise ValueError("Addition only for numeric columns.")
    else:
        raise ValueError("how must be 'concat' or 'add'.")
    return df

def create_binned_column(df: pd.DataFrame, col: str, bins: List[float], labels: List[str], new_name: str = None) -> pd.DataFrame:
    df = df.copy()
    if new_name is None:
        new_name = f"{col}_binned"
    df[new_name] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    return df

def create_column_from_expression(df: pd.DataFrame, expression: str, new_name: str) -> pd.DataFrame:
    df = df.copy()
    try:
        df[new_name] = eval(expression)
        return df
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")

def apply_label_encoding(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    le = LabelEncoder()
    df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
    return df

def apply_one_hot_encoding(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    one_hot = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, one_hot], axis=1)
    return df

def apply_ordinal_encoding(df: pd.DataFrame, col: str, order: List[str]) -> pd.DataFrame:
    df = df.copy()
    oe = OrdinalEncoder(categories=[order])
    df[col + "_ordinal"] = oe.fit_transform(df[[col]])
    return df

def suggest_algorithms(df: pd.DataFrame, target_col: str = None) -> str:
    if df is None:
        return "No data loaded."
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    rows, cols = df.shape
    suggestion = f"**Dataset:** {rows} rows, {cols} columns\n"
    suggestion += f"**Numeric columns:** {len(numeric_cols)}\n"
    suggestion += f"**Categorical columns:** {len(cat_cols)}\n\n"
    if target_col:
        if target_col in numeric_cols:
            suggestion += "**Target is numeric → Regression algorithms:**\n"
            reg_algs = [
                "Linear Regression", "Ridge", "Lasso", "ElasticNet",
                "Decision Tree Regressor", "Random Forest Regressor",
                "Gradient Boosting Regressor", "XGBoost Regressor",
                "LightGBM Regressor", "CatBoost Regressor",
                "SVR", "KNeighbors Regressor", "MLP Regressor",
                "AdaBoost Regressor", "Bagging Regressor", "ExtraTrees Regressor"
            ]
            suggestion += "\n".join(f"- {alg}" for alg in reg_algs) + "\n"
        elif target_col in cat_cols or target_col in df.select_dtypes(include=['bool']).columns:
            suggestion += "**Target is categorical → Classification algorithms:**\n"
            class_algs = [
                "Logistic Regression", "Decision Tree Classifier",
                "Random Forest Classifier", "Gradient Boosting Classifier",
                "XGBoost Classifier", "LightGBM Classifier",
                "CatBoost Classifier", "SVC", "KNeighbors Classifier",
                "Naive Bayes", "MLP Classifier", "AdaBoost Classifier",
                "Bagging Classifier", "ExtraTrees Classifier",
                "QDA", "LDA"
            ]
            suggestion += "\n".join(f"- {alg}" for alg in class_algs) + "\n"
        else:
            suggestion += "**Unsupervised learning:**\n- Clustering (KMeans, DBSCAN)\n- Dimensionality reduction (PCA, t-SNE)\n"
    else:
        suggestion += "**General recommendations:**\n"
        if rows < 100:
            suggestion += "- Linear/Logistic Regression, Decision Trees\n"
        else:
            suggestion += "- Random Forest, XGBoost, LightGBM\n"
        if len(numeric_cols) > 20:
            suggestion += "- Consider PCA for dimensionality reduction.\n"
        if len(cat_cols) > 10:
            suggestion += "- Use one‑hot encoding for categorical variables.\n"
    return suggestion

def data_overview_score(df: pd.DataFrame) -> str:
    if df is None:
        return "No data loaded."
    total_cells = df.shape[0] * df.shape[1]
    missing_pct = df.isnull().sum().sum() / total_cells * 100 if total_cells > 0 else 0
    duplicate_pct = df.duplicated().sum() / df.shape[0] * 100 if df.shape[0] > 0 else 0
    completeness = 100 - missing_pct
    uniqueness = 100 - duplicate_pct
    numeric_ratio = len(df.select_dtypes(include=['number']).columns) / df.shape[1] * 100 if df.shape[1] > 0 else 0
    score = (completeness * 0.4 + uniqueness * 0.3 + numeric_ratio * 0.3)
    grade = "A" if score >= 80 else "B" if score >= 60 else "C" if score >= 40 else "D"
    overview = f"""
**Data Quality Overview**
- **Completeness:** {completeness:.1f}% (missing {missing_pct:.1f}%)
- **Uniqueness:** {uniqueness:.1f}% (duplicates {duplicate_pct:.1f}%)
- **Numeric ratio:** {numeric_ratio:.1f}%
- **Overall Score:** {score:.1f}/100 → Grade **{grade}**

**Recommendations:**
"""
    if missing_pct > 5:
        overview += "- ⚠️ Missing values >5% – impute with mean/mode.\n"
    if duplicate_pct > 5:
        overview += "- ⚠️ Duplicates >5% – run `df.drop_duplicates()`.\n"
    if numeric_ratio < 30:
        overview += "- 📊 Low numeric ratio – encode categorical variables for ML.\n"
    return overview

def ask_data_analyst(df: pd.DataFrame, profile: Dict, question: str, style: str = "balanced") -> Tuple[str, Optional[plt.Figure]]:
    if df is None:
        return "Please upload a dataset first.", None
    if profile is None:
        profile = analyze_dataframe(df)
    q = question.lower().strip()
    numeric = profile.get('numeric_cols', [])
    cat = profile.get('categorical_cols', [])
    if "code" in q:
        col_list = ", ".join(df.columns[:15])
        sample_str = df.head(3).to_markdown()
        prompt = f"""You are a Python data scientist. Generate only the Python code (no explanation) for the following request.

Dataset columns: {col_list}
Data types: {df.dtypes.to_dict()}
First 3 rows:
{sample_str}

User request: {question}

Provide the complete, executable code that works on the dataframe `df`. Use pandas, matplotlib, seaborn, or sklearn as needed.
Only output the code block, no extra text."""
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500
        )
        code_text = response.choices[0].message.content
        code_text = re.sub(r'^```python\s*', '', code_text)
        code_text = re.sub(r'\s*```$', '', code_text)
        return f"**Answer:** Code for your request:\n```python\n{code_text}\n```", None
    row_match = re.search(r'row\s+(\d+)', q)
    if row_match:
        r = int(row_match.group(1))
        if 1 <= r <= len(df):
            row_series = df.iloc[r-1].to_frame().T
            return f"**Answer:** Row {r}\n\n{row_series.to_markdown()}", None
        else:
            return f"Row {r} does not exist.", None
    if "first" in q or "head" in q:
        n = 15 if "15" in q else (10 if "10" in q else 5)
        return f"**Answer:** First {n} rows\n\n{df.head(n).to_markdown()}", None
    if "last" in q or "tail" in q:
        n = 15 if "15" in q else (10 if "10" in q else 5)
        return f"**Answer:** Last {n} rows\n\n{df.tail(n).to_markdown()}", None
    if "sample" in q:
        n = 15 if "15" in q else 5
        return f"**Answer:** {n} random rows\n\n{df.sample(min(n, len(df))).to_markdown()}", None
    if "outlier" in q and "code" not in q:
        outlier_summary = detect_outliers_all_numeric(df, skip_ids=True)
        return f"**Answer:** Outlier detection (IQR method)\n\n{outlier_summary}\n\n**Insights:** Outliers can skew statistics.", None
    if "average" in q or "mean" in q:
        for col in numeric:
            if col in q:
                return f"**Answer:** Average of **{col}** = {df[col].mean():.2f}", None
        if numeric:
            return f"**Answer:** Average of first numeric column **{numeric[0]}** = {df[numeric[0]].mean():.2f}", None
        else:
            return "No numeric columns to compute average.", None
    styles = {
        "short": "Provide answer in 1‑2 short sentences with numbers if possible.",
        "balanced": "Provide answer in 2‑3 sentences with explanation.",
        "detailed": "Provide answer in 3‑4 sentences with insights and recommendation."
    }
    context = f"""DATASET: {len(df)} rows, {len(df.columns)} columns
First 3 rows:
{df.head(3).to_markdown()}

User question: {question}

{styles.get(style, styles['balanced'])}
Return in format:
**Answer:** ...
**Explanation:** ...
**Insights:** ...
**Recommendation:** (if applicable)
"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": context}],
        temperature=0.2,
        max_tokens=600
    )
    return response.choices[0].message.content, None

def create_matplotlib_plot(df, plot_type, x=None, y=None, hue=None):
    fig, ax = plt.subplots(figsize=(10,6))
    fig.patch.set_facecolor('#0f0c29')
    ax.set_facecolor('#1a1a3e')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    try:
        if plot_type == "Histogram":
            if hue:
                for val in df[hue].unique():
                    df[df[hue]==val][x].hist(bins=30, alpha=0.5, label=str(val), ax=ax)
                ax.legend()
            else:
                df[x].hist(bins=30, color='#667eea', edgecolor='white', ax=ax)
            ax.set_xlabel(x); ax.set_ylabel('Frequency')
        elif plot_type == "Bar Chart":
            if y:
                if hue:
                    df.pivot_table(index=x, columns=hue, values=y, aggfunc='mean').plot(kind='bar', ax=ax)
                    ax.set_ylabel(f'Average {y}')
                else:
                    df.groupby(x)[y].mean().plot(kind='bar', color='#667eea', ax=ax)
                    ax.set_ylabel(f'Average {y}')
            else:
                if hue:
                    pd.crosstab(df[x], df[hue]).plot(kind='bar', stacked=True, ax=ax)
                    ax.set_ylabel('Count')
                else:
                    df[x].value_counts().plot(kind='bar', color='#667eea', ax=ax)
                    ax.set_ylabel('Count')
            ax.set_xlabel(x)
            plt.xticks(rotation=45)
        elif plot_type == "Scatter Plot":
            if hue:
                sc = ax.scatter(df[x], df[y], c=df[hue].astype('category').cat.codes, cmap='viridis', alpha=0.6)
                plt.colorbar(sc, label=hue)
            else:
                ax.scatter(df[x], df[y], alpha=0.6, color='#667eea')
            ax.set_xlabel(x); ax.set_ylabel(y)
        elif plot_type == "Box Plot":
            if hue:
                sns.boxplot(data=df, x=x, y=y, hue=hue, palette='Set3', ax=ax)
            else:
                sns.boxplot(data=df, x=x, y=y, palette='Set3', ax=ax)
            plt.xticks(rotation=45)
        elif plot_type == "Line Plot":
            if hue:
                for val in df[hue].unique():
                    sub = df[df[hue]==val].sort_values(x)
                    ax.plot(sub[x], sub[y], marker='o', label=str(val))
                ax.legend()
            else:
                df.sort_values(x).plot(x=x, y=y, kind='line', color='#667eea', marker='o', ax=ax)
            ax.set_xlabel(x); ax.set_ylabel(y)
        elif plot_type == "Pie Chart":
            df[x].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=sns.color_palette("Set3"))
            ax.set_ylabel('')
        ax.set_title(plot_type, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, color='white')
        plt.tight_layout()
    except Exception:
        return None
    return fig

def create_seaborn_plot(df, plot_type, x=None, y=None, hue=None):
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(10,6))
    fig.patch.set_facecolor('#0f0c29')
    ax.set_facecolor('#1a1a3e')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    try:
        if plot_type == "Histogram":
            sns.histplot(data=df, x=x, hue=hue, bins=30, alpha=0.7, ax=ax)
            ax.set_xlabel(x); ax.set_ylabel('Frequency')
        elif plot_type == "Bar Plot":
            if y:
                sns.barplot(data=df, x=x, y=y, hue=hue, palette='Blues_d', ax=ax)
                ax.set_ylabel(f'Average {y}')
            else:
                sns.countplot(data=df, x=x, hue=hue, palette='Blues_d', ax=ax)
                ax.set_ylabel('Count')
            ax.set_xlabel(x)
            plt.xticks(rotation=45)
        elif plot_type == "Scatter Plot":
            sns.scatterplot(data=df, x=x, y=y, hue=hue, palette='viridis', s=50, ax=ax)
            ax.set_xlabel(x); ax.set_ylabel(y)
        elif plot_type == "Box Plot":
            sns.boxplot(data=df, x=x, y=y, hue=hue, palette='Set3', ax=ax)
            plt.xticks(rotation=45)
        elif plot_type == "Heatmap":
            numeric_df = df.select_dtypes(include=['number'])
            if len(numeric_df.columns) > 1:
                sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
                ax.set_title("Correlation Heatmap")
        if plot_type != "Heatmap":
            ax.set_title(plot_type, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, color='white')
        plt.tight_layout()
    except Exception:
        return None
    return fig

def create_plotly_plot(df, plot_type, x=None, y=None, color=None):
    try:
        if plot_type == "Scatter Plot":
            fig = px.scatter(df, x=x, y=y, color=color, title="Scatter Plot", template="plotly_dark")
        elif plot_type == "Line Plot":
            fig = px.line(df, x=x, y=y, color=color, title="Line Plot", template="plotly_dark", markers=True)
        elif plot_type == "Bar Chart":
            if y:
                fig = px.bar(df, x=x, y=y, color=color, title="Bar Chart", template="plotly_dark", barmode='group')
            else:
                data = df[x].value_counts().reset_index()
                fig = px.bar(data, x=x, y='count', color=color, title="Bar Chart", template="plotly_dark")
        elif plot_type == "Histogram":
            fig = px.histogram(df, x=x, color=color, nbins=30, title="Histogram", template="plotly_dark")
        elif plot_type == "Box Plot":
            fig = px.box(df, x=x, y=y, color=color, title="Box Plot", template="plotly_dark")
        elif plot_type == "Heatmap":
            numeric_df = df.select_dtypes(include=['number'])
            fig = px.imshow(numeric_df.corr(), text_auto=True, title="Heatmap", template="plotly_dark")
        else:
            return None
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        return fig
    except Exception:
        return None

# ========================== STREAMLIT UI ==========================
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); }
    .hero-section { background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%); border-radius: 20px; padding: 1rem 2rem; margin-bottom: 1.5rem; text-align: center; position: relative; }
    .hero-title { font-size: 2rem; font-weight: bold; color: white; margin: 0; }
    .hero-description { color: rgba(255,255,255,0.9); font-size: 0.85rem; margin-top: 0.2rem; }
    .badge { position: absolute; top: 0.8rem; right: 1.2rem; background: rgba(0,0,0,0.4); padding: 0.2rem 0.7rem; border-radius: 20px; font-size: 0.7rem; color: rgba(255,255,255,0.9); }
    .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1rem 0; }
    .stat-item { background: rgba(255,255,255,0.05); border-radius: 12px; padding: 0.7rem; text-align: center; }
    .stat-number { font-size: 1.5rem; font-weight: bold; background: linear-gradient(135deg, #667eea 0%, #f093fb 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .stat-label { color: rgba(255,255,255,0.6); font-size: 0.7rem; }
    .message-user { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 0.7rem 1rem; border-radius: 15px; margin-left: 20%; margin-bottom: 1rem; }
    .message-assistant { background: rgba(255,255,255,0.05); color: #e0e0e0; padding: 0.7rem 1rem; border-radius: 15px; margin-right: 20%; margin-bottom: 1rem; border-left: 3px solid #667eea; }
    .chat-container { height: 450px; overflow-y: auto; margin-bottom: 90px; padding: 1rem; }
    .chat-input-fixed { position: fixed; bottom: 0; left: 320px; right: 0; background: rgba(26,26,46,0.95); backdrop-filter: blur(10px); padding: 1rem; border-top: 1px solid rgba(255,255,255,0.1); z-index: 1000; }
    .footer { text-align: center; padding: 1rem; color: rgba(255,255,255,0.3); font-size: 0.7rem; border-top: 1px solid rgba(255,255,255,0.05); margin-top: 2rem; }
    .stButton > button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 8px; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; background: rgba(255,255,255,0.05); border-radius: 10px; padding: 0.3rem; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 0.4rem 1rem; color: rgba(255,255,255,0.7); }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-section">
    <div class="hero-title">🎙️ Talk with Data</div>
    <div class="hero-description">🧞‍♂️ Hi, I'm Data Genie! Your senior data analyst. Ask me anything – I'll answer, show rows, draw graphs, give you code, rename columns, change types, compare data, and suggest algorithms.</div>
    <div class="badge">⚡ Powered by GPT-4o Mini</div>
</div>
""", unsafe_allow_html=True)

# Session state
if "_df" not in st.session_state:
    st.session_state._df = None
if "_profile" not in st.session_state:
    st.session_state._profile = None
if "_messages" not in st.session_state:
    st.session_state._messages = []
if "_last_hash" not in st.session_state:
    st.session_state._last_hash = None

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file")

uploaded_file = st.file_uploader("", type=['csv', 'xlsx', 'xls'], label_visibility="collapsed")

if uploaded_file:
    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
    if file_hash != st.session_state._last_hash:
        st.session_state._messages = []
        st.session_state._df = None
        st.session_state._profile = None
        st.session_state._last_hash = file_hash

    try:
        df = load_data(uploaded_file)
        st.session_state._df = df
        st.session_state._profile = analyze_dataframe(df)
        p = st.session_state._profile
        st.markdown('<div class="stats-grid">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(f'<div class="stat-item"><div class="stat-number">{p["rows"]:,}</div><div class="stat-label">Rows</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="stat-item"><div class="stat-number">{p["cols"]}</div><div class="stat-label">Columns</div></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="stat-item"><div class="stat-number">{p["missing_total"]}</div><div class="stat-label">Missing</div></div>', unsafe_allow_html=True)
        with c4: st.markdown(f'<div class="stat-item"><div class="stat-number">{p["duplicates"]}</div><div class="stat-label">Duplicates</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.success(f"✅ Loaded {uploaded_file.name}")
    except Exception as e:
        st.error(f"Error loading file: {e}")

if st.session_state._df is not None:
    df = st.session_state._df
    profile = st.session_state._profile
    if profile is None:
        profile = analyze_dataframe(df)
        st.session_state._profile = profile

    tabs = st.tabs(["💬 Chat", "📊 Visualize", "📋 Column Info", "✏️ Rename", "🔢 Encoding", "🧠 Algorithms", "⚙️ Feature Engineering", "💾 Download"])

    # ----- Chat Tab -----
    with tabs[0]:
        st.markdown("### 💬 Ask Questions")
        style = st.radio("Style", ["Short", "Balanced", "Detailed"], horizontal=True)
        style_map = {"Short": "short", "Balanced": "balanced", "Detailed": "detailed"}
        for msg in st.session_state._messages[-30:]:
            if msg['role'] == 'user':
                st.markdown(f'<div class="message-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                content = msg["content"]
                code_pattern = r'```python\n(.*?)```'
                parts = re.split(code_pattern, content, flags=re.DOTALL)
                if len(parts) > 1:
                    for i in range(0, len(parts), 2):
                        if parts[i].strip():
                            st.markdown(f'<div class="message-assistant">🤖 {parts[i]}</div>', unsafe_allow_html=True)
                        if i+1 < len(parts):
                            code = parts[i+1].strip()
                            col_code, col_btn = st.columns([10, 1])
                            with col_code:
                                st.code(code, language='python')
                            with col_btn:
                                if st.button("📋", key=f"copy_{i}"):
                                    st.write(f'<script>navigator.clipboard.writeText(`{code}`);</script>', unsafe_allow_html=True)
                                    st.toast("Code copied!", icon="✅")
                else:
                    st.markdown(f'<div class="message-assistant">🤖 {content}</div>', unsafe_allow_html=True)

        st.markdown('<div class="chat-input-fixed">', unsafe_allow_html=True)
        with st.form("chat_form", clear_on_submit=True):
            col1, col2 = st.columns([5,1])
            with col1:
                user_q = st.text_input("", placeholder="Ask about your data...", label_visibility="collapsed")
            with col2:
                sent = st.form_submit_button("Send", use_container_width=True)
            if sent and user_q:
                st.session_state._messages.append({'role': 'user', 'content': user_q})
                with st.spinner("Thinking like a data analyst..."):
                    txt, fig = ask_data_analyst(df, profile, user_q, style_map[style])
                    if fig:
                        st.pyplot(fig)
                        plt.close()
                    st.session_state._messages.append({'role': 'assistant', 'content': txt})
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ----- Visualize Tab -----
    with tabs[1]:
        st.markdown("### 📊 Visualizations")
        lib = st.radio("Library", ["Matplotlib", "Seaborn", "Plotly"], horizontal=True)
        if lib == "Matplotlib":
            plot_types = ["Histogram", "Bar Chart", "Scatter Plot", "Box Plot", "Line Plot", "Pie Chart"]
        elif lib == "Seaborn":
            plot_types = ["Histogram", "Bar Plot", "Scatter Plot", "Box Plot", "Heatmap"]
        else:
            plot_types = ["Scatter Plot", "Line Plot", "Bar Chart", "Histogram", "Box Plot", "Heatmap"]
        plot_type = st.selectbox("Plot Type", plot_types)
        numeric = df.select_dtypes(include=['number']).columns.tolist()
        cat = df.select_dtypes(include=['object']).columns.tolist()
        x_col = y_col = hue_col = None
        if plot_type in ["Histogram"]:
            x_col = st.selectbox("X (numeric)", numeric if numeric else df.columns)
            if lib != "Plotly" and cat:
                hue_col = st.selectbox("Hue (optional)", ["None"] + cat)
                hue_col = None if hue_col == "None" else hue_col
        elif plot_type in ["Bar Chart", "Bar Plot", "Pie Chart"]:
            x_col = st.selectbox("Category", cat if cat else df.columns)
            if plot_type in ["Bar Chart", "Bar Plot"] and numeric:
                y_col = st.selectbox("Value (optional)", ["None"] + numeric)
                y_col = None if y_col == "None" else y_col
            if lib != "Plotly" and plot_type != "Pie Chart" and cat:
                hue_col = st.selectbox("Hue (optional)", ["None"] + cat)
                hue_col = None if hue_col == "None" else hue_col
        elif plot_type in ["Scatter Plot", "Box Plot", "Line Plot"]:
            x_col = st.selectbox("X", numeric if plot_type != "Box Plot" else cat)
            y_col = st.selectbox("Y", numeric)
            if lib != "Plotly" and cat:
                hue_col = st.selectbox("Hue (optional)", ["None"] + cat)
                hue_col = None if hue_col == "None" else hue_col
        elif plot_type == "Heatmap":
            pass
        if st.button("Generate", use_container_width=True):
            if x_col or plot_type == "Heatmap":
                with st.spinner("Drawing..."):
                    try:
                        if lib == "Matplotlib":
                            fig = create_matplotlib_plot(df, plot_type, x_col, y_col, hue_col)
                            if fig:
                                st.pyplot(fig)
                                plt.close()
                        elif lib == "Seaborn":
                            fig = create_seaborn_plot(df, plot_type, x_col, y_col, hue_col)
                            if fig:
                                st.pyplot(fig)
                                plt.close()
                        else:
                            fig = create_plotly_plot(df, plot_type, x_col, y_col, hue_col)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Visualization error: {e}")

    # ----- Column Info Tab -----
    with tabs[2]:
        st.markdown("### 📋 Column Info")
        col_cat = st.radio("Show", ["Numeric Columns", "Categorical Columns", "Boolean Columns", "Other Columns"], horizontal=True)
        if col_cat == "Numeric Columns":
            cols = profile['numeric_cols']
        elif col_cat == "Categorical Columns":
            cols = profile['categorical_cols']
        elif col_cat == "Boolean Columns":
            cols = profile['boolean_cols']
        else:
            cols = profile['other_cols']
        if cols:
            sel = st.selectbox("Select column", cols)
            if sel:
                stats = get_column_stats_structured(df, sel)
                st.markdown(stats)

    # ----- Rename Columns Tab (NOW MODIFIES THE DATAFRAME) -----
    with tabs[3]:
        st.markdown("### ✏️ Rename Columns")
        col_to_rename = st.selectbox("Select column to rename", df.columns)
        new_name = st.text_input("New column name", value=col_to_rename)
        if st.button("Apply Rename"):
            if new_name and new_name != col_to_rename:
                try:
                    # Apply rename and update session state
                    new_df = rename_column(df, col_to_rename, new_name)
                    st.session_state._df = new_df
                    st.session_state._profile = analyze_dataframe(new_df)
                    st.success(f"✅ Renamed '{col_to_rename}' to '{new_name}'")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please enter a new name.")

    # ----- Encoding Tab (NOW MODIFIES THE DATAFRAME) -----
    with tabs[4]:
        st.markdown("### 🔢 Encoding Categorical Columns")
        encoding_type = st.selectbox("Encoding Method", ["Label Encoding", "One-Hot Encoding", "Ordinal Encoding"])
        cat_cols_enc = profile['categorical_cols'] + [c for c in df.select_dtypes(include=['object']).columns if c not in profile['categorical_cols']]
        if cat_cols_enc:
            col_to_encode = st.selectbox("Select column to encode", cat_cols_enc)
            if encoding_type == "Ordinal Encoding":
                unique_vals = df[col_to_encode].dropna().unique().tolist()
                st.write("Current unique values:", unique_vals)
                order_input = st.text_input("Enter order (comma-separated)", value=",".join(unique_vals))
                order = [v.strip() for v in order_input.split(",")]
                if st.button("Apply Ordinal Encoding"):
                    try:
                        new_df = apply_ordinal_encoding(df, col_to_encode, order)
                        st.session_state._df = new_df
                        st.session_state._profile = analyze_dataframe(new_df)
                        st.success(f"✅ Applied ordinal encoding to '{col_to_encode}'. New column: {col_to_encode}_ordinal")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                if st.button(f"Apply {encoding_type}"):
                    try:
                        if encoding_type == "Label Encoding":
                            new_df = apply_label_encoding(df, col_to_encode)
                            st.success(f"✅ Label encoding applied. New column: {col_to_encode}_encoded")
                        else:
                            new_df = apply_one_hot_encoding(df, col_to_encode)
                            st.success(f"✅ One‑hot encoding applied. New columns created with prefix '{col_to_encode}'")
                        st.session_state._df = new_df
                        st.session_state._profile = analyze_dataframe(new_df)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.info("No categorical columns available for encoding.")

    # ----- Algorithms Tab (unchanged) -----
    with tabs[5]:
        st.markdown("### 🧠 Suggested Algorithms")
        target_col = st.selectbox("Target column (optional)", ["None"] + df.columns.tolist())
        target = None if target_col == "None" else target_col
        suggestion = suggest_algorithms(df, target)
        st.markdown(suggestion)

    # ----- Feature Engineering Tab (NOW MODIFIES THE DATAFRAME) -----
    with tabs[6]:
        st.markdown("### ⚙️ Feature Engineering")
        feat_type = st.selectbox("Operation", ["Create Binned Column", "Merge Two Columns", "Create Column from Expression"])

        if feat_type == "Create Binned Column":
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                col = st.selectbox("Numeric column", numeric_cols)
                n_bins = st.number_input("Number of bins", min_value=2, max_value=20, value=4)
                bin_labels = st.text_input("Bin labels (comma-separated)", value=",".join([f"bin_{i}" for i in range(n_bins)]))
                labels = [l.strip() for l in bin_labels.split(",")]
                if len(labels) != n_bins:
                    st.warning(f"Number of labels must equal number of bins ({n_bins}).")
                else:
                    if st.button("Apply Binning"):
                        try:
                            bins = np.linspace(df[col].min(), df[col].max(), n_bins+1)
                            new_df = create_binned_column(df, col, bins.tolist(), labels)
                            st.session_state._df = new_df
                            st.session_state._profile = analyze_dataframe(new_df)
                            st.success(f"✅ Created binned column: {col}_binned")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
            else:
                st.info("No numeric columns available for binning.")

        elif feat_type == "Merge Two Columns":
            cols = df.columns.tolist()
            col1 = st.selectbox("First column", cols)
            col2 = st.selectbox("Second column", cols)
            new_name = st.text_input("New column name", value=f"{col1}_{col2}")
            merge_type = st.radio("Merge type", ["Concat (string)", "Add (numeric)"])
            how = "concat" if "Concat" in merge_type else "add"
            if st.button("Merge Columns"):
                try:
                    new_df = merge_two_columns(df, col1, col2, new_name, how)
                    st.session_state._df = new_df
                    st.session_state._profile = analyze_dataframe(new_df)
                    st.success(f"✅ Created new column: {new_name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

        elif feat_type == "Create Column from Expression":
            st.info("Use pandas syntax. Example: `df['Age'] * 2` or `df['Salary'] / 1000`")
            expression = st.text_input("Expression", placeholder="df['Age'] * 2")
            new_col_name = st.text_input("New column name")
            if st.button("Create Column"):
                if expression and new_col_name:
                    try:
                        new_df = create_column_from_expression(df, expression, new_col_name)
                        st.session_state._df = new_df
                        st.session_state._profile = analyze_dataframe(new_df)
                        st.success(f"✅ Created column: {new_col_name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Invalid expression: {e}")
                else:
                    st.warning("Please provide expression and column name.")

    # ----- Download Tab (includes custom filename) -----
    with tabs[7]:
        st.markdown("### 💾 Download Processed Data")
        default_filename = "processed_data.csv"
        custom_filename = st.text_input("File name (without .csv)", value=default_filename.replace(".csv", ""))
        final_filename = custom_filename.strip() + ".csv" if custom_filename.strip() else default_filename
        st.download_button(
            label="📥 Download CSV",
            data=st.session_state._df.to_csv(index=False).encode('utf-8'),
            file_name=final_filename,
            mime="text/csv",
            use_container_width=True
        )

else:
    st.info("📁 **Upload a CSV or Excel file to start**")
    st.markdown("""
    <div style="text-align:center; padding:2rem;">
        <p style="color:rgba(255,255,255,0.6);">Supported: CSV, XLSX, XLS</p>
        <p style="color:rgba(255,255,255,0.4);">🚀 Senior Data Analyst | GPT-4o Mini | Auto Insights | Code Generation | Feature Engineering | Algorithm Suggestions</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="footer"><p>🚀 Powered by GPT-4o Mini | Professional Data Analyst behavior</p></div>', unsafe_allow_html=True)