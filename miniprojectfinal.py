import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.ensemble import IsolationForest
import tempfile
import os
from datetime import datetime
import warnings

# Import reportlab for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

warnings.filterwarnings('ignore')

# ===========================
# PAGE CONFIG & STYLING
# ===========================
st.set_page_config(page_title="Smart Data Cleaner & Visualizer", layout="wide", page_icon="🧹")

THEME_CSS = """
<style>
:root { --radius: 18px; }
.main { padding-top: 0rem; }
section[data-testid="stSidebar"] > div { padding-top: .5rem; }

.app-hero {
  background: linear-gradient(135deg, #0b1220, #1f2937 40%, #2563eb 100%);
  color: #e5e7eb;
  border-radius: var(--radius);
  padding: 28px 28px;
  box-shadow: 0 10px 30px rgba(0,0,0,.25);
  margin-bottom: 18px;
}
.app-hero h1 { margin: 0; color: #ffffff; letter-spacing: .2px; font-size: 34px; }
.app-hero p { margin-top: 8px; color: #c9d1e6; font-size: 15px; }

.card { background: #ffffff; border-radius: var(--radius); padding: 16px 18px; margin-bottom: 16px; border: 1px solid #e9eef5; box-shadow: 0 8px 24px rgba(28, 41, 61, 0.08); }
.status-panel { background: #f1f5ff; color: #0e1b2b; border: 1px solid #dbe6ff; border-radius: 16px; padding: 12px 14px; margin: 8px 0 16px 0; font-size: 14px; }
.section-label { display: inline-block; padding: 6px 10px; border-radius: 999px; background: linear-gradient(90deg, #2563eb, #22c55e); color: white; font-weight: 700; font-size: 12px; letter-spacing: .5px; }
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="app-hero">
      <h1>🧹 Smart Data Cleaner & Visualizer</h1>
      <p>Upload → Recommendations → Clean → Visualize → Transform → Anomaly → Report</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ===========================
# SESSION STATE & HELPERS
# ===========================
def ensure_session_state():
    if "df_original" not in st.session_state:
        st.session_state.df_original = None
    if "df" not in st.session_state:
        st.session_state.df = None
    if "df_transformed" not in st.session_state:
        st.session_state.df_transformed = None
    if "rec_settings" not in st.session_state:
        st.session_state.rec_settings = None
    if "action_log" not in st.session_state:
        st.session_state.action_log = []
    if "transformation_log" not in st.session_state:
        st.session_state.transformation_log = []
    if "transform_metadata" not in st.session_state:
        st.session_state.transform_metadata = {}

def log_action(text: str):
    st.session_state.action_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {text}")

def log_transformation(text: str):
    st.session_state.transformation_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {text}")

def auto_recommendations(df: pd.DataFrame) -> dict:
    n = len(df)
    rec = {
        "remove_duplicates": df.duplicated().sum() > 0,
        "drop_high_missing_cols": False,
        "high_missing_threshold": 0.6,
        "numeric_impute_strategy": "Median",
        "categorical_impute_strategy": "Mode",
        "cap_outliers": True if len(df.select_dtypes(include=np.number).columns) else False,
    }
    miss_ratio = (df.isnull().sum() / max(1, n)).max()
    if miss_ratio >= 0.6:
        rec["drop_high_missing_cols"] = True
    return rec

def generate_pdf_report(df_before, df_after, df_transformed, df_anomaly, action_log, transformation_log):
    """Generate comprehensive PDF report with ALL 4 graphs, anomaly plot, and detailed analysis"""
    try:
        temp_dir = tempfile.mkdtemp()
        pdf_path = os.path.join(temp_dir, "data_cleaning_report.pdf")
        
        doc = SimpleDocTemplate(pdf_path, pagesize=letter, topMargin=0.4*inch, bottomMargin=0.4*inch)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=20,
            textColor=colors.HexColor('#2563eb'),
            spaceAfter=8
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=12,
            textColor=colors.HexColor('#2563eb'),
            spaceAfter=8,
            spaceBefore=8
        )
        
        # Title Page
        story.append(Paragraph("📊 DATA CLEANING & ANALYSIS REPORT", title_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph("1. EXECUTIVE SUMMARY", heading_style))
        summary_data = [
            ["Metric", "Before Cleaning", "After Cleaning", "Change"],
            ["Rows", str(df_before.shape[0]), str(df_after.shape[0]), f"{df_after.shape[0] - df_before.shape[0]:+d}"],
            ["Columns", str(df_before.shape[1]), str(df_after.shape[1]), f"{df_after.shape[1] - df_before.shape[1]:+d}"],
            ["Missing Values", str(int(df_before.isnull().sum().sum())), str(int(df_after.isnull().sum().sum())), 
             f"{int(df_after.isnull().sum().sum()) - int(df_before.isnull().sum().sum()):+d}"],
            ["Duplicate Rows", str(df_before.duplicated().sum()), str(df_after.duplicated().sum()), 
             f"{df_after.duplicated().sum() - df_before.duplicated().sum():+d}"],
        ]
        
        summary_table = Table(summary_data, colWidths=[1.3*inch, 1.1*inch, 1.1*inch, 1*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563eb')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8)
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph("2. CLEANING ACTIONS LOG (Steps 1-6)", heading_style))
        if action_log:
            action_text = "<br/>".join([f"• {action}" for action in action_log[:20]])
            story.append(Paragraph(action_text, styles['Normal']))
        else:
            story.append(Paragraph("No actions recorded.", styles['Normal']))
        story.append(Spacer(1, 0.15*inch))
        
        if transformation_log:
            story.append(Paragraph("3. TRANSFORMATIONS APPLIED (Step 5)", heading_style))
            trans_text = "<br/>".join([f"• {trans}" for trans in transformation_log])
            story.append(Paragraph(trans_text, styles['Normal']))
            story.append(Spacer(1, 0.15*inch))
        
        anomaly_section_num = "4" if transformation_log else "3"
        if df_anomaly is not None and "anomaly_label" in df_anomaly.columns:
            story.append(Paragraph(f"{anomaly_section_num}. ANOMALY DETECTION RESULTS (Step 6)", heading_style))
            anomaly_counts = df_anomaly["anomaly_label"].value_counts(dropna=True)
            normal_pts = anomaly_counts.get('Normal', 0)
            anomaly_pts = anomaly_counts.get('Anomaly', 0)
            anomaly_rate = (anomaly_pts / len(df_anomaly) * 100) if len(df_anomaly) > 0 else 0
            
            anomaly_text = f"""
            Normal Points: {normal_pts}<br/>
            Anomalies Detected: {anomaly_pts}<br/>
            Anomaly Rate: {anomaly_rate:.2f}%<br/>
            Detection Method: Isolation Forest
            """
            story.append(Paragraph(anomaly_text, styles['Normal']))
            story.append(Spacer(1, 0.15*inch))
        
        story.append(PageBreak())
        
        numeric_cols = df_after.select_dtypes(include=np.number).columns.tolist()
        
        if len(numeric_cols) >= 2:
            story.append(Paragraph("5. CORRELATION ANALYSIS", heading_style))
            try:
                # Before Correlation
                fig, ax = plt.subplots(figsize=(4.5, 3.5))
                fig.patch.set_facecolor('white')
                ax.set_facecolor('white')
                corr_before = df_before[numeric_cols].corr(numeric_only=True)
                sns.heatmap(corr_before, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, 
                           cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1)
                ax.set_title("Before Cleaning", fontsize=10, fontweight='bold')
                
                img_path = os.path.join(temp_dir, "corr_before.png")
                fig.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                
                # After Correlation
                fig, ax = plt.subplots(figsize=(4.5, 3.5))
                fig.patch.set_facecolor('white')
                ax.set_facecolor('white')
                corr_after = df_after[numeric_cols].corr(numeric_only=True)
                sns.heatmap(corr_after, annot=True, cmap="coolwarm", fmt=".2f", ax=ax,
                           cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1)
                ax.set_title("After Cleaning", fontsize=10, fontweight='bold')
                
                img_path_after = os.path.join(temp_dir, "corr_after.png")
                fig.savefig(img_path_after, dpi=100, bbox_inches='tight')
                plt.close(fig)
                
                story.append(Paragraph("Correlation Heatmap (Before vs After)", styles['Heading3']))
                
                # Two column layout for heatmaps
                table_data = [[RLImage(img_path, width=2.8*inch, height=2.2*inch),
                              RLImage(img_path_after, width=2.8*inch, height=2.2*inch)]]
                heatmap_table = Table(table_data)
                story.append(heatmap_table)
                story.append(Spacer(1, 0.2*inch))
            except Exception as e:
                story.append(Paragraph(f"Error: {str(e)}", styles['Normal']))
        
        if numeric_cols:
            story.append(Paragraph("6. DISTRIBUTION ANALYSIS", heading_style))
            
            for idx, col in enumerate(numeric_cols[:2]):
                try:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5))
                    fig.patch.set_facecolor('white')
                    
                    # Before
                    ax1.set_facecolor('white')
                    sns.histplot(df_before[col].dropna(), kde=True, ax=ax1, color="skyblue", bins=15)
                    ax1.set_title(f"Before: {col}", fontsize=9, fontweight='bold')
                    ax1.set_xlabel(col, fontsize=8)
                    ax1.set_ylabel('Frequency', fontsize=8)
                    
                    # After
                    ax2.set_facecolor('white')
                    sns.histplot(df_after[col].dropna(), kde=True, ax=ax2, color="lightgreen", bins=15)
                    ax2.set_title(f"After: {col}", fontsize=9, fontweight='bold')
                    ax2.set_xlabel(col, fontsize=8)
                    ax2.set_ylabel('Frequency', fontsize=8)
                    
                    img_path = os.path.join(temp_dir, f"dist_{idx}.png")
                    fig.savefig(img_path, dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    
                    story.append(RLImage(img_path, width=6*inch, height=2.3*inch))
                    story.append(Spacer(1, 0.1*inch))
                except Exception as e:
                    story.append(Paragraph(f"Error generating distribution for {col}: {str(e)}", styles['Normal']))
        
        story.append(PageBreak())
        
        if numeric_cols:
            story.append(Paragraph("7. BOX PLOT ANALYSIS", heading_style))
            
            try:
                col = numeric_cols[0]
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
                fig.patch.set_facecolor('white')
                
                ax1.set_facecolor('white')
                sns.boxplot(y=df_before[col], ax=ax1, color="skyblue")
                ax1.set_title(f"Before: {col}", fontsize=10, fontweight='bold')
                
                ax2.set_facecolor('white')
                sns.boxplot(y=df_after[col], ax=ax2, color="lightgreen")
                ax2.set_title(f"After: {col}", fontsize=10, fontweight='bold')
                
                img_path = os.path.join(temp_dir, "boxplot.png")
                fig.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                
                story.append(RLImage(img_path, width=5.5*inch, height=2.5*inch))
                story.append(Spacer(1, 0.2*inch))
            except Exception as e:
                story.append(Paragraph(f"Error: {str(e)}", styles['Normal']))
        
        story.append(Paragraph("8. MISSING VALUES ANALYSIS", heading_style))
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.5))
            fig.patch.set_facecolor('white')
            
            sns.heatmap(df_before.isnull(), cbar=False, cmap="RdYlGn_r", ax=ax1)
            ax1.set_title("Before Cleaning", fontsize=10, fontweight='bold')
            
            sns.heatmap(df_after.isnull(), cbar=False, cmap="RdYlGn_r", ax=ax2)
            ax2.set_title("After Cleaning", fontsize=10, fontweight='bold')
            
            img_path = os.path.join(temp_dir, "missing.png")
            fig.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            story.append(RLImage(img_path, width=6.5*inch, height=2*inch))
            story.append(Spacer(1, 0.2*inch))
        except Exception as e:
            story.append(Paragraph(f"Error: {str(e)}", styles['Normal']))
        
        story.append(PageBreak())
        
        if df_anomaly is not None and "anomaly_label" in df_anomaly.columns and len(numeric_cols) >= 2:
            story.append(Paragraph("9. ANOMALY DETECTION SCATTER PLOT", heading_style))
            try:
                x_col = numeric_cols[0]
                y_col = numeric_cols[1]
                
                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_facecolor('white')
                ax.set_facecolor('white')
                
                normal = df_anomaly[df_anomaly["anomaly_label"] == "Normal"]
                anomaly = df_anomaly[df_anomaly["anomaly_label"] == "Anomaly"]
                
                ax.scatter(normal[x_col], normal[y_col], c='blue', alpha=0.6, label='Normal', s=50)
                ax.scatter(anomaly[x_col], anomaly[y_col], c='red', alpha=0.8, label='Anomaly', s=100, marker='X')
                
                ax.set_xlabel(x_col, fontsize=10)
                ax.set_ylabel(y_col, fontsize=10)
                ax.set_title("Isolation Forest Anomaly Detection Results", fontsize=11, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
                
                img_path = os.path.join(temp_dir, "anomaly_scatter.png")
                fig.savefig(img_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                
                story.append(RLImage(img_path, width=5.5*inch, height=3.5*inch))
                story.append(Spacer(1, 0.2*inch))
            except Exception as e:
                story.append(Paragraph(f"Error: {str(e)}", styles['Normal']))
        
        story.append(PageBreak())
        
        story.append(Paragraph("10. DATA QUALITY METRICS", heading_style))
        if numeric_cols:
            numeric_data = [["Column", "Mean", "Std Dev", "Min", "Max", "Median"]]
            for col in numeric_cols[:12]:
                numeric_data.append([
                    col[:12],
                    f"{df_after[col].mean():.2f}",
                    f"{df_after[col].std():.2f}",
                    f"{df_after[col].min():.2f}",
                    f"{df_after[col].max():.2f}",
                    f"{df_after[col].median():.2f}"
                ])
            
            numeric_table = Table(numeric_data, colWidths=[1*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch])
            numeric_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563eb')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 7),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
            ]))
            story.append(numeric_table)
            story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph("REPORT COMPLETION SUMMARY", heading_style))
        completion_text = f"""
        Report generated successfully on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        Total actions performed: {len(action_log)}<br/>
        Transformations applied: {len(transformation_log)}<br/>
        All visualizations included: Correlation, Distribution, Box Plot, Missing Values, Anomaly Detection<br/>
        Data quality assessment complete and documented.
        """
        story.append(Paragraph(completion_text, styles['Normal']))
        
        # Build PDF
        doc.build(story)
        return pdf_path
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None

ensure_session_state()

# ===========================
# SIDEBARNAVIGATION
# ===========================
step = st.sidebar.radio(
    "Workflow Steps",
    ["Upload", "Recommendations", "Clean Data", "Visualize", "Transform", "Anomaly Detection", "Report & Download"],
    label_visibility="collapsed"
)

# ===========================
# STEP 1: UPLOAD
# ===========================
if step == "Upload":
    st.markdown('<span class="section-label">Step 1: Upload Dataset</span>', unsafe_allow_html=True)
    st.header("Upload your dataset")
    
    file = st.file_uploader("Upload CSV / XLSX / XLS", type=["csv", "xlsx", "xls"])
    if file is not None:
        try:
            if file.name.lower().endswith(".csv"):
                df = pd.read_csv(file, encoding='utf-8-sig')
            else:
                df = pd.read_excel(file)
            
            if df.empty:
                st.error("The uploaded file is empty.")
                st.stop()
            
            st.session_state.df_original = df.copy()
            st.session_state.df = df.copy()
            st.session_state.rec_settings = auto_recommendations(df)
            log_action(f"Dataset uploaded: {file.name} ({df.shape[0]} rows, {df.shape[1]} columns)")
            
            st.success("File loaded successfully")
            st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]} | **Duplicates:** {df.duplicated().sum()} | **Missing cells:** {df.isnull().sum().sum()}")
            st.dataframe(df.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

# ===========================
# STEP 2: RECOMMENDATIONS
# ===========================
elif step == "Recommendations":
    st.markdown('<span class="section-label">Step 2: Auto Recommendations</span>', unsafe_allow_html=True)
    if st.session_state.df is None:
        st.warning("Upload a dataset first.")
    else:
        df = st.session_state.df
        rec = st.session_state.rec_settings or auto_recommendations(df)
        
        st.header("Data Cleaning Recommendations")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Recommended Actions")
            st.write(f"• Remove duplicates: {'Yes' if rec['remove_duplicates'] else 'No'}")
            st.write(f"• Drop high missing columns: {'Yes' if rec['drop_high_missing_cols'] else 'No'}")
            st.write(f"• Missing threshold: {rec['high_missing_threshold']*100:.0f}%")
            st.write(f"• Numeric strategy: {rec['numeric_impute_strategy']}")
            st.write(f"• Categorical strategy: {rec['categorical_impute_strategy']}")
            st.write(f"• Cap outliers: {'Yes' if rec['cap_outliers'] else 'No'}")
        
        with col2:
            st.subheader("Dataset Statistics")
            st.metric("Total Rows", len(df))
            st.metric("Total Columns", len(df.columns))
            st.metric("Duplicate Rows", df.duplicated().sum())
            st.metric("Missing Values", df.isnull().sum().sum())

# ===========================
# STEP 3: CLEAN DATA
# ===========================
elif step == "Clean Data":
    st.markdown('<span class="section-label">Step 3: Clean Data</span>', unsafe_allow_html=True)
    if st.session_state.df is None:
        st.warning("Upload a dataset first.")
    else:
        df = st.session_state.df.copy()
        rec = st.session_state.rec_settings
        
        st.header("Data Cleaning")
        
        col1, col2 = st.columns(2)
        with col1:
            remove_dup = st.checkbox("Remove duplicates", value=rec["remove_duplicates"])
            drop_high_missing = st.checkbox("Drop columns with high missing", value=rec["drop_high_missing_cols"])
            thresh = st.slider("High missing threshold", 0.3, 0.95, float(rec["high_missing_threshold"]), 0.05)
        
        with col2:
            num_imp = st.selectbox("Numeric imputation", ["Median", "Mean", "None"], index=["Median", "Mean", "None"].index(rec["numeric_impute_strategy"]))
            cat_imp = st.selectbox("Categorical imputation", ["Mode", "Constant", "None"], index=["Mode", "Constant", "None"].index(rec["categorical_impute_strategy"]))
            cap_outliers = st.checkbox("Cap outliers (IQR)", value=rec["cap_outliers"])
        
        if st.button("Apply Cleaning", use_container_width=True):
            before_shape = df.shape
            
            # Drop high-missing columns
            if drop_high_missing:
                cols_to_drop = [c for c in df.columns if df[c].isnull().mean() > thresh]
                if cols_to_drop:
                    df = df.drop(columns=cols_to_drop)
                    log_action(f"Dropped {len(cols_to_drop)} columns with >{thresh:.0%} missing")
            
            # Remove duplicates
            if remove_dup:
                dup_before = df.duplicated().sum()
                df = df.drop_duplicates()
                if dup_before:
                    log_action(f"Removed {dup_before} duplicate rows")
            
            # Numeric imputation
            if num_imp != "None":
                for c in df.select_dtypes(include=np.number).columns:
                    if df[c].isnull().any():
                        val = df[c].median() if num_imp == "Median" else df[c].mean()
                        df[c] = df[c].fillna(val)
                log_action(f"Applied {num_imp} imputation to numeric columns")
            
            # Categorical imputation
            if cat_imp != "None":
                for c in df.select_dtypes(exclude=np.number).columns:
                    if df[c].isnull().any():
                        fill = df[c].mode().iloc[0] if (cat_imp == "Mode" and not df[c].mode().empty) else ""
                        df[c] = df[c].fillna(fill)
                log_action(f"Applied {cat_imp} imputation to categorical columns")
            
            # Outlier capping
            if cap_outliers:
                for c in df.select_dtypes(include=np.number).columns:
                    s = df[c].dropna()
                    if len(s) >= 5:
                        q1, q3 = s.quantile(0.25), s.quantile(0.75)
                        iqr = q3 - q1
                        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                        df[c] = np.clip(df[c], low, high)
                log_action("Capped outliers using IQR method")
            
            st.session_state.df = df
            st.success(f"Cleaning applied | Before: {before_shape} → After: {df.shape}")
            st.dataframe(df.head(20), use_container_width=True)

# ===========================
# STEP 4: VISUALIZE - ENHANCED WITH 20+ ROW CALCULATIONS
# ===========================
elif step == "Visualize":
    st.markdown('<span class="section-label">Step 4: Visualization & Calculations</span>', unsafe_allow_html=True)
    if st.session_state.df is None or st.session_state.df_original is None:
        st.warning("Upload and clean data first.")
    else:
        df_before = st.session_state.df_original.copy()
        df_after = st.session_state.df.copy()
        
        # Remove anomaly columns if present
        for col in ["anomaly", "anomaly_label"]:
            if col in df_after.columns:
                df_after = df_after.drop(columns=[col])
        
        numeric_cols = df_after.select_dtypes(include=np.number).columns.tolist()
        
        st.header("Detailed Visualization Calculations (20+ Rows Shown)")
        st.write("All visualizations now include comprehensive step-by-step calculations showing exactly how graphs are plotted.")
        
        viz_type = st.selectbox(
            "Select Visualization Type",
            ["Correlation Heatmap", "Box/Violin Plot", "Distribution Plot", "Missingness Map"]
        )
        
        def show_calc_table(title, df_calc, max_rows=25):
            """Display calculation table with proper formatting"""
            with st.expander(f"📊 {title}", expanded=True):
                if df_calc is None or df_calc.empty:
                    st.info("No data available")
                else:
                    display_rows = min(max_rows, len(df_calc))
                    st.write(f"**Showing {display_rows} of {len(df_calc)} rows**")
                    st.dataframe(df_calc.head(display_rows), use_container_width=True, height=600)
        
        # ===== CORRELATION HEATMAP =====
        if viz_type == "Correlation Heatmap":
            if len(numeric_cols) < 2:
                st.warning("Need at least 2 numeric columns.")
            else:
                st.subheader("Correlation Heatmap Analysis")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Before Cleaning**")
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(df_before[numeric_cols].corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                    st.pyplot(fig)
                    plt.close(fig)
                
                with c2:
                    st.write("**After Cleaning**")
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(df_after[numeric_cols].corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                    st.pyplot(fig)
                    plt.close(fig)
                
                st.write("---")
                st.write("### Detailed Pairwise Correlation Calculations")
                
                def build_corr_calc(from_df, col_x, col_y):
                    """Build correlation calculation with all 20+ rows"""
                    try:
                        pair = from_df[[col_x, col_y]].dropna()
                        if len(pair) < 2:
                            return None
                        
                        x = pair[col_x].astype(float).values
                        y = pair[col_y].astype(float).values
                        
                        mean_x = np.mean(x)
                        mean_y = np.mean(y)
                        
                        df_calc = pd.DataFrame({
                            "Row": range(1, len(x) + 1),
                            col_x: np.round(x, 4),
                            col_y: np.round(y, 4),
                            f"({col_x} - mean)": np.round(x - mean_x, 6),
                            f"({col_y} - mean)": np.round(y - mean_y, 6),
                            "Product": np.round((x - mean_x) * (y - mean_y), 6)
                        })
                        
                        sum_prod = np.sum((x - mean_x) * (y - mean_y))
                        sum_x_sq = np.sum((x - mean_x) ** 2)
                        sum_y_sq = np.sum((y - mean_y) ** 2)
                        corr = sum_prod / np.sqrt(sum_x_sq * sum_y_sq) if (sum_x_sq * sum_y_sq) > 0 else 0
                        
                        summary = pd.DataFrame([{
                            "Row": "SUMMARY",
                            col_x: f"μ={mean_x:.4f}",
                            col_y: f"μ={mean_y:.4f}",
                            f"({col_x} - mean)": f"Σ={np.sum(x - mean_x):.2f}",
                            f"({col_y} - mean)": f"Σ={np.sum(y - mean_y):.2f}",
                            "Product": f"Correlation = {corr:.6f}"
                        }])
                        
                        return pd.concat([df_calc, summary], ignore_index=True)
                    except Exception as e:
                        st.error(f"Error: {e}")
                        return None
                
                for i, col_x in enumerate(numeric_cols):
                    for col_y in numeric_cols[i + 1:]:
                        col1, col2 = st.columns(2)
                        with col1:
                            before_calc = build_corr_calc(df_before, col_x, col_y)
                            show_calc_table(f"Before: {col_x} vs {col_y}", before_calc)
                        with col2:
                            after_calc = build_corr_calc(df_after, col_x, col_y)
                            show_calc_table(f"After: {col_x} vs {col_y}", after_calc)
        
        # ===== BOX / VIOLIN PLOT =====
        elif viz_type == "Box/Violin Plot":
            if not numeric_cols:
                st.warning("No numeric columns available.")
            else:
                st.subheader("Box/Violin Plot Analysis")
                y_col = st.selectbox("Select Column", numeric_cols)
                plot_type = st.radio("Type", ["Boxplot", "Violin"], horizontal=True)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Before Cleaning**")
                    fig, ax = plt.subplots(figsize=(6, 5))
                    if plot_type == "Boxplot":
                        sns.boxplot(y=df_before[y_col], ax=ax)
                    else:
                        sns.violinplot(y=df_before[y_col], ax=ax)
                    st.pyplot(fig)
                    plt.close(fig)
                
                with c2:
                    st.write("**After Cleaning**")
                    fig, ax = plt.subplots(figsize=(6, 5))
                    if plot_type == "Boxplot":
                        sns.boxplot(y=df_after[y_col], ax=ax)
                    else:
                        sns.violinplot(y=df_after[y_col], ax=ax)
                    st.pyplot(fig)
                    plt.close(fig)
                
                st.write("---")
                
                def build_box_calc(df, col):
                    """Build box plot statistics with 20+ rows"""
                    try:
                        s = df[[col]].dropna()[col].astype(float).values
                        if len(s) < 2:
                            return None
                        
                        s_sorted = np.sort(s)
                        mean = np.mean(s)
                        median = np.median(s)
                        q1 = np.percentile(s, 25)
                        q3 = np.percentile(s, 75)
                        iqr = q3 - q1
                        std = np.std(s, ddof=1)
                        
                        df_calc = pd.DataFrame({
                            "Row": range(1, len(s) + 1),
                            col: np.round(s, 4),
                            "Sorted": np.round(s_sorted, 4),
                            "Deviation": np.round(s - mean, 6),
                            "Squared Dev": np.round((s - mean) ** 2, 6),
                            "Distance from Median": np.round(np.abs(s - median), 6)
                        })
                        
                        summary = pd.DataFrame([{
                            "Row": "STATS",
                            col: f"Count={len(s)}",
                            "Sorted": f"Min={np.min(s):.4f} | Max={np.max(s):.4f}",
                            "Deviation": f"Mean={mean:.6f}",
                            "Squared Dev": f"Q1={q1:.4f} | Q3={q3:.4f}",
                            "Distance from Median": f"Median={median:.4f} | IQR={iqr:.4f} | Std={std:.4f}"
                        }])
                        
                        return pd.concat([df_calc, summary], ignore_index=True)
                    except Exception as e:
                        st.error(f"Error: {e}")
                        return None
                
                col1, col2 = st.columns(2)
                with col1:
                    before_box = build_box_calc(df_before, y_col)
                    show_calc_table(f"Before: {y_col} Statistics", before_box)
                with col2:
                    after_box = build_box_calc(df_after, y_col)
                    show_calc_table(f"After: {y_col} Statistics", after_box)
        
        # ===== DISTRIBUTION PLOT =====
        elif viz_type == "Distribution Plot":
            if not numeric_cols:
                st.warning("No numeric columns available.")
            else:
                st.subheader("Distribution Analysis")
                col = st.selectbox("Select Column", numeric_cols)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Before Cleaning**")
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.histplot(df_before[col].dropna(), kde=True, ax=ax, bins=20)
                    st.pyplot(fig)
                    plt.close(fig)
                
                with c2:
                    st.write("**After Cleaning**")
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.histplot(df_after[col].dropna(), kde=True, ax=ax, bins=20)
                    st.pyplot(fig)
                    plt.close(fig)
                
                st.write("---")
                
                def build_dist_calc(df, col, n_bins=20):
                    """Build distribution calculations with bin analysis"""
                    try:
                        s = df[[col]].dropna()[col].astype(float).values
                        if len(s) < 1:
                            return None, None
                        
                        mean = np.mean(s)
                        median = np.median(s)
                        std = np.std(s, ddof=1)
                        
                        # Bin analysis
                        counts, bin_edges = np.histogram(s, bins=n_bins)
                        bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
                        
                        bin_df = pd.DataFrame({
                            "Bin": range(1, len(counts) + 1),
                            "Range": [f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})" for i in range(len(bin_edges)-1)],
                            "Center": np.round(bin_centers, 2),
                            "Frequency": counts,
                            "Freq %": np.round(counts / counts.sum() * 100, 2),
                            "Cumulative %": np.round(np.cumsum(counts) / counts.sum() * 100, 2)
                        })
                        
                        summary_bin = pd.DataFrame([{
                            "Bin": "TOTAL",
                            "Range": f"Count={len(s)}",
                            "Center": f"Mean={mean:.4f}",
                            "Frequency": f"Median={median:.4f}",
                            "Freq %": f"Std={std:.4f}",
                            "Cumulative %": f"Skew={pd.Series(s).skew():.4f}"
                        }])
                        bin_df = pd.concat([bin_df, summary_bin], ignore_index=True)
                        
                        # Value analysis
                        val_df = pd.DataFrame({
                            "Row": range(1, len(s) + 1),
                            col: np.round(s, 4),
                            "Deviation": np.round(s - mean, 6),
                            "Squared Dev": np.round((s - mean) ** 2, 6),
                            "Z-Score": np.round((s - mean) / std, 4) if std > 0 else 0,
                            "Percentile": np.round([sum(s <= v) / len(s) * 100 for v in s], 2)
                        })
                        
                        return bin_df, val_df
                    except Exception as e:
                        st.error(f"Error: {e}")
                        return None, None
                
                st.write("**Bin Analysis**")
                before_bin, before_val = build_dist_calc(df_before, col)
                after_bin, after_val = build_dist_calc(df_after, col)
                
                col1, col2 = st.columns(2)
                with col1:
                    show_calc_table(f"Before: {col} - Bins", before_bin)
                with col2:
                    show_calc_table(f"After: {col} - Bins", after_bin)
                
                st.write("**Value Analysis**")
                col1, col2 = st.columns(2)
                with col1:
                    show_calc_table(f"Before: {col} - Values", before_val)
                with col2:
                    show_calc_table(f"After: {col} - Values", after_val)
        
        # ===== MISSINGNESS MAP =====
        elif viz_type == "Missingness Map":
            st.subheader("Missingness Analysis")
            
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Before Cleaning**")
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.heatmap(df_before.isnull(), cbar=False, cmap="YlOrRd", ax=ax)
                st.pyplot(fig)
                plt.close(fig)
            
            with c2:
                st.write("**After Cleaning**")
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.heatmap(df_after.isnull(), cbar=False, cmap="YlOrRd", ax=ax)
                st.pyplot(fig)
                plt.close(fig)
            
            st.write("---")
            
            def build_missing_calc(df):
                """Build missingness breakdown with all columns"""
                try:
                    miss_list = []
                    for col in df.columns:
                        miss_count = df[col].isnull().sum()
                        total = len(df)
                        miss_pct = (miss_count / total * 100) if total > 0 else 0
                        
                        miss_list.append({
                            "Column": col,
                            "Type": str(df[col].dtype),
                            "Total": total,
                            "Valid": total - miss_count,
                            "Missing": miss_count,
                            "Missing %": round(miss_pct, 2),
                            "Valid %": round(100 - miss_pct, 2)
                        })
                    
                    return pd.DataFrame(miss_list)
                except Exception as e:
                    st.error(f"Error: {e}")
                    return None
            
            col1, col2 = st.columns(2)
            with col1:
                before_miss = build_missing_calc(df_before)
                show_calc_table("Before: Column-wise Missingness", before_miss)
            with col2:
                after_miss = build_missing_calc(df_after)
                show_calc_table("After: Column-wise Missingness", after_miss)

# ===========================
# STEP 5: TRANSFORM - WITH SEPARATE COLUMNS
# ===========================
elif step == "Transform":
    st.markdown('<span class="section-label">Step 5: Data Transformation</span>', unsafe_allow_html=True)
    if st.session_state.df is None:
        st.warning("Upload and clean data first.")
        st.stop()
    
    if "df_transformed" not in st.session_state or st.session_state.df_transformed is None:
        st.session_state.df_transformed = st.session_state.df.copy()
    
    df = st.session_state.df_transformed.copy()
    
    cat_cols = st.session_state.df.select_dtypes(exclude=np.number).columns.tolist()
    if cat_cols:
        st.subheader("Label Encoding (Convert Text to Numbers)")
        label_encode_cols = st.multiselect("Select categorical columns to encode", cat_cols)
        
        if label_encode_cols and st.button("Apply Label Encoding", use_container_width=True):
            try:
                for col in label_encode_cols:
                    encoder = LabelEncoder()
                    encoded_vals = encoder.fit_transform(df[col].astype(str))
                    df[f"{col}_encoded"] = encoded_vals
                    
                    mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
                    st.session_state.transform_metadata[f"{col}_mapping"] = mapping
                    log_transformation(f"Label Encoded: '{col}' → '{col}_encoded' (Original preserved)")
                
                st.session_state.df_transformed = df
                st.success(f"Encoded {len(label_encode_cols)} column(s). Original + encoded columns both saved.")
                st.write("**Preview (showing first 10 rows):**")
                st.dataframe(df[[col for c in label_encode_cols for col in [c, f"{c}_encoded"]]].head(10), use_container_width=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        st.subheader("Numeric Scaling (Normalize/Standardize)")
        scale_cols = st.multiselect("Select numeric columns to scale", numeric_cols)
        scale_type = st.radio("Method", ["Standard (Z-score)", "Min-Max (0-1)"], horizontal=True)
        
        if scale_cols and st.button("Apply Scaling", use_container_width=True):
            try:
                if scale_type == "Standard (Z-score)":
                    scaler = StandardScaler()
                    scaled_vals = scaler.fit_transform(df[scale_cols])
                    suffix = "_zscore"
                else:
                    scaler = MinMaxScaler()
                    scaled_vals = scaler.fit_transform(df[scale_cols])
                    suffix = "_minmax"
                
                for i, col in enumerate(scale_cols):
                    df[f"{col}{suffix}"] = scaled_vals[:, i]
                    log_transformation(f"Scaled: '{col}' → '{col}{suffix}' using {scale_type} (Original preserved)")
                
                st.session_state.df_transformed = df
                st.success(f"Scaled {len(scale_cols)} column(s). Original + scaled columns both saved.")
                st.write("**Preview (showing first 10 rows with original and scaled values):**")
                preview_cols = []
                for c in scale_cols:
                    preview_cols.extend([c, f"{c}{suffix}"])
                st.dataframe(df[preview_cols].head(10), use_container_width=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    st.write("---")
    st.subheader("Transformation Summary")
    if st.session_state.df_transformed is not None:
        orig_cols = st.session_state.df.shape[1]
        trans_cols = st.session_state.df_transformed.shape[1]
        new_cols = trans_cols - orig_cols
        st.success(f"Transformed data ready: {orig_cols} original columns + {new_cols} new transformation columns = {trans_cols} total")
    else:
        st.info("No transformations applied yet.")

# ===========================
# STEP 6: ANOMALY DETECTION
# ===========================
elif step == "Anomaly Detection":
    st.markdown('<span class="section-label">Step 6: Anomaly Detection</span>', unsafe_allow_html=True)
    
    if "df_transformed" in st.session_state and st.session_state.df_transformed is not None:
        df = st.session_state.df_transformed.copy()
        st.info("Using transformed dataset.")
    elif st.session_state.df is not None:
        df = st.session_state.df.copy()
        st.info("Using cleaned dataset.")
    else:
        st.warning("Upload and clean data first.")
        st.stop()
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns.")
        st.stop()
    
    st.header("Anomaly Detection with Isolation Forest")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        x_col = st.selectbox("X-axis", numeric_cols)
    with col2:
        y_col = st.selectbox("Y-axis", [c for c in numeric_cols if c != x_col])
    with col3:
        contamination = st.slider("Contamination", 0.01, 0.25, 0.05, 0.01)
    
    if st.button("Run Detection", use_container_width=True):
        try:
            clean_subset = df[[x_col, y_col]].dropna()
            if len(clean_subset) < 10:
                st.warning("Too few valid rows.")
            else:
                iso = IsolationForest(contamination=contamination, random_state=42)
                preds = iso.fit_predict(clean_subset.values)
                
                df["anomaly"] = np.nan
                df.loc[clean_subset.index, "anomaly"] = preds
                df["anomaly_label"] = df["anomaly"].map({1: "Normal", -1: "Anomaly"})
                
                st.session_state.df_anomaly = df
                
                counts = df["anomaly_label"].value_counts(dropna=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Normal", counts.get("Normal", 0))
                with col2:
                    st.metric("Anomalies", counts.get("Anomaly", 0))
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(data=df, x=x_col, y=y_col, hue="anomaly_label",
                               palette={"Normal": "blue", "Anomaly": "red"}, alpha=0.7, ax=ax)
                st.pyplot(fig)
                plt.close(fig)
                
                st.success("Anomaly detection complete!")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ===========================
# STEP 7: REPORT & DOWNLOAD
# ===========================
elif step == "Report & Download":
    st.markdown('<span class="section-label">Step 7: Report & Downloads</span>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("Upload and clean data first.")
        st.stop()
    
    st.header("Download Results")
    
    df_before = st.session_state.df_original.copy() if st.session_state.df_original is not None else pd.DataFrame()
    df_after = st.session_state.df.copy()
    df_transformed = st.session_state.df_transformed if "df_transformed" in st.session_state else None
    df_anomaly = st.session_state.df_anomaly if "df_anomaly" in st.session_state else None
    
    st.subheader("Summary Statistics")
    summary_data = {
        "Metric": ["Rows", "Columns", "Missing Cells", "Duplicates"],
        "Before": [df_before.shape[0], df_before.shape[1], int(df_before.isnull().sum().sum()), int(df_before.duplicated().sum())],
        "After": [df_after.shape[0], df_after.shape[1], int(df_after.isnull().sum().sum()), int(df_after.duplicated().sum())]
    }
    if df_transformed is not None:
        summary_data["Metric"].append("Transformation Columns")
        summary_data["Before"].append("-")
        summary_data["After"].append(df_transformed.shape[1] - df_after.shape[1])
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Action Log
    if st.session_state.action_log:
        st.subheader("Action Log (Steps 1-6)")
        log_df = pd.DataFrame({"Action": st.session_state.action_log})
        st.dataframe(log_df, use_container_width=True)
    
    if st.session_state.transformation_log:
        st.subheader("Transformations Applied (Step 5)")
        trans_df = pd.DataFrame({"Transformation": st.session_state.transformation_log})
        st.dataframe(trans_df, use_container_width=True)
    
    # Downloads
    st.subheader("Download CSV Files")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        csv = df_after.to_csv(index=False).encode("utf-8")
        st.download_button("Download Cleaned Data", csv, "cleaned_data.csv", "text/csv", use_container_width=True)
    
    with col2:
        if df_transformed is not None and df_transformed.shape[1] > df_after.shape[1]:
            csv = df_transformed.to_csv(index=False).encode("utf-8")
            num_new_cols = df_transformed.shape[1] - df_after.shape[1]
            st.download_button(f"Download Transformed Data\n({num_new_cols} new columns)", csv, "transformed_data.csv", "text/csv", use_container_width=True)
        else:
            st.info("No transformations applied")
    
    with col3:
        if df_anomaly is not None and "anomaly_label" in df_anomaly.columns:
            csv = df_anomaly.to_csv(index=False).encode("utf-8")
            st.download_button("Download Anomaly Data", csv, "anomaly_data.csv", "text/csv", use_container_width=True)
        else:
            st.info("No anomaly data")
    
    st.subheader("Generate PDF Report")
    if st.button("Generate Comprehensive PDF Report", use_container_width=True):
        with st.spinner("Generating PDF report with all graphs, calculations, and analysis..."):
            pdf_path = generate_pdf_report(
                df_before, 
                df_after, 
                df_transformed if df_transformed is not None else df_after,
                df_anomaly,
                st.session_state.action_log,
                st.session_state.transformation_log
            )
            
            if pdf_path and os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "Download PDF Report",
                        f,
                        "data_cleaning_report.pdf",
                        "application/pdf",
                        use_container_width=True
                    )
                st.success("PDF report generated successfully with all sections and visualizations!")
            else:
                st.error("Failed to generate PDF report")
