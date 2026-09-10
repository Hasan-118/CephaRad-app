import streamlit as st
import numpy as np
import plotly.graph_objects as go
import trimesh
import io

# ==========================================
# 1. توابع محاسباتی آنالیز بولتون (Bolton Logic)
# ==========================================
def calculate_bolton_analysis(mandible_widths, maxilla_widths):
    """
    محاسبه آنالیز بولتون بر اساس عرض مزیودیستال دندان‌ها (به میلی‌متر)
    mandible_widths: لیست عرض دندان‌های فک پایین [L6 to R6] (12 دندان)
    maxilla_widths: لیست عرض دندان‌های فک بالا [L6 to R6] (12 دندان)
    """
    # 6 دندان قدامی (از نیش تا نیش - نمایه 3 تا 8)
    mand_ant = sum(mandible_widths[3:9])
    max_ant = sum(maxilla_widths[3:9])
    
    # 12 دندان کلی (از آسیاب اول تا آسیاب اول - نمایه 0 تا 11)
    mand_overall = sum(mandible_widths)
    max_overall = sum(maxilla_widths)
    
    # نسبت‌های بولتون
    ant_ratio = (mand_ant / max_ant) * 100 if max_ant > 0 else 0
    overall_ratio = (mand_overall / max_overall) * 100 if max_overall > 0 else 0
    
    # مقادیر استاندارد بالینی (Norms)
    ANT_NORM = 77.2
    OVERALL_NORM = 91.3
    
    # محاسبه عدم هماهنگی (Discrepancy)
    ant_diff = mand_ant - (max_ant * (ANT_NORM / 100))
    overall_diff = mand_overall - (max_overall * (OVERALL_NORM / 100))
    
    return {
        "ant_ratio": round(ant_ratio, 2),
        "overall_ratio": round(overall_ratio, 2),
        "ant_diff_mm": round(ant_diff, 2),
        "overall_diff_mm": round(overall_diff, 2),
        "mand_ant_sum": round(mand_ant, 2),
        "max_ant_sum": round(max_ant, 2),
        "mand_overall_sum": round(mand_overall, 2),
        "max_overall_sum": round(max_overall, 2)
    }

# ==========================================
# 2. تابع رندر و نمایش فایل STL در Plotly
# ==========================================
def render_stl_mesh(file_bytes, color='lightblue', opacity=1.0):
    """
    تبدیل فایل STL به Mesh سه بعدی قابل رویت در Plotly
    """
    mesh = trimesh.load(io.BytesIO(file_bytes), file_type='stl')
    vertices = mesh.vertices
    faces = mesh.faces
    
    fig = go.Figure(data=[
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=color,
            opacity=opacity,
            flatshading=True
        )
    ])
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        ),
        margin=dict(r=0, l=0, b=0, t=0),
        height=500
    )
    return fig

# ==========================================
# 3. رابط کاربری بخش اسکن داخل دهانی
# ==========================================
def render_intraoral_3d_tab():
    st.header("🦷 آنالیز سه بعدی اسکن داخل دهانی (3D Intraoral Scan Analysis)")
    st.markdown("---")
    
    col_upload1, col_upload2 = st.columns(2)
    
    with col_upload1:
        maxilla_file = st.file_uploader("آپلود فایل STL فک بالا (Maxilla)", type=["stl"], key="max_stl")
    with col_upload2:
        mandible_file = st.file_uploader("آپلود فایل STL فک پایین (Mandible)", type=["stl"], key="mand_stl")
        
    st.markdown("---")
    
    # نمایش سه‌بعدی مدل‌ها
    col_vis1, col_vis2 = st.columns(2)
    
    if maxilla_file is not None:
        with col_vis1:
            st.subheader("مدل سه‌بعدی فک بالا")
            fig_max = render_stl_mesh(maxilla_file.getvalue(), color='coral')
            st.plotly_chart(fig_max, use_container_width=True)
            
    if mandible_file is not None:
        with col_vis2:
            st.subheader("مدل سه‌بعدی فک پایین")
            fig_mand = render_stl_mesh(mandible_file.getvalue(), color='teal')
            st.plotly_chart(fig_mand, use_container_width=True)

    st.markdown("---")
    st.subheader("📐 ورودی عرض دیامتر دندان‌ها جهت محاسبه آنالیز بولتون (به میلی‌متر)")
    
    tooth_labels = ["6 (آسیاب اول راست)", "5", "4", "3 (نیش راست)", "2", "1 (سانترال)",
                    "1 (سانترال)", "2", "3 (نیش چپ)", "4", "5", "6 (آسیاب اول چپ)"]
    
    tab_max, tab_mand = st.tabs(["دندان‌های فک بالا", "دندان‌های فک پایین"])
    
    max_widths = []
    mand_widths = []
    
    # مقادیر پیش‌فرض استاندارد بالینی (Default Averages)
    default_max = [10.0, 7.0, 7.0, 7.5, 6.5, 8.5, 8.5, 6.5, 7.5, 7.0, 7.0, 10.0]
    default_mand = [10.5, 7.0, 7.0, 7.0, 5.5, 5.0, 5.0, 5.5, 7.0, 7.0, 7.0, 10.5]
    
    with tab_max:
        cols_m1 = st.columns(6)
        cols_m2 = st.columns(6)
        all_cols_m = cols_m1 + cols_m2
        for idx, col in enumerate(all_cols_m):
            val = col.number_input(f"بالا - {tooth_labels[idx]}", min_value=0.0, max_value=20.0, 
                                   value=default_max[idx], step=0.1, key=f"m_{idx}")
            max_widths.append(val)
            
    with tab_mand:
        cols_d1 = st.columns(6)
        cols_d2 = st.columns(6)
        all_cols_d = cols_d1 + cols_d2
        for idx, col in enumerate(all_cols_d):
            val = col.number_input(f"پایین - {tooth_labels[idx]}", min_value=0.0, max_value=20.0, 
                                   value=default_mand[idx], step=0.1, key=f"d_{idx}")
            mand_widths.append(val)

    if st.button("📊 محاسبه آنالیز بولتون و گزارش بالینی", type="primary"):
        res = calculate_bolton_analysis(mand_widths, max_widths)
        
        st.markdown("### 📋 نتایج آنالیز بولتون (Bolton Results)")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Anterior Ratio", f"{res['ant_ratio']}%", "نرمال: 77.2%")
        c2.metric("Overall Ratio", f"{res['overall_ratio']}%", "نرمال: 91.3%")
        c3.metric("مجموع قدامی فک پایین", f"{res['mand_ant_sum']} mm")
        c4.metric("مجموع قدامی فک بالا", f"{res['max_ant_sum']} mm")
        
        # تفسیر بالینی
        st.markdown("#### 🩺 تفسیر بالینی و پیشنهاد طرح درمان:")
        
        if abs(res['ant_ratio'] - 77.2) < 0.5:
            st.success("✅ **نسبت قدامی نرمال:** هماهنگی کامل بین سایز دندان‌های قدامی فک بالا و پایین وجود دارد.")
        elif res['ant_ratio'] > 77.2:
            st.warning(f"⚠️ **اضافه بافت قدامی فک پایین (یا کمبود فک بالا):** مقدار اضافی حدود **{abs(res['ant_diff_mm'])} میلی‌متر** در فک پایین است. (راهکار: IPR/استریپینگ در فک پایین یا بیلدآپ/کامپوزیت در فک بالا)")
        else:
            st.warning(f"⚠️ **اضافه بافت قدامی فک بالا (یا کمبود فک پایین):** مقدار اضافی در فک بالا است. (راهکار: IPR فک بالا یا اصلاح فرم دندان‌های فک پایین)")

# برای اجرای مستقل جهت تست ماژول
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Aariz 3D Intraoral Station")
    render_intraoral_3d_tab()
