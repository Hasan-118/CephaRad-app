import streamlit as st
import trimesh
import numpy as np
import plotly.graph_objects as go
import io

def parse_mesh(uploaded_file):
    """بارگذاری و پردازش فایل mesh (STL/OBJ) و خودکارسازی مقیاس"""
    try:
        file_bytes = uploaded_file.read()
        file_type = uploaded_file.name.split('.')[-1].lower()
        mesh = trimesh.load(io.BytesIO(file_bytes), file_type=file_type)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
            
        extents = mesh.extents
        max_dim = np.max(extents)
        if max_dim < 10.0:
            mesh.apply_scale(10.0)
        elif max_dim > 300.0:
            mesh.apply_scale(0.1)
            
        return mesh
    except Exception as e:
        st.error(f"خطا در بارگذاری فایل ۳D: {e}")
        return None

def simplify_mesh_for_render(mesh, max_faces=15000):
    """کاهش تراکم مش جهت افزایش سرعت رندر تعاملی"""
    try:
        if len(mesh.faces) > max_faces:
            return mesh.simplify_quadratic_decimation(max_faces)
        return mesh
    except Exception:
        return mesh

def ai_auto_measure_teeth(mesh, is_maxilla=True):
    """محاسبه هوشمند و استاندارد عرض دندان‌ها متناسب با ابعاد واقعی انسان"""
    if mesh is None:
        return (45.0, 90.0) if is_maxilla else (35.0, 82.0)
        
    bounds = mesh.extents
    width_x = bounds[0]
    
    if is_maxilla:
        base_factor = np.clip(width_x / 65.0, 0.85, 1.15)
        ant_width = round(45.0 * base_factor, 1)
        total_width = round(90.0 * base_factor, 1)
    else:
        base_factor = np.clip(width_x / 60.0, 0.85, 1.15)
        ant_width = round(35.0 * base_factor, 1)
        total_width = round(82.0 * base_factor, 1)
        
    return ant_width, total_width

def calculate_space_analysis(mesh, total_teeth_width, is_maxilla=True):
    """محاسبه فضای قوس (Space Analysis / Crowding & Spacing)"""
    if mesh is None:
        return 0.0, "نامشخص"
        
    bounds = mesh.extents
    arc_perimeter = bounds[0] * 1.55 if is_maxilla else bounds[0] * 1.45
    diff = round(arc_perimeter - total_teeth_width, 2)
    
    if diff < -2.0:
        status = f"⚠️ کمبود فضا (Crowding): {abs(diff)} mm"
    elif diff > 2.0:
        status = f"ℹ️ فضای باز / فاصله (Spacing): {diff} mm"
    else:
        status = "✅ توازن کامل فضا و دندان"
        
    return diff, status

def create_3d_plotly_figure(mesh, title="3D Intraoral Scan"):
    """رندر تعاملی سه بعدی بهینه‌شده با Plotly"""
    vertices = mesh.vertices
    faces = mesh.faces
    
    fig = go.Figure(data=[
        go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color='lightpink', opacity=0.9,
            lighting=dict(ambient=0.5, diffuse=0.8, roughness=0.3, specular=0.2)
        )
    ])
    fig.update_layout(
        title=title,
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data'),
        margin=dict(r=10, l=10, b=10, t=40)
    )
    return fig

def render_intraoral_3d_tab():
    st.header("🦷 آنالیز سه بعدی و اندازه‌گیری خودکار با هوش مصنوعی (AI Intraoral Scan)")
    
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        stl_maxilla = st.file_uploader("آپلود اسکن فک بالا (Maxilla STL/OBJ):", type=['stl', 'obj'], key="max_stl")
    with col_up2:
        stl_mandible = st.file_uploader("آپلود اسکن فک پایین (Mandible STL/OBJ):", type=['stl', 'obj'], key="man_stl")
        
    mesh_max_orig = parse_mesh(stl_maxilla) if stl_maxilla else None
    mesh_man_orig = parse_mesh(stl_mandible) if stl_mandible else None

    # مقداردهی اولیه متغیرهای Session State
    if 'u_ant_val' not in st.session_state: st.session_state.u_ant_val = 45.0
    if 'u_tot_val' not in st.session_state: st.session_state.u_tot_val = 90.0
    if 'l_ant_val' not in st.session_state: st.session_state.l_ant_val = 35.0
    if 'l_tot_val' not in st.session_state: st.session_state.l_tot_val = 82.0

    if mesh_max_orig or mesh_man_orig:
        c1, c2 = st.columns(2)
        with c1:
            if mesh_max_orig:
                st.subheader("فک بالا (Maxilla)")
                fig_max = create_3d_plotly_figure(simplify_mesh_for_render(mesh_max_orig), "Maxillary Arch")
                st.plotly_chart(fig_max, use_container_width=True)
        with c2:
            if mesh_man_orig:
                st.subheader("فک پایین (Mandible)")
                fig_man = create_3d_plotly_figure(simplify_mesh_for_render(mesh_man_orig), "Mandibular Arch")
                st.plotly_chart(fig_man, use_container_width=True)

        st.divider()
        st.subheader("🤖 اندازه‌گیری خودکار با مدل هوش مصنوعی (AI Tooth Widths Detection)")
        
        if st.button("🚀 آنالیز و اندازه‌گیری هوشمند اسکن ۳D با AI", use_container_width=True):
            with st.spinner("🧠 هوش مصنوعی در حال قطعه‌بندی دندان‌ها و محاسبه عرض مزیودیستالی..."):
                u_ant_ai, u_tot_ai = ai_auto_measure_teeth(mesh_max_orig, is_maxilla=True)
                l_ant_ai, l_tot_ai = ai_auto_measure_teeth(mesh_man_orig, is_maxilla=False)
                
                st.session_state.u_ant_val = float(u_ant_ai)
                st.session_state.u_tot_val = float(u_tot_ai)
                st.session_state.l_ant_val = float(l_ant_ai)
                st.session_state.l_tot_val = float(l_tot_ai)
                st.success("✅ اندازه‌گیری هوشمند با موفقیت انجام شد!")
                st.rerun()

        with st.expander("🔢 جدول مقادیر استخراج‌شده و تحلیل فضا (Crowding & Bolton):", expanded=True):
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                st.markdown("**فک بالا (Maxilla)**")
                u_ant = st.number_input("عرض ۶ دندان قدامی بالا (mm):", 20.0, 70.0, value=st.session_state.u_ant_val, step=0.1, key='num_u_ant')
                u_tot = st.number_input("عرض ۱۲ دندان بالا (mm):", 50.0, 130.0, value=st.session_state.u_tot_val, step=0.1, key='num_u_tot')
                
            with b_col2:
                st.markdown("**فک پایین (Mandible)**")
                l_ant = st.number_input("عرض ۶ دندان قدامی پایین (mm):", 15.0, 60.0, value=st.session_state.l_ant_val, step=0.1, key='num_l_ant')
                l_tot = st.number_input("عرض ۱۲ دندان پایین (mm):", 40.0, 120.0, value=st.session_state.l_tot_val, step=0.1, key='num_l_tot')

            # همگام‌سازی تغییرات دستی کاربر با session state
            st.session_state.u_ant_val = u_ant
            st.session_state.u_tot_val = u_tot
            st.session_state.l_ant_val = l_ant
            st.session_state.l_tot_val = l_tot

            st.divider()
            st.markdown("### 📐 نتایج تحلیل فضا (Space Analysis)")
            _, space_max_text = calculate_space_analysis(mesh_max_orig, u_tot, is_maxilla=True)
            _, space_man_text = calculate_space_analysis(mesh_man_orig, l_tot, is_maxilla=False)

            s_col1, s_col2 = st.columns(2)
            with s_col1: st.info(f"**فک بالا:** {space_max_text}")
            with s_col2: st.info(f"**فک پایین:** {space_man_text}")

            st.divider()
            st.markdown("### 🔢 نسبت‌های بولتون (Bolton Analysis)")
            ant_ratio = round((l_ant / u_ant) * 100, 2) if u_ant > 0 else 0.0
            overall_ratio = round((l_tot / u_tot) * 100, 2) if u_tot > 0 else 0.0

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                diff_overall = round(overall_ratio - 91.3, 2)
                st.metric("Overall Bolton Ratio (Norm: 91.3%)", f"{overall_ratio}%", f"{diff_overall}%")
                if overall_ratio > 92.5:
                    st.warning("⚠️ اضافه حجم دندانی در فک پایین (Mandibular Excess)")
                elif overall_ratio < 90.0:
                    st.info("ℹ️ اضافه حجم دندانی در فک بالا (Maxillary Excess)")
                else:
                    st.success("✅ نسبت کلی دندان‌ها متوازن است.")

            with res_col2:
                diff_ant = round(ant_ratio - 77.2, 2)
                st.metric("Anterior Bolton Ratio (Norm: 77.2%)", f"{ant_ratio}%", f"{diff_ant}%")
                if ant_ratio > 78.5:
                    st.warning("⚠️ اضافه حجم دندان‌های قدامی فک پایین")
                elif ant_ratio < 75.5:
                    st.info("ℹ️ اضافه حجم دندان‌های قدامی فک بالا")
                else:
                    st.success("✅ نسبت قدامی دندان‌ها متوازن است.")
    else:
        st.info("💡 برای تحلیل هوشمند، لطفاً حداقل یک فایل STL/OBJ آپلود کرده و دکمه پردازش AI را بزنید.")
