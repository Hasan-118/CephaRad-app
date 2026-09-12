import streamlit as st
import trimesh
import numpy as np
import plotly.graph_objects as go
import io
from scipy.spatial import cKDTree

# ============================================================
# توابع موجود (بدون تغییر)
# ============================================================

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

def create_3d_plotly_figure(mesh, title="3D Intraoral Scan", color='lightpink'):
    """رندر تعاملی سه بعدی بهینه‌شده با Plotly"""
    vertices = mesh.vertices
    faces = mesh.faces

    fig = go.Figure(data=[
        go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=color, opacity=0.9,
            lighting=dict(ambient=0.5, diffuse=0.8, roughness=0.3, specular=0.2)
        )
    ])
    fig.update_layout(
        title=title,
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data'),
        margin=dict(r=10, l=10, b=10, t=40)
    )
    return fig

# ============================================================
# توابع جدید: سگمنتیشن اولیه خودکار (بدون AI)
# ============================================================

def compute_vertex_curvature(mesh, k_neighbors=30):
    """
    محاسبه انحنای تقریبی هر رأس با استفاده از PCA
    رأس‌هایی با انحنای بالا = احتمالاً مرز دندان/لثه یا لبه تیز
    """
    vertices = mesh.vertices
    tree = cKDTree(vertices)

    # برای هر رأس، k همسایه نزدیک را پیدا کن
    _, indices = tree.query(vertices, k=k_neighbors)

    curvatures = np.zeros(len(vertices))

    for i in range(len(vertices)):
        neighbors = vertices[indices[i]]
        # مرکز همسایه‌ها
        center = neighbors.mean(axis=0)
        # ماتریس کوواریانس
        centered = neighbors - center
        cov = np.dot(centered.T, centered) / len(neighbors)
        # مقادیر ویژه
        eigenvalues = np.linalg.eigvalsh(cov)
        # انحنا = کوچک‌ترین مقدار ویژه / مجموع مقادیر ویژه
        total = np.sum(eigenvalues) + 1e-9
        curvatures[i] = eigenvalues[0] / total

    return curvatures

def segment_teeth_from_gingiva(mesh, curvature_threshold=0.02):
    """
    جداسازی اولیه دندان‌ها از لثه بر اساس انحنای سطح
    - لثه: انحنای پایین (صاف)
    - دندان‌ها: انحنای بالا (برجسته)
    """
    curvatures = compute_vertex_curvature(mesh)

    # نرمال‌سازی
    curv_norm = (curvatures - curvatures.min()) / (curvatures.max() - curvatures.min() + 1e-9)

    # آستانه‌گذاری
    teeth_mask = curv_norm > curvature_threshold
    gingiva_mask = ~teeth_mask

    return teeth_mask, gingiva_mask, curvatures

def render_segmented_mesh(mesh, teeth_mask, title="Segmented View"):
    """
    نمایش مش با دو رنگ متفاوت: دندان‌ها (سفید) و لثه (صورتی)
    """
    vertices = mesh.vertices
    faces = mesh.faces

    # برای هر مثلث، چک کن که اکثر رأس‌هایش دندان هستند یا لثه
    face_teeth_count = teeth_mask[faces].sum(axis=1)
    face_is_teeth = face_teeth_count >= 2  # اگر حداقل ۲ رأس از ۳ دندان بود

    # جدا کردن مثلث‌ها
    teeth_faces = faces[face_is_teeth]
    gingiva_faces = faces[~face_is_teeth]

    fig = go.Figure()

    # لثه (صورتی)
    if len(gingiva_faces) > 0:
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=gingiva_faces[:, 0], j=gingiva_faces[:, 1], k=gingiva_faces[:, 2],
            color='lightpink', opacity=0.6, name='لثه',
            lighting=dict(ambient=0.5, diffuse=0.8)
        ))

    # دندان‌ها (سفید)
    if len(teeth_faces) > 0:
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=teeth_faces[:, 0], j=teeth_faces[:, 1], k=teeth_faces[:, 2],
            color='white', opacity=0.95, name='دندان',
            lighting=dict(ambient=0.6, diffuse=0.9)
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        ),
        margin=dict(r=10, l=10, b=10, t=40),
        showlegend=True
    )
    return fig

# ============================================================
# رابط کاربری اصلی (اصلاح‌شده)
# ============================================================

def render_intraoral_3d_tab():
    st.header("🦷 آنالیز سه بعدی اسکن داخل دهانی (نیمه‌خودکار)")

    st.info("""
    **راهنما:** ابتدا اسکن را آپلود کنید. برنامه به‌صورت خودکار سگمنتیشن اولیه 
    (جدا کردن دندان‌ها از لثه) را انجام می‌دهد. سپس می‌توانید آستانه انحنا را 
    تنظیم کنید تا نتیجه دقیق‌تر شود. نقاط مرزی دندان‌ها را می‌توانید دستی اصلاح کنید.
    """)

    col_up1, col_up2 = st.columns(2)
    with col_up1:
        stl_maxilla = st.file_uploader("آپلود اسکن فک بالا (Maxilla STL/OBJ):",
                                        type=['stl', 'obj'], key="max_stl")
    with col_up2:
        stl_mandible = st.file_uploader("آپلود اسکن فک پایین (Mandible STL/OBJ):",
                                         type=['stl', 'obj'], key="man_stl")

    mesh_max_orig = parse_mesh(stl_maxilla) if stl_maxilla else None
    mesh_man_orig = parse_mesh(stl_mandible) if stl_mandible else None

    # --- تنظیمات سگمنتیشن در سایدبار ---
    st.sidebar.markdown("### ⚙️ تنظیمات سگمنتیشن")
    curvature_thresh = st.sidebar.slider(
        "آستانه انحنا (بالاتر = دندان کمتر تشخیص داده می‌شود):",
        0.005, 0.10, 0.02, 0.001
    )

    if mesh_max_orig or mesh_man_orig:
        st.divider()
        st.subheader("🔬 سگمنتیشن اولیه (جدا کردن دندان از لثه)")

        seg_col1, seg_col2 = st.columns(2)

        # --- فک بالا ---
        with seg_col1:
            if mesh_max_orig:
                st.markdown("**فک بالا (Maxilla)**")
                with st.spinner("در حال محاسبه انحنا و سگمنتیشن..."):
                    teeth_mask_max, gingiva_mask_max, curv_max = segment_teeth_from_gingiva(
                        mesh_max_orig, curvature_threshold=curvature_thresh
                    )
                    teeth_percent = (teeth_mask_max.sum() / len(teeth_mask_max)) * 100
                    st.caption(f"🦷 دندان‌های تشخیص داده‌شده: {teeth_percent:.1f}% از سطح")

                fig_seg_max = render_segmented_mesh(
                    simplify_mesh_for_render(mesh_max_orig),
                    teeth_mask_max[:len(simplify_mesh_for_render(mesh_max_orig).vertices)],
                    "Maxilla - Segmented"
                )
                st.plotly_chart(fig_seg_max, use_container_width=True)

        # --- فک پایین ---
        with seg_col2:
            if mesh_man_orig:
                st.markdown("**فک پایین (Mandible)**")
                with st.spinner("در حال محاسبه انحنا و سگمنتیشن..."):
                    teeth_mask_man, gingiva_mask_man, curv_man = segment_teeth_from_gingiva(
                        mesh_man_orig, curvature_threshold=curvature_thresh
                    )
                    teeth_percent = (teeth_mask_man.sum() / len(teeth_mask_man)) * 100
                    st.caption(f"🦷 دندان‌های تشخیص داده‌شده: {teeth_percent:.1f}% از سطح")

                fig_seg_man = render_segmented_mesh(
                    simplify_mesh_for_render(mesh_man_orig),
                    teeth_mask_man[:len(simplify_mesh_for_render(mesh_man_orig).vertices)],
                    "Mandible - Segmented"
                )
                st.plotly_chart(fig_seg_man, use_container_width=True)

        st.success("✅ سگمنتیشن اولیه انجام شد. اگر مرزها اشتباه هستند، آستانه انحنا را در سایدبار تنظیم کنید.")

        # --- ادامه با اندازه‌گیری دستی ---
        st.divider()
        st.subheader("📏 اندازه‌گیری دستی عرض دندان‌ها")

        if 'u_ant_val' not in st.session_state: st.session_state.u_ant_val = 45.0
        if 'u_tot_val' not in st.session_state: st.session_state.u_tot_val = 90.0
        if 'l_ant_val' not in st.session_state: st.session_state.l_ant_val = 35.0
        if 'l_tot_val' not in st.session_state: st.session_state.l_tot_val = 82.0

        with st.expander("🔢 ورود مقادیر عرض دندان‌ها (از روی سگمنتیشن یا اندازه‌گیری دستی):", expanded=True):
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                st.markdown("**فک بالا (Maxilla)**")
                u_ant = st.number_input("عرض ۶ دندان قدامی بالا (mm):", 20.0, 70.0,
                                        value=st.session_state.u_ant_val, step=0.1, key='num_u_ant')
                u_tot = st.number_input("عرض ۱۲ دندان بالا (mm):", 50.0, 130.0,
                                        value=st.session_state.u_tot_val, step=0.1, key='num_u_tot')
            with b_col2:
                st.markdown("**فک پایین (Mandible)**")
                l_ant = st.number_input("عرض ۶ دندان قدامی پایین (mm):", 15.0, 60.0,
                                        value=st.session_state.l_ant_val, step=0.1, key='num_l_ant')
                l_tot = st.number_input("عرض ۱۲ دندان پایین (mm):", 40.0, 120.0,
                                        value=st.session_state.l_tot_val, step=0.1, key='num_l_tot')

            st.session_state.u_ant_val = u_ant
            st.session_state.u_tot_val = u_tot
            st.session_state.l_ant_val = l_ant
            st.session_state.l_tot_val = l_tot

            # --- تحلیل Bolton ---
            st.divider()
            st.markdown("### 🔢 نسبت‌های بولتون (Bolton Analysis)")
            ant_ratio = round((l_ant / u_ant) * 100, 2) if u_ant > 0 else 0.0
            overall_ratio = round((l_tot / u_tot) * 100, 2) if u_tot > 0 else 0.0

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                diff_overall = round(overall_ratio - 91.3, 2)
                st.metric("Overall Bolton Ratio (Norm: 91.3%)", f"{overall_ratio}%", f"{diff_overall}%")
                if overall_ratio > 92.5:
                    st.warning("⚠️ اضافه حجم دندانی در فک پایین")
                elif overall_ratio < 90.0:
                    st.info("ℹ️ اضافه حجم دندانی در فک بالا")
                else:
                    st.success("✅ نسبت کلی متوازن است.")
            with res_col2:
                diff_ant = round(ant_ratio - 77.2, 2)
                st.metric("Anterior Bolton Ratio (Norm: 77.2%)", f"{ant_ratio}%", f"{diff_ant}%")
                if ant_ratio > 78.5:
                    st.warning("⚠️ اضافه حجم دندان‌های قدامی فک پایین")
                elif ant_ratio < 75.5:
                    st.info("ℹ️ اضافه حجم دندان‌های قدامی فک بالا")
                else:
                    st.success("✅ نسبت قدامی متوازن است.")
    else:
        st.info("💡 برای شروع، لطفاً حداقل یک فایل STL/OBJ آپلود کنید.")
