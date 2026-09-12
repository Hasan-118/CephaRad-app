import streamlit as st
import trimesh
import numpy as np
import plotly.graph_objects as go
import io
from scipy.spatial import cKDTree

# ============================================================
# توابع کمکی
# ============================================================

def parse_mesh(uploaded_file):
    """بارگذاری فایل mesh با مدیریت خطای کامل"""
    if uploaded_file is None:
        return None
    try:
        file_bytes = uploaded_file.read()
        if len(file_bytes) == 0:
            st.error(f"❌ فایل `{uploaded_file.name}` خالی است.")
            return None

        file_type = uploaded_file.name.split('.')[-1].lower()
        mesh = trimesh.load(io.BytesIO(file_bytes), file_type=file_type, force='mesh')

        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
            st.error(f"❌ مش `{uploaded_file.name}` رأس ندارد.")
            return None

        # نرمال‌سازی مقیاس
        extents = mesh.extents
        max_dim = np.max(extents)
        if max_dim < 10.0:
            mesh.apply_scale(10.0)
        elif max_dim > 300.0:
            mesh.apply_scale(0.1)

        return mesh
    except Exception as e:
        st.error(f"❌ خطا در `{uploaded_file.name}`: {type(e).__name__}: {e}")
        return None


def safe_simplify_mesh(mesh, target_faces=10000):
    """
    کاهش تراکم مش با روش‌های مطمئن (چند مرحله‌ای)
    """
    if mesh is None:
        return None

    current_faces = len(mesh.faces)

    if current_faces <= target_faces:
        return mesh

    try:
        return mesh.simplify_quadratic_decimation(target_faces)
    except Exception:
        pass

    try:
        ratio = target_faces / current_faces
        if hasattr(mesh, 'simplify_quadric_decimation'):
            return mesh.simplify_quadric_decimation(target_faces)
    except Exception:
        pass

    try:
        if current_faces > target_faces:
            np.random.seed(42)
            keep_indices = np.random.choice(current_faces, target_faces, replace=False)
            keep_indices = np.sort(keep_indices)

            new_faces = mesh.faces[keep_indices]
            used_vertices = np.unique(new_faces)
            new_vertices = mesh.vertices[used_vertices]

            index_map = np.zeros(len(mesh.vertices), dtype=np.int64)
            index_map[used_vertices] = np.arange(len(used_vertices))
            new_faces = index_map[new_faces]

            return trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    except Exception:
        pass

    return mesh


def create_3d_plotly_figure(mesh, title="3D Intraoral Scan", color='lightpink'):
    """رندر تعاملی سه بعدی"""
    if mesh is None:
        return None
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
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(visible=False), aspectmode='data'),
        margin=dict(r=10, l=10, b=10, t=40)
    )
    return fig


# ============================================================
# سگمنتیشن
# ============================================================

def compute_vertex_curvature(mesh, k_neighbors=20):
    """محاسبه انحنای تقریبی هر رأس با PCA"""
    vertices = mesh.vertices
    tree = cKDTree(vertices)
    _, indices = tree.query(vertices, k=k_neighbors)

    curvatures = np.zeros(len(vertices))
    for i in range(len(vertices)):
        neighbors = vertices[indices[i]]
        center = neighbors.mean(axis=0)
        centered = neighbors - center
        cov = np.dot(centered.T, centered) / len(neighbors)
        eigenvalues = np.linalg.eigvalsh(cov)
        total = np.sum(eigenvalues) + 1e-9
        curvatures[i] = eigenvalues[0] / total
    return curvatures


def segment_teeth_from_gingiva(mesh, curvature_threshold=0.02):
    """جداسازی دندان از لثه بر اساس انحنا"""
    curvatures = compute_vertex_curvature(mesh)
    curv_norm = (curvatures - curvatures.min()) / (curvatures.max() - curvatures.min() + 1e-9)
    teeth_mask = curv_norm > curvature_threshold
    return teeth_mask, curvatures


def render_segmented_mesh(mesh_simple, teeth_mask_simple, title="Segmented"):
    """نمایش مش دو رنگ (مش باید از قبل ساده‌شده باشد)"""
    vertices = mesh_simple.vertices
    faces = mesh_simple.faces

    if len(teeth_mask_simple) != len(vertices):
        teeth_mask_simple = np.ones(len(vertices), dtype=bool)

    face_teeth_count = teeth_mask_simple[faces].sum(axis=1)
    face_is_teeth = face_teeth_count >= 2

    teeth_faces = faces[face_is_teeth]
    gingiva_faces = faces[~face_is_teeth]

    fig = go.Figure()

    if len(gingiva_faces) > 0:
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=gingiva_faces[:, 0], j=gingiva_faces[:, 1], k=gingiva_faces[:, 2],
            color='lightpink', opacity=0.7, name='لثه',
            lighting=dict(ambient=0.5, diffuse=0.8)
        ))

    if len(teeth_faces) > 0:
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=teeth_faces[:, 0], j=teeth_faces[:, 1], k=teeth_faces[:, 2],
            color='white', opacity=0.95, name='دندان',
            lighting=dict(ambient=0.6, diffuse=0.9)
        ))

    fig.update_layout(
        title=title,
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(visible=False), aspectmode='data'),
        margin=dict(r=10, l=10, b=10, t=40),
        showlegend=True
    )
    return fig


# ============================================================
# رابط کاربری اصلی
# ============================================================

def render_intraoral_3d_tab():
    st.header("🦷 آنالیز سه بعدی اسکن داخل دهانی (نیمه‌خودکار)")

    st.info("""
    **راهنما:** ابتدا اسکن را آپلود کنید. برنامه سگمنتیشن اولیه را انجام می‌دهد.
    با تغییر آستانه انحنا در سایدبار می‌توانید نتیجه را دقیق‌تر کنید.
    """)

    col_up1, col_up2 = st.columns(2)
    with col_up1:
        stl_maxilla = st.file_uploader("آپلود اسکن فک بالا (Maxilla STL/OBJ):",
                                        type=['stl', 'obj'], key="max_stl")
    with col_up2:
        stl_mandible = st.file_uploader("آپلود اسکن فک پایین (Mandible STL/OBJ):",
                                         type=['stl', 'obj'], key="man_stl")

    # === بارگذاری مستقیم (بدون کش) ===
    mesh_max_orig = None
    mesh_man_orig = None

    if stl_maxilla is not None:
        try:
            file_bytes = stl_maxilla.read()
            if len(file_bytes) > 0:
                mesh_max_orig = trimesh.load(
                    io.BytesIO(file_bytes),
                    file_type='stl',
                    force='mesh'
                )
                if isinstance(mesh_max_orig, trimesh.Scene):
                    mesh_max_orig = mesh_max_orig.dump(concatenate=True)

                max_dim = np.max(mesh_max_orig.extents)
                if max_dim < 10.0:
                    mesh_max_orig.apply_scale(10.0)
                elif max_dim > 300.0:
                    mesh_max_orig.apply_scale(0.1)

                st.success(f"✅ فک بالا بارگذاری شد: {len(mesh_max_orig.vertices)} رأس")
        except Exception as e:
            st.error(f"❌ خطا در فک بالا: {type(e).__name__}: {e}")
            mesh_max_orig = None

    if stl_mandible is not None:
        try:
            file_bytes = stl_mandible.read()
            if len(file_bytes) > 0:
                mesh_man_orig = trimesh.load(
                    io.BytesIO(file_bytes),
                    file_type='stl',
                    force='mesh'
                )
                if isinstance(mesh_man_orig, trimesh.Scene):
                    mesh_man_orig = mesh_man_orig.dump(concatenate=True)

                max_dim = np.max(mesh_man_orig.extents)
                if max_dim < 10.0:
                    mesh_man_orig.apply_scale(10.0)
                elif max_dim > 300.0:
                    mesh_man_orig.apply_scale(0.1)

                st.success(f"✅ فک پایین بارگذاری شد: {len(mesh_man_orig.vertices)} رأس")
        except Exception as e:
            st.error(f"❌ خطا در فک پایین: {type(e).__name__}: {e}")
            mesh_man_orig = None

    # --- تنظیمات سایدبار ---
    st.sidebar.markdown("### ⚙️ تنظیمات سگمنتیشن")
    curvature_thresh = st.sidebar.slider(
        "آستانه انحنا:", 0.005, 0.10, 0.02, 0.001,
        help="بالاتر = دندان کمتر تشخیص داده می‌شود"
    )

    # --- اگر مش‌ها بارگذاری شدند ---
    if mesh_max_orig is not None or mesh_man_orig is not None:

        # ============ سگمنتیشن ============
        st.divider()
        st.subheader("🔬 سگمنتیشن اولیه (جدا کردن دندان از لثه)")

        seg_col1, seg_col2 = st.columns(2)

        # --- فک بالا ---
        with seg_col1:
            if mesh_max_orig is not None:
                st.markdown("**فک بالا (Maxilla)**")
                try:
                    with st.spinner("در حال محاسبه انحنا..."):
                        teeth_mask_max, _ = segment_teeth_from_gingiva(
                            mesh_max_orig, curvature_threshold=curvature_thresh
                        )
                        teeth_percent_max = (teeth_mask_max.sum() / len(teeth_mask_max)) * 100
                        st.caption(f"🦷 دندان‌های تشخیص داده‌شده: {teeth_percent_max:.1f}%")

                    mesh_max_simple = safe_simplify_mesh(mesh_max_orig, target_faces=8000)

                    if len(mesh_max_simple.vertices) == len(mesh_max_orig.vertices):
                        mask_max_simple = teeth_mask_max
                    else:
                        tree = cKDTree(mesh_max_orig.vertices)
                        _, idx = tree.query(mesh_max_simple.vertices, k=1)
                        mask_max_simple = teeth_mask_max[idx]

                    fig_seg_max = render_segmented_mesh(
                        mesh_max_simple, mask_max_simple, "Maxilla - Segmented"
                    )
                    st.plotly_chart(fig_seg_max, use_container_width=True, key="fig_seg_max_v2")
                except Exception as e:
                    st.error(f"خطا در سگمنتیشن فک بالا: {type(e).__name__}: {e}")
            else:
                st.info("فک بالا آپلود نشده")

        # --- فک پایین ---
        with seg_col2:
            if mesh_man_orig is not None:
                st.markdown("**فک پایین (Mandible)**")
                try:
                    with st.spinner("در حال محاسبه انحنا..."):
                        teeth_mask_man, _ = segment_teeth_from_gingiva(
                            mesh_man_orig, curvature_threshold=curvature_thresh
                        )
                        teeth_percent_man = (teeth_mask_man.sum() / len(teeth_mask_man)) * 100
                        st.caption(f"🦷 دندان‌های تشخیص داده‌شده: {teeth_percent_man:.1f}%")

                    mesh_man_simple = safe_simplify_mesh(mesh_man_orig, target_faces=8000)

                    if len(mesh_man_simple.vertices) == len(mesh_man_orig.vertices):
                        mask_man_simple = teeth_mask_man
                    else:
                        tree = cKDTree(mesh_man_orig.vertices)
                        _, idx = tree.query(mesh_man_simple.vertices, k=1)
                        mask_man_simple = teeth_mask_man[idx]

                    fig_seg_man = render_segmented_mesh(
                        mesh_man_simple, mask_man_simple, "Mandible - Segmented"
                    )
                    st.plotly_chart(fig_seg_man, use_container_width=True, key="fig_seg_man_v2")
                except Exception as e:
                    st.error(f"خطا در سگمنتیشن فک پایین: {type(e).__name__}: {e}")
            else:
                st.info("فک پایین آپلود نشده")

        # ============ اندازه‌گیری دستی ============
        st.divider()
        st.subheader("📏 ورود دستی عرض دندان‌ها")

        if 'u_ant_val' not in st.session_state: st.session_state.u_ant_val = 45.0
        if 'u_tot_val' not in st.session_state: st.session_state.u_tot_val = 90.0
        if 'l_ant_val' not in st.session_state: st.session_state.l_ant_val = 35.0
        if 'l_tot_val' not in st.session_state: st.session_state.l_tot_val = 82.0

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

        # ============ Bolton ============
        st.divider()
        st.markdown("### 🔢 نسبت‌های بولتون")
        ant_ratio = round((l_ant / u_ant) * 100, 2) if u_ant > 0 else 0.0
        overall_ratio = round((l_tot / u_tot) * 100, 2) if u_tot > 0 else 0.0

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            diff_overall = round(overall_ratio - 91.3, 2)
            st.metric("Overall Bolton (Norm: 91.3%)", f"{overall_ratio}%", f"{diff_overall}%")
        with res_col2:
            diff_ant = round(ant_ratio - 77.2, 2)
            st.metric("Anterior Bolton (Norm: 77.2%)", f"{ant_ratio}%", f"{diff_ant}%")
    else:
        st.info("💡 برای شروع، لطفاً حداقل یک فایل STL/OBJ آپلود کنید.")
