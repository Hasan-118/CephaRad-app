import streamlit as st
import trimesh
import numpy as np
import io
import plotly.graph_objects as go

# ============================================================
# توابع کمکی
# ============================================================

def parse_mesh(uploaded_file):
    """بارگذاری فایل mesh"""
    if uploaded_file is None:
        return None
    try:
        file_bytes = uploaded_file.read()
        if len(file_bytes) == 0:
            return None
        file_type = uploaded_file.name.split('.')[-1].lower()
        mesh = trimesh.load(io.BytesIO(file_bytes), file_type=file_type, force='mesh')
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
            return None

        extents = mesh.extents
        max_dim = np.max(extents)
        if max_dim < 10.0:
            mesh.apply_scale(10.0)
        elif max_dim > 300.0:
            mesh.apply_scale(0.1)

        return mesh
    except Exception as e:
        st.error(f"❌ خطا در بارگذاری `{uploaded_file.name}`: {type(e).__name__}: {e}")
        return None


def safe_simplify_mesh(mesh, target_faces=20000):
    """کاهش تراکم مش برای رندر سریع‌تر"""
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
        np.random.seed(42)
        keep_indices = np.sort(np.random.choice(current_faces, target_faces, replace=False))
        new_faces = mesh.faces[keep_indices]
        used_vertices = np.unique(new_faces)
        new_vertices = mesh.vertices[used_vertices]
        index_map = np.zeros(len(mesh.vertices), dtype=np.int64)
        index_map[used_vertices] = np.arange(len(used_vertices))
        new_faces = index_map[new_faces]
        return trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    except Exception:
        return mesh


def create_3d_plotly_figure(mesh, title="3D Scan"):
    """رندر مش با Plotly"""
    if mesh is None:
        return None
    vertices = mesh.vertices
    faces = mesh.faces

    fig = go.Figure(data=[
        go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color='#F0E6D2',
            opacity=1.0,
            flatshading=False,
            lighting=dict(ambient=0.7, diffuse=0.9, specular=0.3,
                          roughness=0.4, fresnel=0.1),
            lightposition=dict(x=100, y=200, z=150)
        )
    ])
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center', font=dict(size=14)),
        scene=dict(
            xaxis=dict(visible=False, showbackground=False),
            yaxis=dict(visible=False, showbackground=False),
            zaxis=dict(visible=False, showbackground=False),
            aspectmode='data',
            bgcolor='white',
            camera=dict(eye=dict(x=0, y=0, z=2.5), up=dict(x=0, y=1, z=0))
        ),
        margin=dict(r=5, l=5, b=5, t=40),
        paper_bgcolor='white',
        height=500
    )
    return fig


# ============================================================
# رابط کاربری اصلی
# ============================================================

def render_intraoral_3d_tab():
    st.header("🦷 آنالیز سه بعدی اسکن داخل دهانی")

    st.info("""
    **راهنما:** ابتدا اسکن فک بالا و پایین را آپلود کنید.
    سپس نمای سه‌بعدی نمایش داده می‌شود و می‌توانید با ماوس بچرخانید.
    """)

    col_up1, col_up2 = st.columns(2)
    with col_up1:
        stl_maxilla = st.file_uploader("آپلود اسکن فک بالا (Maxilla STL/OBJ):",
                                        type=['stl', 'obj'], key="max_stl")
    with col_up2:
        stl_mandible = st.file_uploader("آپلود اسکن فک پایین (Mandible STL/OBJ):",
                                         type=['stl', 'obj'], key="man_stl")

    mesh_max = parse_mesh(stl_maxilla) if stl_maxilla else None
    mesh_man = parse_mesh(stl_mandible) if stl_mandible else None

    if mesh_max is not None:
        st.success(f"✅ فک بالا بارگذاری شد: {len(mesh_max.vertices)} رأس")
    if mesh_man is not None:
        st.success(f"✅ فک پایین بارگذاری شد: {len(mesh_man.vertices)} رأس")

    if mesh_max is not None or mesh_man is not None:
        st.divider()
        st.subheader("🖼 نمای سه‌بعدی (قابل چرخش)")

        view_col1, view_col2 = st.columns(2)

        with view_col1:
            if mesh_max is not None:
                st.markdown("**فک بالا (Maxilla)**")
                try:
                    mesh_simple = safe_simplify_mesh(mesh_max, target_faces=20000)
                    fig_max = create_3d_plotly_figure(mesh_simple, "Maxillary Arch")
                    st.plotly_chart(fig_max, use_container_width=True, key="plotly_max")
                except Exception as e:
                    st.error(f"خطا در نمایش فک بالا: {e}")

        with view_col2:
            if mesh_man is not None:
                st.markdown("**فک پایین (Mandible)**")
                try:
                    mesh_simple = safe_simplify_mesh(mesh_man, target_faces=20000)
                    fig_man = create_3d_plotly_figure(mesh_simple, "Mandibular Arch")
                    st.plotly_chart(fig_man, use_container_width=True, key="plotly_man")
                except Exception as e:
                    st.error(f"خطا در نمایش فک پایین: {e}")

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
