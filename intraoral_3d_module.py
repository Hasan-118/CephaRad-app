import streamlit as st
import trimesh
import numpy as np
import io
import pyvista as pv
from stpyvista import stpyvista

# ============================================================
# توابع کمکی
# ============================================================

def parse_mesh_pyvista(uploaded_file):
    """بارگذاری فایل mesh و تبدیل به فرمت PyVista"""
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

        # تبدیل trimesh به pyvista
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)

        # PyVista نیاز به فرمت خاص faces دارد
        # [3, i0, i1, i2, 3, i3, i4, i5, ...]
        faces_pv = np.hstack([
            np.full((len(faces), 1), 3),
            faces
        ]).flatten()

        pv_mesh = pv.PolyData(vertices, faces_pv)
        return pv_mesh
    except Exception as e:
        st.error(f"خطا در بارگذاری: {type(e).__name__}: {e}")
        return None


def render_pyvista_mesh(pv_mesh, title="3D Scan"):
    """رندر مش با PyVista و نمایش در Streamlit"""
    if pv_mesh is None:
        return

    # محاسبه نرمال‌ها برای نورپردازی صاف
    pv_mesh.compute_normals(
        cell_normals=False,
        point_normals=True,
        inplace=True,
        auto_orient_normals=True
    )

    # ایجاد پلاتر
    plotter = pv.Plotter(
        window_size=[800, 600],
        off_screen=True,
        border=False
    )

    # رنگ کرم روشن (شبیه گچ دندانی)
    plotter.add_mesh(
        pv_mesh,
        color='#F0E6D2',  # کرم روشن
        smooth_shading=True,
        specular=0.3,
        diffuse=0.8,
        ambient=0.3,
        show_edges=False,
        lighting=True
    )

    # تنظیم پس‌زمینه سفید
    plotter.background_color = 'white'

    # تنظیم دوربین
    plotter.camera_position = 'xy'  # نمای بالا
    plotter.camera.elevation = 60   # کمی از بالا

    # حذف محورها
    plotter.remove_all_lights()
    plotter.add_light(pv.Light(
        position=(1, 1, 1),
        light_type='scene light',
        intensity=0.8
    ))

    # نمایش در Streamlit
    stpyvista(plotter, key=f"pv_{title}")

    plotter.close()


# ============================================================
# رابط کاربری اصلی
# ============================================================

def render_intraoral_3d_tab():
    st.header("🦷 آنالیز سه بعدی اسکن داخل دهانی")

    st.info("""
    **راهنما:** ابتدا اسکن فک بالا و پایین را آپلود کنید.
    نمای سه‌بعدی با کیفیت بالا نمایش داده می‌شود.
    """)

    col_up1, col_up2 = st.columns(2)
    with col_up1:
        stl_maxilla = st.file_uploader("آپلود اسکن فک بالا (Maxilla STL/OBJ):",
                                        type=['stl', 'obj'], key="max_stl")
    with col_up2:
        stl_mandible = st.file_uploader("آپلود اسکن فک پایین (Mandible STL/OBJ):",
                                         type=['stl', 'obj'], key="man_stl")

    mesh_max = parse_mesh_pyvista(stl_maxilla) if stl_maxilla else None
    mesh_man = parse_mesh_pyvista(stl_mandible) if stl_mandible else None

    if mesh_max is not None:
        st.success(f"✅ فک بالا بارگذاری شد: {mesh_max.n_points} رأس")
    if mesh_man is not None:
        st.success(f"✅ فک پایین بارگذاری شد: {mesh_man.n_points} رأس")

    # ============ نمای سه‌بعدی ============
    if mesh_max is not None or mesh_man is not None:
        st.divider()
        st.subheader("🖼 نمای سه‌بعدی (قابل چرخش با ماوس)")

        view_col1, view_col2 = st.columns(2)
        with view_col1:
            if mesh_max is not None:
                st.markdown("**فک بالا (Maxilla)**")
                try:
                    render_pyvista_mesh(mesh_max, "maxilla")
                except Exception as e:
                    st.error(f"خطا در رندر فک بالا: {e}")

        with view_col2:
            if mesh_man is not None:
                st.markdown("**فک پایین (Mandible)**")
                try:
                    render_pyvista_mesh(mesh_man, "mandible")
                except Exception as e:
                    st.error(f"خطا در رندر فک پایین: {e}")

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
