import streamlit as st
import trimesh
import numpy as np
import io
import plotly.graph_objects as go
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates


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
        height=450
    )
    return fig


def render_occlusal_view(mesh, img_size=800, highlight_z_percentile=30):
    """
    رندر نمای اکلوزال (از بالا) به صورت تصویر دوبعدی
    برای کلیک کاربر
    """
    if mesh is None:
        return None, None

    vertices = mesh.vertices

    x_min, y_min, z_min = vertices.min(axis=0)
    x_max, y_max, z_max = vertices.max(axis=0)

    pad_ratio = 0.08
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * pad_ratio
    x_max += x_range * pad_ratio
    y_min -= y_range * pad_ratio
    y_max += y_range * pad_ratio

    x_scale = (img_size - 60) / (x_max - x_min)
    y_scale = (img_size - 60) / (y_max - y_min)
    scale = min(x_scale, y_scale)

    img = Image.new('RGB', (img_size, img_size), 'white')
    draw = ImageDraw.Draw(img)

    z_threshold = z_min + (z_max - z_min) * (1 - highlight_z_percentile / 100.0)

    def to_pixel(v):
        px = int((v[0] - x_min) * scale + 30)
        py = int(img_size - (v[1] - y_min) * scale - 30)
        return px, py

    pts = []
    for v in vertices:
        if v[2] > z_threshold:
            px, py = to_pixel(v)
            if 0 <= px < img_size and 0 <= py < img_size:
                pts.append((px, py, v[2]))

    for px, py, z in pts:
        intensity = int(220 - 120 * (z - z_threshold) / (z_max - z_threshold + 1e-9))
        intensity = max(60, min(220, intensity))
        draw.ellipse([px-1, py-1, px+1, py+1], fill=(intensity, intensity, intensity))

    return img, (x_min, x_max, y_min, y_max, scale, img_size, z_threshold)


def pixel_to_mm(px, py, transform, mesh):
    """تبدیل مختصات پیکسل به میلی‌متر روی مش"""
    if transform is None or mesh is None:
        return None
    x_min, x_max, y_min, y_max, scale, img_size, _ = transform

    vx = (px - 30) / scale + x_min
    vy = (img_size - py - 30) / scale + y_min

    mesh_center = mesh.vertices.mean(axis=0)
    z_est = mesh.vertices[:, 2].max() - 2.0

    closest_idx = np.argmin(np.linalg.norm(
        mesh.vertices - np.array([vx, vy, z_est]), axis=1
    ))
    return mesh.vertices[closest_idx]


def compute_measurements_from_clicks(clicks, transform, mesh, pixel_size_mm=0.05):
    """
    محاسبه عرض مزیودیستال از کلیک‌های کاربر
    clicks: لیست نقاط [px, py]
    transform: (x_min, x_max, y_min, y_max, scale, img_size, z_threshold)
    """
    if transform is None or mesh is None or len(clicks) < 2:
        return None

    x_min, x_max, y_min, y_max, scale, img_size, _ = transform

    # تبدیل پیکسل به مختصات سه‌بعدی مش
    mm_points = []
    for px, py in clicks:
        vx = (px - 30) / scale + x_min
        vy = (img_size - py - 30) / scale + y_min
        mesh_center = mesh.vertices.mean(axis=0)
        z_est = mesh.vertices[:, 2].max() - 2.0
        closest_idx = np.argmin(np.linalg.norm(
            mesh.vertices - np.array([vx, vy, z_est]), axis=1
        ))
        mm_points.append(mesh.vertices[closest_idx])

    return np.array(mm_points)


# ============================================================
# رابط کاربری اصلی
# ============================================================

def render_intraoral_3d_tab():
    st.header("🦷 آنالیز سه بعدی اسکن داخل دهانی")

    st.info("""
    **راهنما:** ابتدا اسکن فک بالا و پایین را آپلود کنید.
    سپس در بخش **اندازه‌گیری دستی**، روی نمای اکلوزال کلیک کنید 
    تا نقاط مزیال و دیستال هر دندان را مشخص کنید.
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

    if mesh_max is None and mesh_man is None:
        st.info("💡 برای شروع، لطفاً حداقل یک فایل STL/OBJ آپلود کنید.")
        return

    # ============ نمای سه‌بعدی ============
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

    # ============ اندازه‌گیری دستی با کلیک ============
    st.divider()
    st.subheader("📐 اندازه‌گیری دستی از روی نمای اکلوزال")

    st.caption("""
    روی **نمای اکلوزال** (از بالا) زیر کلیک کنید تا نقاط مزیال و دیستال دندان‌ها 
    را مشخص کنید. هر کلیک به عنوان یک نقطه مرزی ثبت می‌شود.
    """)

    measure_col1, measure_col2 = st.columns(2)

    with measure_col1:
        if mesh_max is not None:
            st.markdown("**فک بالا - نمای اکلوزال**")
            img_max, transform_max = render_occlusal_view(mesh_max, img_size=800)

            if img_max is not None:
                # ذخیره نقاط کلیک‌شده
                if 'clicks_max' not in st.session_state:
                    st.session_state.clicks_max = []

                # نمایش تصویر با کلیک
                res = streamlit_image_coordinates(
                    img_max, key="occlusal_max_click"
                )

                if res:
                    new_click = [res["x"], res["y"]]
                    # بررسی تکراری نبودن
                    is_duplicate = False
                    for existing in st.session_state.clicks_max:
                        if abs(existing[0] - new_click[0]) < 5 and \
                           abs(existing[1] - new_click[1]) < 5:
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        st.session_state.clicks_max.append(new_click)
                        st.rerun()

                # نمایش تصویر با نقاط علامت‌گذاری‌شده
                img_display = img_max.copy()
                draw_display = ImageDraw.Draw(img_display)
                for i, (px, py) in enumerate(st.session_state.clicks_max):
                    draw_display.ellipse([px-5, py-5, px+5, py+5],
                                         fill="red", outline="white", width=2)
                    draw_display.text((px+8, py-8), str(i+1), fill="red")

                st.image(img_display, caption=f"کلیک‌ها: {len(st.session_state.clicks_max)}",
                         use_container_width=True)

                st.caption(f"🖱 نقاط ثبت‌شده: {len(st.session_state.clicks_max)}")

                if st.button("🗑 پاک کردن کلیک‌های فک بالا", key="clear_max"):
                    st.session_state.clicks_max = []
                    st.rerun()

                # محاسبه فاصله بین نقاط
                if len(st.session_state.clicks_max) >= 2:
                    if 'u_ant_val' not in st.session_state:
                        st.session_state.u_ant_val = 45.0
                    if 'u_tot_val' not in st.session_state:
                        st.session_state.u_tot_val = 90.0

    with measure_col2:
        if mesh_man is not None:
            st.markdown("**فک پایین - نمای اکلوزال**")
            img_man, transform_man = render_occlusal_view(mesh_man, img_size=800)

            if img_man is not None:
                if 'clicks_man' not in st.session_state:
                    st.session_state.clicks_man = []

                res = streamlit_image_coordinates(
                    img_man, key="occlusal_man_click"
                )

                if res:
                    new_click = [res["x"], res["y"]]
                    is_duplicate = False
                    for existing in st.session_state.clicks_man:
                        if abs(existing[0] - new_click[0]) < 5 and \
                           abs(existing[1] - new_click[1]) < 5:
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        st.session_state.clicks_man.append(new_click)
                        st.rerun()

                img_display = img_man.copy()
                draw_display = ImageDraw.Draw(img_display)
                for i, (px, py) in enumerate(st.session_state.clicks_man):
                    draw_display.ellipse([px-5, py-5, px+5, py+5],
                                         fill="blue", outline="white", width=2)
                    draw_display.text((px+8, py-8), str(i+1), fill="blue")

                st.image(img_display, caption=f"کلیک‌ها: {len(st.session_state.clicks_man)}",
                         use_container_width=True)

                st.caption(f"🖱 نقاط ثبت‌شده: {len(st.session_state.clicks_man)}")

                if st.button("🗑 پاک کردن کلیک‌های فک پایین", key="clear_man"):
                    st.session_state.clicks_man = []
                    st.rerun()

                if 'l_ant_val' not in st.session_state:
                    st.session_state.l_ant_val = 35.0
                if 'l_tot_val' not in st.session_state:
                    st.session_state.l_tot_val = 82.0

    # ============ ورود مقادیر نهایی و Bolton ============
    st.divider()
    st.subheader("📏 ورود نهایی عرض دندان‌ها (از کلیک یا دستی)")

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        st.markdown("**فک بالا (Maxilla)**")
        u_ant = st.number_input("عرض ۶ دندان قدامی بالا (mm):", 20.0, 70.0,
                                value=st.session_state.u_ant_val, step=0.1,
                                key='num_u_ant')
        u_tot = st.number_input("عرض ۱۲ دندان بالا (mm):", 50.0, 130.0,
                                value=st.session_state.u_tot_val, step=0.1,
                                key='num_u_tot')
    with b_col2:
        st.markdown("**فک پایین (Mandible)**")
        l_ant = st.number_input("عرض ۶ دندان قدامی پایین (mm):", 15.0, 60.0,
                                value=st.session_state.l_ant_val, step=0.1,
                                key='num_l_ant')
        l_tot = st.number_input("عرض ۱۲ دندان پایین (mm):", 40.0, 120.0,
                                value=st.session_state.l_tot_val, step=0.1,
                                key='num_l_tot')

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
        if overall_ratio > 92.5:
            st.warning("⚠️ اضافه حجم دندانی در فک پایین")
        elif overall_ratio < 90.0:
            st.info("ℹ️ اضافه حجم دندانی در فک بالا")
        else:
            st.success("✅ نسبت کلی متوازن است.")

    with res_col2:
        diff_ant = round(ant_ratio - 77.2, 2)
        st.metric("Anterior Bolton (Norm: 77.2%)", f"{ant_ratio}%", f"{diff_ant}%")
        if ant_ratio > 78.5:
            st.warning("⚠️ اضافه حجم دندانی در فک پایین (قدامی)")
        elif ant_ratio < 75.5:
            st.info("ℹ️ اضافه حجم دندانی در فک بالا (قدامی)")
        else:
            st.success("✅ نسبت قدامی متوازن است.")

    # ============ ذخیره نتایج ۳D در session_state برای PDF ============
    st.session_state.bolton_3d = {
        "u_ant": u_ant,
        "u_tot": u_tot,
        "l_ant": l_ant,
        "l_tot": l_tot,
        "ant_ratio": ant_ratio,
        "overall_ratio": overall_ratio,
        "clicks_max": st.session_state.get('clicks_max', []),
        "clicks_man": st.session_state.get('clicks_man', [])
    }
