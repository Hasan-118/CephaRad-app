import streamlit as st
import trimesh
import numpy as np
import plotly.graph_objects as go
import io
from PIL import Image, ImageDraw

# ============================================================
# توابع کمکی
# ============================================================

def parse_mesh_simple(uploaded_file):
    """بارگذاری ساده فایل mesh"""
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
        return mesh
    except Exception as e:
        st.error(f"خطا در بارگذاری: {e}")
        return None


def create_clear_3d_figure(mesh, title="3D Scan"):
    """رندر واضح سه‌بعدی با یک رنگ ساده"""
    if mesh is None:
        return None
    vertices = mesh.vertices
    faces = mesh.faces

    fig = go.Figure(data=[
        go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color='#E8D4B8',  # رنگ کرم روشن (شبیه گچ دندانی)
            opacity=1.0,
            flatshading=False,
            lighting=dict(
                ambient=0.7,
                diffuse=0.9,
                specular=0.3,
                roughness=0.4,
                fresnel=0.1
            ),
            lightposition=dict(x=100, y=200, z=150)
        )
    ])
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center', font=dict(size=16)),
        scene=dict(
            xaxis=dict(visible=False, showbackground=False),
            yaxis=dict(visible=False, showbackground=False),
            zaxis=dict(visible=False, showbackground=False),
            aspectmode='data',
            bgcolor='white',
            camera=dict(
                eye=dict(x=0, y=0, z=2.5),  # نمای بالا
                up=dict(x=0, y=1, z=0)
            )
        ),
        margin=dict(r=5, l=5, b=5, t=40),
        paper_bgcolor='white'
    )
    return fig


def render_occlusal_view(mesh, img_size=800):
    """
    رندر نمای اکلوزال (از بالا) به صورت تصویر دوبعدی
    برای کلیک کاربر روی نقاط مرزی دندان‌ها
    """
    if mesh is None:
        return None, None, None

    vertices = mesh.vertices

    # محاسبه bounding box
    x_min, y_min, z_min = vertices.min(axis=0)
    x_max, y_max, z_max = vertices.max(axis=0)

    # نرمال‌سازی مختصات به تصویر
    x_range = x_max - x_min
    y_range = y_max - y_min

    # اضافه کردن padding
    pad_ratio = 0.1
    pad_x = x_range * pad_ratio
    pad_y = y_range * pad_ratio

    x_min -= pad_x
    x_max += pad_x
    y_min -= pad_y
    y_max += pad_y

    # محاسبه مقیاس
    x_scale = (img_size - 40) / (x_max - x_min)
    y_scale = (img_size - 40) / (y_max - y_min)
    scale = min(x_scale, y_scale)

    # تصویر سفید
    img = Image.new('RGB', (img_size, img_size), 'white')
    draw = ImageDraw.Draw(img)

    # رسم نقاط مش (نمای اکلوزال)
    # از بالا نگاه می‌کنیم: x افقی، y عمودی، z ارتفاع
    # فقط نقاطی که z بالاتر است (سطح اکلوزال) رسم می‌شوند

    # ضخامت بر اساس z
    z_threshold = z_min + (z_max - z_min) * 0.3  # 30٪ بالایی

    for v in vertices:
        if v[2] > z_threshold:  # فقط نقاط سطح بالا
            px = int((v[0] - x_min) * scale + 20)
            py = int(img_size - (v[1] - y_min) * scale - 20)

            if 0 <= px < img_size and 0 <= py < img_size:
                # شدت رنگ بر اساس ارتفاع
                intensity = int(200 - 100 * (v[2] - z_threshold) / (z_max - z_threshold + 1e-9))
                intensity = max(80, min(200, intensity))
                draw.point((px, py), fill=(intensity, intensity, intensity))

    # تبدیل به bytes
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    img_bytes = buf.getvalue()

    return img, img_bytes, (x_min, x_max, y_min, y_max, scale, img_size)


# ============================================================
# رابط کاربری اصلی
# ============================================================

def render_intraoral_3d_tab():
    st.header("🦷 آنالیز سه بعدی اسکن داخل دهانی")

    st.info("""
    **راهنما:** ابتدا اسکن فک بالا و پایین را آپلود کنید.
    سپس نمای سه‌بعدی و نمای اکلوزال (از بالا) نمایش داده می‌شود.
    برای اندازه‌گیری، روی نمای اکلوزال کلیک کنید و نقاط مرزی دندان‌ها را مشخص کنید.
    """)

    col_up1, col_up2 = st.columns(2)
    with col_up1:
        stl_maxilla = st.file_uploader("آپلود اسکن فک بالا (Maxilla STL/OBJ):",
                                        type=['stl', 'obj'], key="max_stl")
    with col_up2:
        stl_mandible = st.file_uploader("آپلود اسکن فک پایین (Mandible STL/OBJ):",
                                         type=['stl', 'obj'], key="man_stl")

    mesh_max = parse_mesh_simple(stl_maxilla) if stl_maxilla else None
    mesh_man = parse_mesh_simple(stl_mandible) if stl_mandible else None

    if mesh_max is not None:
        st.success(f"✅ فک بالا بارگذاری شد: {len(mesh_max.vertices)} رأس")
    if mesh_man is not None:
        st.success(f"✅ فک پایین بارگذاری شد: {len(mesh_man.vertices)} رأس")

    # ============ نمای سه‌بعدی واضح ============
    if mesh_max is not None or mesh_man is not None:
        st.divider()
        st.subheader("🖼 نمای سه‌بعدی (قابل چرخش با ماوس)")

        view_col1, view_col2 = st.columns(2)
        with view_col1:
            if mesh_max is not None:
                st.markdown("**فک بالا (Maxilla)**")
                fig_max = create_clear_3d_figure(mesh_max, "Maxillary Arch")
                st.plotly_chart(fig_max, use_container_width=True, key="3d_max")

        with view_col2:
            if mesh_man is not None:
                st.markdown("**فک پایین (Mandible)**")
                fig_man = create_clear_3d_figure(mesh_man, "Mandibular Arch")
                st.plotly_chart(fig_man, use_container_width=True, key="3d_man")

        # ============ نمای اکلوزال (از بالا) ============
        st.divider()
        st.subheader("📐 نمای اکلوزال (از بالا) - برای اندازه‌گیری")

        st.caption("""
        این نمای دوبعدی از بالای مش گرفته شده است. 
        برای اندازه‌گیری، روی نقاط مرزی دندان‌ها کلیک کنید.
        """)

        occ_col1, occ_col2 = st.columns(2)

        with occ_col1:
            if mesh_max is not None:
                st.markdown("**فک بالا - نمای اکلوزال**")
                img_max, _, _ = render_occlusal_view(mesh_max, img_size=800)
                if img_max is not None:
                    st.image(img_max, caption="Maxillary Occlusal View", use_container_width=True)

        with occ_col2:
            if mesh_man is not None:
                st.markdown("**فک پایین - نمای اکلوزال**")
                img_man, _, _ = render_occlusal_view(mesh_man, img_size=800)
                if img_man is not None:
                    st.image(img_man, caption="Mandibular Occlusal View", use_container_width=True)

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
                st.warning("⚠️ اضافه حجم دندان‌های قدامی فک پایین")
            elif ant_ratio < 75.5:
                st.info("ℹ️ اضافه حجم دندان‌های قدامی فک بالا")
            else:
                st.success("✅ نسبت قدامی متوازن است.")

    else:
        st.info("💡 برای شروع، لطفاً حداقل یک فایل STL/OBJ آپلود کنید.")
