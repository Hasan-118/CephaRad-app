"""
ماژول اندازه‌گیری نقطه‌به‌نقطه دندان‌ها روی نمای اکلوزال
Aariz Precision Station - Tooth Measurement Module
نسخه: 6.0 - با PyVista برای رندر پایدار
"""

import streamlit as st
import numpy as np
import io
import pandas as pd
from PIL import Image, ImageDraw


# ============================================================
# تلاش برای import PyVista
# ============================================================
PYVISTA_AVAILABLE = False
PYVISTA_ERROR = None
STPYVISTA_AVAILABLE = False

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError as e:
    PYVISTA_ERROR = str(e)

try:
    from stpyvista import stpyvista
    STPYVISTA_AVAILABLE = True
except ImportError as e:
    STPYVISTA_AVAILABLE = False


# ============================================================
# لیست دندان‌ها
# ============================================================

def get_arch_teeth(is_maxilla=True):
    if is_maxilla:
        return [
            {"id": 16, "name": "6 راست", "type": "molar", "side": "right", "order": 6},
            {"id": 15, "name": "5 راست", "type": "premolar", "side": "right", "order": 5},
            {"id": 14, "name": "4 راست", "type": "premolar", "side": "right", "order": 4},
            {"id": 13, "name": "3 راست", "type": "canine", "side": "right", "order": 3},
            {"id": 12, "name": "2 راست", "type": "incisor", "side": "right", "order": 2},
            {"id": 11, "name": "1 راست", "type": "incisor", "side": "right", "order": 1},
            {"id": 21, "name": "1 چپ", "type": "incisor", "side": "left", "order": 1},
            {"id": 22, "name": "2 چپ", "type": "incisor", "side": "left", "order": 2},
            {"id": 23, "name": "3 چپ", "type": "canine", "side": "left", "order": 3},
            {"id": 24, "name": "4 چپ", "type": "premolar", "side": "left", "order": 4},
            {"id": 25, "name": "5 چپ", "type": "premolar", "side": "left", "order": 5},
            {"id": 26, "name": "6 چپ", "type": "molar", "side": "left", "order": 6},
        ]
    else:
        return [
            {"id": 46, "name": "6 راست", "type": "molar", "side": "right", "order": 6},
            {"id": 45, "name": "5 راست", "type": "premolar", "side": "right", "order": 5},
            {"id": 44, "name": "4 راست", "type": "premolar", "side": "right", "order": 4},
            {"id": 43, "name": "3 راست", "type": "canine", "side": "right", "order": 3},
            {"id": 42, "name": "2 راست", "type": "incisor", "side": "right", "order": 2},
            {"id": 41, "name": "1 راست", "type": "incisor", "side": "right", "order": 1},
            {"id": 31, "name": "1 چپ", "type": "incisor", "side": "left", "order": 1},
            {"id": 32, "name": "2 چپ", "type": "incisor", "side": "left", "order": 2},
            {"id": 33, "name": "3 چپ", "type": "canine", "side": "left", "order": 3},
            {"id": 34, "name": "4 چپ", "type": "premolar", "side": "left", "order": 4},
            {"id": 35, "name": "5 چپ", "type": "premolar", "side": "left", "order": 5},
            {"id": 36, "name": "6 چپ", "type": "molar", "side": "left", "order": 6},
        ]


def get_expected_point_type(tooth, is_maxilla=True):
    return "distal" if tooth["side"] == "right" else "mesial"


def get_next_point_type(current_type, tooth):
    if tooth["side"] == "right":
        return "mesial" if current_type == "distal" else None
    else:
        return "distal" if current_type == "mesial" else None


def get_current_point_type(tooth, pts, is_maxilla):
    if not pts:
        return get_expected_point_type(tooth, is_maxilla)
    has_d = "distal" in pts
    has_m = "mesial" in pts
    if has_d and has_m:
        return None
    if has_d:
        return "mesial"
    if has_m:
        return "distal"
    return get_expected_point_type(tooth, is_maxilla)


# ============================================================
# تبدیل trimesh به PyVista
# ============================================================

def trimesh_to_pyvista(mesh):
    """تبدیل مش trimesh به PyVista PolyData"""
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    # PyVista فرمت: [3, i0, i1, i2, 3, i3, i4, i5, ...]
    faces_pv = np.hstack([
        np.full((len(faces), 1), 3),
        faces
    ]).flatten()

    return pv.PolyData(vertices, faces_pv)


# ============================================================
# رندر با PyVista
# ============================================================

def render_occlusal_view_with_pyvista(mesh, points_dict, current_tooth_id,
                                       current_point_type, is_maxilla,
                                       widget_key="occlusal_pv"):
    """
    رندر نمای اکلوزال با PyVista و نمایش با stpyvista.
    """
    if not PYVISTA_AVAILABLE:
        st.error(f"❌ PyVista در دسترس نیست: {PYVISTA_ERROR}")
        return False

    if not STPYVISTA_AVAILABLE:
        st.error("❌ stpyvista در دسترس نیست")
        return False

    try:
        pv_mesh = trimesh_to_pyvista(mesh)
        pv_mesh.compute_normals(
            cell_normals=False, point_normals=True,
            inplace=True, auto_orient_normals=True
        )

        # پلاتر off-screen
        plotter = pv.Plotter(window_size=[900, 900], off_screen=True, border=False)

        # افزودن مش
        plotter.add_mesh(
            pv_mesh,
            color='#F5EFE0',
            smooth_shading=True,
            specular=0.4,
            diffuse=0.85,
            ambient=0.4,
            show_edges=False,
            lighting=True
        )

        # پس‌زمینه سفید
        plotter.background_color = 'white'

        # افزودن نقاط و خطوط
        for tooth_id, pts in points_dict.items():
            if "mesial" in pts:
                mx, my = pts["mesial"]
                color = 'red'
                size = 15 if (tooth_id == current_tooth_id and current_point_type == "mesial") else 10
                plotter.add_points(
                    np.array([[mx, my, 0]]),
                    color=color,
                    point_size=size,
                    render_points_as_spheres=True
                )

            if "distal" in pts:
                dx, dy = pts["distal"]
                color = 'blue'
                size = 15 if (tooth_id == current_tooth_id and current_point_type == "distal") else 10
                plotter.add_points(
                    np.array([[dx, dy, 0]]),
                    color=color,
                    point_size=size,
                    render_points_as_spheres=True
                )

            if "mesial" in pts and "distal" in pts:
                mx, my = pts["mesial"]
                dx, dy = pts["distal"]
                line = pv.Line(
                    np.array([mx, my, 0]),
                    np.array([dx, dy, 0])
                )
                plotter.add_mesh(line, color='green', line_width=2)

        # تنظیم دوربین (نمای از بالا)
        plotter.camera_position = 'xy'
        plotter.camera.elevation = 90
        plotter.camera.azimuth = 0
        plotter.camera.zoom(1.2)

        # نورپردازی
        plotter.remove_all_lights()
        plotter.add_light(pv.Light(position=(1, 1, 1), intensity=0.5))
        plotter.add_light(pv.Light(position=(-1, -1, 1), intensity=0.3))
        plotter.add_light(pv.Light(position=(0, 0, 2), intensity=0.4))

        # نمایش با stpyvista
        stpyvista(plotter, key=widget_key)
        plotter.close()
        return True

    except Exception as e:
        st.error(f"❌ خطا در رندر PyVista: {type(e).__name__}: {e}")
        return False


# ============================================================
# Fallback: رندر با Plotly + تبدیل به تصویر
# ============================================================

def render_occlusal_view_fallback(mesh, img_size=900):
    """رندر نمای اکلوزال با Plotly (روش قبلی)"""
    if mesh is None:
        return None

    try:
        import plotly.graph_objects as go
        vertices = mesh.vertices
        faces = mesh.faces

        fig = go.Figure(data=[
            go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color='#F5EFE0', opacity=1.0, flatshading=False,
                lighting=dict(ambient=0.6, diffuse=0.9, specular=0.3,
                              roughness=0.4, fresnel=0.1),
                lightposition=dict(x=0, y=0, z=1000)
            )
        ])
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False, showbackground=False),
                yaxis=dict(visible=False, showbackground=False),
                zaxis=dict(visible=False, showbackground=False),
                aspectmode='data', bgcolor='white',
                camera=dict(eye=dict(x=0, y=0, z=2.5), up=dict(x=0, y=1, z=0))
            ),
            margin=dict(r=0, l=0, b=0, t=0),
            paper_bgcolor='white', showlegend=False
        )

        img_bytes = fig.to_image(format="png", width=img_size, height=img_size, scale=1)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        return img
    except Exception as e:
        st.warning(f"⚠️ خطا در رندر Plotly: {e}")
        return None


# ============================================================
# مدیریت state
# ============================================================

def init_measurement_state(is_maxilla):
    key = "max" if is_maxilla else "man"
    if f"missing_teeth_{key}" not in st.session_state:
        st.session_state[f"missing_teeth_{key}"] = set()
    if f"tooth_points_{key}" not in st.session_state:
        st.session_state[f"tooth_points_{key}"] = {}
    if not isinstance(st.session_state[f"tooth_points_{key}"], dict):
        st.session_state[f"tooth_points_{key}"] = {}
    if f"current_tooth_idx_{key}" not in st.session_state:
        st.session_state[f"current_tooth_idx_{key}"] = 0
    if not isinstance(st.session_state[f"current_tooth_idx_{key}"], int):
        st.session_state[f"current_tooth_idx_{key}"] = 0


# ============================================================
# محاسبات
# ============================================================

def compute_tooth_widths(points_dict, teeth_list, missing_set, pixel_size_mm=0.1):
    results = []
    previous_distal = None

    for tooth in teeth_list:
        tooth_id = tooth["id"]
        if tooth_id in missing_set:
            results.append({
                "tooth_id": tooth_id, "tooth_name": tooth["name"],
                "type": tooth["type"], "width_mm": None,
                "space_before_mm": None, "status": "غایب (Missing)"
            })
            continue

        pts = points_dict.get(tooth_id, {})
        if "mesial" not in pts or "distal" not in pts:
            results.append({
                "tooth_id": tooth_id, "tooth_name": tooth["name"],
                "type": tooth["type"], "width_mm": None,
                "space_before_mm": None, "status": "علامت‌گذاری نشده"
            })
            continue

        mx, my = pts["mesial"]
        dx, dy = pts["distal"]
        distance_px = np.sqrt((mx - dx) ** 2 + (my - dy) ** 2)
        distance_mm = distance_px * pixel_size_mm

        space_mm = None
        if previous_distal is not None:
            pdx, pdy = previous_distal
            gap_px = np.sqrt((mx - pdx) ** 2 + (my - pdy) ** 2)
            space_mm = gap_px * pixel_size_mm

        if space_mm is None:
            status = "—"
        elif space_mm > 1.0:
            status = f"⚠️ Spacing ({space_mm:.1f} mm)"
        elif space_mm < -1.0:
            status = f"⚠️ Crowding ({space_mm:.1f} mm)"
        else:
            status = "✅ نرمال"

        results.append({
            "tooth_id": tooth_id, "tooth_name": tooth["name"],
            "type": tooth["type"], "width_mm": round(distance_mm, 2),
            "space_before_mm": round(space_mm, 2) if space_mm is not None else None,
            "status": status
        })
        previous_distal = pts["distal"]

    return results


def compute_bolton_summary(widths_max, widths_man):
    max_valid = [t for t in widths_max if t["width_mm"] is not None]
    man_valid = [t for t in widths_man if t["width_mm"] is not None]
    anterior_types = ["incisor", "canine"]
    max_anterior = [t for t in max_valid if t["type"] in anterior_types]
    man_anterior = [t for t in man_valid if t["type"] in anterior_types]

    max_total = sum(t["width_mm"] for t in max_valid) if max_valid else 0
    man_total = sum(t["width_mm"] for t in man_valid) if man_valid else 0
    max_ant = sum(t["width_mm"] for t in max_anterior) if max_anterior else 0
    man_ant = sum(t["width_mm"] for t in man_anterior) if man_anterior else 0

    return {
        "max_total": round(max_total, 2), "man_total": round(man_total, 2),
        "max_anterior": round(max_ant, 2), "man_anterior": round(man_ant, 2),
        "overall_ratio": round((man_total / max_total) * 100, 2) if max_total > 0 else 0,
        "anterior_ratio": round((man_ant / max_ant) * 100, 2) if max_ant > 0 else 0,
        "max_count": len(max_valid), "man_count": len(man_valid),
    }


# ============================================================
# رابط کاربری اصلی
# ============================================================

def render_tooth_measurement_tab(mesh_max=None, mesh_man=None, pixel_size_default=0.1):

    if mesh_max is None:
        mesh_max = st.session_state.get("uploaded_mesh_max", None)
    if mesh_man is None:
        mesh_man = st.session_state.get("uploaded_mesh_man", None)

    # --- نمایش وضعیت PyVista ---
    if PYVISTA_AVAILABLE and STPYVISTA_AVAILABLE:
        st.success("✅ PyVista و stpyvista آماده هستند")
    elif PYVISTA_AVAILABLE and not STPYVISTA_AVAILABLE:
        st.warning("⚠️ PyVista هست ولی stpyvista نیست - از روش جایگزین استفاده می‌شود")
    else:
        st.error(f"❌ PyVista در دسترس نیست: {PYVISTA_ERROR}")

    st.header("📏 اندازه‌گیری نقطه‌به‌نقطه عرض دندان‌ها")

    arch = st.radio(
        "انتخاب فک:",
        ["🦷 فک بالا (Maxilla)", "🦷 فک پایین (Mandible)"],
        horizontal=True,
        key="measurement_arch_pv"
    )
    is_maxilla = "بالا" in arch
    key = "max" if is_maxilla else "man"

    mesh = mesh_max if is_maxilla else mesh_man
    if mesh is None:
        st.warning(f"⚠️ فک {'بالا' if is_maxilla else 'پایین'} آپلود نشده است.")
        return None

    init_measurement_state(is_maxilla)

    col_px1, col_px2 = st.columns([1, 3])
    with col_px1:
        pixel_size_mm = st.number_input(
            "Pixel Size (mm/px):",
            min_value=0.01, max_value=1.0,
            value=pixel_size_default, step=0.01,
            format="%.3f", key=f"px_size_{key}_pv"
        )

    teeth = get_arch_teeth(is_maxilla=is_maxilla)
    missing_teeth = st.session_state[f"missing_teeth_{key}"]

    # --- Missing teeth ---
    with st.expander("📋 مشخص کردن دندان‌های غایب (Missing)", expanded=False):
        cols = st.columns(6)
        changed = False
        for idx, tooth in enumerate(teeth):
            with cols[idx % 6]:
                was_missing = tooth["id"] in missing_teeth
                is_missing = st.checkbox(
                    tooth["name"],
                    value=was_missing,
                    key=f"missing_cb_{key}_{tooth['id']}_pv"
                )
                if is_missing != was_missing:
                    if is_missing:
                        missing_teeth.add(tooth["id"])
                    else:
                        missing_teeth.discard(tooth["id"])
                    changed = True

        if changed:
            st.session_state[f"missing_teeth_{key}"] = missing_teeth
            st.session_state[f"current_tooth_idx_{key}"] = 0
            st.rerun()

        st.info(f"📊 غایب: **{len(missing_teeth)}** | موجود: **{len(teeth) - len(missing_teeth)}**")

    available_teeth = [t for t in teeth if t["id"] not in missing_teeth]
    if not available_teeth:
        st.warning("⚠️ همه دندان‌ها غایب هستند.")
        return None

    tooth_points = st.session_state[f"tooth_points_{key}"]

    completed_count = sum(
        1 for t in available_teeth
        if t["id"] in tooth_points
        and "mesial" in tooth_points[t["id"]]
        and "distal" in tooth_points[t["id"]]
    )
    total_count = len(available_teeth)

    # --- بررسی اتمام ---
    if completed_count == total_count and total_count > 0:
        st.success(f"✅ فک {'بالا' if is_maxilla else 'پایین'} کامل شد! ({total_count} / {total_count})")

        widths = compute_tooth_widths(tooth_points, teeth, missing_teeth, pixel_size_mm)
        total_width = sum(w["width_mm"] for w in widths if w["width_mm"] is not None)
        st.metric("مجموع عرض", f"{round(total_width, 2)} mm")

        table_data = []
        for w in widths:
            width_str = f"{w['width_mm']} mm" if w["width_mm"] else "—"
            space_str = f"{w['space_before_mm']} mm" if w["space_before_mm"] is not None else "—"
            table_data.append({
                "دندان": f"{w['tooth_name']} ({w['tooth_id']})",
                "نوع": w["type"], "عرض (mm)": width_str,
                "فاصله با قبلی": space_str, "وضعیت": w["status"]
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        st.session_state[f"measured_widths_{key}"] = widths

        if st.button("🔄 شروع مجدد این فک", key=f"reset_arch_{key}_pv"):
            st.session_state[f"tooth_points_{key}"] = {}
            st.session_state[f"current_tooth_idx_{key}"] = 0
            st.rerun()

        return widths

    # --- پیدا کردن اولین دندان ناتمام ---
    current_idx = st.session_state.get(f"current_tooth_idx_{key}", 0)
    if current_idx < 0:
        current_idx = 0
    if current_idx >= len(available_teeth):
        current_idx = len(available_teeth) - 1

    while current_idx < len(available_teeth):
        t = available_teeth[current_idx]
        pts = tooth_points.get(t["id"], {})
        if "distal" in pts and "mesial" in pts:
            current_idx += 1
        else:
            break

    if current_idx >= len(available_teeth):
        current_idx = len(available_teeth) - 1

    st.session_state[f"current_tooth_idx_{key}"] = current_idx
    current_tooth = available_teeth[current_idx]

    pts = tooth_points.get(current_tooth["id"], {})
    current_type = get_current_point_type(current_tooth, pts, is_maxilla)

    if current_type is None:
        if current_idx < len(available_teeth) - 1:
            st.session_state[f"current_tooth_idx_{key}"] = current_idx + 1
            st.rerun()
        current_type = get_expected_point_type(current_tooth, is_maxilla)

    # نمایش وضعیت
    st.markdown("### 🎯 علامت‌گذاری")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("دندان فعلی", current_tooth['name'])
    with col2:
        side = "🟢 راست" if current_tooth["side"] == "right" else "🔵 چپ"
        st.metric("سمت", side)
    with col3:
        type_label = "🔵 دیستال" if current_type == "distal" else "🔴 مزیال"
        st.metric("نقطه", type_label)
    with col4: st.metric("پیشرفت", f"{completed_count} / {total_count}")

    # ============ دکمه‌ها ============
    col_nav1, col_nav2, col_nav3, col_nav4, col_nav5, col_nav6 = st.columns(6)

    with col_nav1:
        if st.button("◀ قبلی", use_container_width=True, key=f"prev_{key}_pv"):
            new_idx = max(0, current_idx - 1)
            st.session_state[f"current_tooth_idx_{key}"] = new_idx
            st.rerun()

    with col_nav2:
        if st.button("⏭ بعدی", use_container_width=True, key=f"next_{key}_pv"):
            new_idx = min(len(available_teeth) - 1, current_idx + 1)
            st.session_state[f"current_tooth_idx_{key}"] = new_idx
            st.rerun()

    with col_nav3:
        if st.button("↩️ پاک آخرین نقطه", use_container_width=True, key=f"undo_{key}_pv"):
            points = dict(st.session_state.get(f"tooth_points_{key}", {}))
            pts_c = dict(points.get(current_tooth["id"], {}))

            removed = False
            if current_type and current_type in pts_c:
                del pts_c[current_type]
                removed = True
            elif pts_c:
                order = ["distal", "mesial"] if current_tooth["side"] == "right" else ["mesial", "distal"]
                for pt_type in reversed(order):
                    if pt_type in pts_c:
                        del pts_c[pt_type]
                        removed = True
                        break

            if pts_c:
                points[current_tooth["id"]] = pts_c
            else:
                points.pop(current_tooth["id"], None)

            st.session_state[f"tooth_points_{key}"] = points
            st.rerun()

    with col_nav4:
        if st.button("🔵 دیستال", use_container_width=True, key=f"set_d_{key}_pv"):
            points = dict(st.session_state.get(f"tooth_points_{key}", {}))
            pts_c = dict(points.get(current_tooth["id"], {}))
            if "distal" in pts_c:
                del pts_c["distal"]
            points[current_tooth["id"]] = pts_c
            st.session_state[f"tooth_points_{key}"] = points
            st.rerun()

    with col_nav5:
        if st.button("🔴 مزیال", use_container_width=True, key=f"set_m_{key}_pv"):
            points = dict(st.session_state.get(f"tooth_points_{key}", {}))
            pts_c = dict(points.get(current_tooth["id"], {}))
            if "mesial" in pts_c:
                del pts_c["mesial"]
            points[current_tooth["id"]] = pts_c
            st.session_state[f"tooth_points_{key}"] = points
            st.rerun()

    with col_nav6:
        if st.button("🗑 پاک دندان", use_container_width=True, key=f"clear_{key}_pv"):
            points = dict(st.session_state.get(f"tooth_points_{key}", {}))
            points.pop(current_tooth["id"], None)
            st.session_state[f"tooth_points_{key}"] = points
            st.rerun()

    # ============ رندر PyVista ============
    st.markdown(f"**👆 روی مش کلیک کنید تا نقطه {current_type} دندان {current_tooth['name']} ثبت شود:**")

    # نمایش مش با PyVista
    render_occlusal_view_with_pyvista(
        mesh, tooth_points, current_tooth["id"],
        current_type, is_maxilla, widget_key=f"occlusal_pv_{key}"
    )

    # ============ نمایش نتایج جزئی ============
    st.divider()
    st.markdown("### 📊 نتایج اندازه‌گیری")

    if len(tooth_points) == 0:
        st.info("ℹ️ هنوز هیچ دندانی علامت‌گذاری نشده است.")
        return None

    widths = compute_tooth_widths(tooth_points, teeth, missing_teeth, pixel_size_mm)

    table_data = []
    for w in widths:
        width_str = f"{w['width_mm']} mm" if w["width_mm"] else "—"
        space_str = f"{w['space_before_mm']} mm" if w["space_before_mm"] is not None else "—"
        table_data.append({
            "دندان": f"{w['tooth_name']} ({w['tooth_id']})",
            "نوع": w["type"], "عرض (mm)": width_str,
            "فاصله با قبلی": space_str, "وضعیت": w["status"]
        })

    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    total_width = sum(w["width_mm"] for w in widths if w["width_mm"] is not None)
    st.metric("مجموع عرض دندان‌های علامت‌گذاری‌شده", f"{round(total_width, 2)} mm")

    st.session_state[f"measured_widths_{key}"] = widths

    return widths
