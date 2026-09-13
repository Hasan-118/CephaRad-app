"""
ماژول اندازه‌گیری نقطه‌به‌نقطه دندان‌ها روی نمای اکلوزال
Aariz Precision Station - Tooth Measurement Module
نسخه: 1.1 - با پشتیبانی از session_state و مش‌های پاس داده شده
"""

import streamlit as st
import numpy as np
import io
import pandas as pd
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates


# ============================================================
# لیست دندان‌ها (FDI Notation)
# ============================================================

def get_arch_teeth(is_maxilla=True):
    """لیست دندان‌ها از آخرین دندان سمت راست تا آخرین دندان سمت چپ"""
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
    """نقطه شروع برای دندان (بر اساس سمت)"""
    if tooth["side"] == "right":
        return "distal"
    else:
        return "mesial"


def get_next_point_type(current_type, tooth):
    """نقطه بعدی برای همان دندان"""
    if tooth["side"] == "right":
        if current_type == "distal":
            return "mesial"
        return None
    else:
        if current_type == "mesial":
            return "distal"
        return None


# ============================================================
# رندر نمای اکلوزال
# ============================================================

def render_occlusal_view(mesh, img_size=900, use_top_surface=True):
    """رندر نمای اکلوزال (از بالا) از مش STL"""
    if mesh is None:
        return None, None

    vertices = mesh.vertices

    x_min, y_min, z_min = vertices.min(axis=0)
    x_max, y_max, z_max = vertices.max(axis=0)

    x_range = x_max - x_min
    y_range = y_max - y_min

    pad_ratio = 0.08
    pad_x = x_range * pad_ratio
    pad_y = y_range * pad_ratio

    x_min -= pad_x
    x_max += pad_x
    y_min -= pad_y
    y_max += pad_y

    scale_x = (img_size - 40) / (x_max - x_min)
    scale_y = (img_size - 40) / (y_max - y_min)
    scale = min(scale_x, scale_y)

    img = Image.new('RGB', (img_size, img_size), (250, 248, 245))
    draw = ImageDraw.Draw(img)

    if use_top_surface:
        z_threshold = z_min + (z_max - z_min) * 0.55
    else:
        z_threshold = z_min

    visible_verts = vertices[vertices[:, 2] > z_threshold]

    for v in visible_verts:
        px = int((v[0] - x_min) * scale + 20)
        py = int(img_size - (v[1] - y_min) * scale - 20)

        if 0 <= px < img_size and 0 <= py < img_size:
            z_norm = (v[2] - z_threshold) / (z_max - z_threshold + 1e-9)
            intensity = int(230 - 60 * (1 - z_norm))
            intensity = max(140, min(230, intensity))
            draw.point((px, py), fill=(intensity, intensity - 5, intensity - 10))

    transform_info = {
        "x_min": x_min,
        "y_min": y_min,
        "scale": scale,
        "img_size": img_size,
        "padding": 20,
    }

    return img, transform_info


# ============================================================
# مدیریت وضعیت
# ============================================================

def init_measurement_state(is_maxilla):
    """مقداردهی اولیه وضعیت"""
    key = "max" if is_maxilla else "man"

    if f"missing_teeth_{key}" not in st.session_state:
        st.session_state[f"missing_teeth_{key}"] = set()

    if f"tooth_points_{key}" not in st.session_state:
        st.session_state[f"tooth_points_{key}"] = {}

    if f"current_tooth_idx_{key}" not in st.session_state:
        st.session_state[f"current_tooth_idx_{key}"] = 0


def draw_points_on_image(img, points_dict, teeth_list, missing_set,
                          current_tooth_id=None, current_point_type=None,
                          is_maxilla=True):
    """رسم نقاط روی تصویر"""
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)

    COLOR_MESIAL = (220, 38, 38)
    COLOR_DISTAL = (37, 99, 235)
    COLOR_LINE = (16, 185, 129)

    for tooth_id, pts in points_dict.items():
        if "mesial" in pts and "distal" in pts:
            mx, my = pts["mesial"]
            dx, dy = pts["distal"]
            draw.line([(mx, my), (dx, dy)], fill=COLOR_LINE, width=2)

    for tooth_id, pts in points_dict.items():
        if "mesial" in pts:
            mx, my = pts["mesial"]
            r = 8 if (tooth_id == current_tooth_id and current_point_type == "mesial") else 5
            draw.ellipse([mx - r, my - r, mx + r, my + r],
                         fill=COLOR_MESIAL, outline="white", width=2)
            if tooth_id == current_tooth_id:
                draw.text((mx + 12, my - 8), f"{tooth_id}M", fill=COLOR_MESIAL)

        if "distal" in pts:
            dx, dy = pts["distal"]
            r = 8 if (tooth_id == current_tooth_id and current_point_type == "distal") else 5
            draw.ellipse([dx - r, dy - r, dx + r, dy + r],
                         fill=COLOR_DISTAL, outline="white", width=2)
            if tooth_id == current_tooth_id:
                draw.text((dx + 12, dy - 8), f"{tooth_id}D", fill=COLOR_DISTAL)

    # راهنما
    draw.rectangle([10, 10, 340, 100], fill=(255, 255, 255), outline=(200, 200, 200))
    draw.text((20, 20), f"🦷 {'فک بالا' if is_maxilla else 'فک پایین'}", fill=(0, 0, 0))
    draw.text((20, 45), "🔴 مزیال (M)  🔵 دیستال (D)", fill=(60, 60, 60))
    draw.text((20, 70), "مسیر: از راست به چپ", fill=(60, 60, 60))

    return img_copy


# ============================================================
# محاسبات
# ============================================================

def compute_tooth_widths(points_dict, teeth_list, missing_set, pixel_size_mm=0.1):
    """محاسبه عرض و فضای بین دندانی"""
    results = []
    previous_distal = None

    for tooth in teeth_list:
        tooth_id = tooth["id"]

        if tooth_id in missing_set:
            results.append({
                "tooth_id": tooth_id,
                "tooth_name": tooth["name"],
                "type": tooth["type"],
                "width_mm": None,
                "space_before_mm": None,
                "status": "غایب (Missing)"
            })
            continue

        pts = points_dict.get(tooth_id, {})

        if "mesial" not in pts or "distal" not in pts:
            results.append({
                "tooth_id": tooth_id,
                "tooth_name": tooth["name"],
                "type": tooth["type"],
                "width_mm": None,
                "space_before_mm": None,
                "status": "علامت‌گذاری نشده"
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
            "tooth_id": tooth_id,
            "tooth_name": tooth["name"],
            "type": tooth["type"],
            "width_mm": round(distance_mm, 2),
            "space_before_mm": round(space_mm, 2) if space_mm is not None else None,
            "status": status
        })

        previous_distal = pts["distal"]

    return results


def compute_bolton_summary(widths_max, widths_man):
    """محاسبه نسبت‌های بولتون"""
    max_valid = [t for t in widths_max if t["width_mm"] is not None]
    man_valid = [t for t in widths_man if t["width_mm"] is not None]

    anterior_types = ["incisor", "canine"]

    max_anterior = [t for t in max_valid if t["type"] in anterior_types]
    man_anterior = [t for t in man_valid if t["type"] in anterior_types]

    max_total = sum(t["width_mm"] for t in max_valid) if max_valid else 0
    man_total = sum(t["width_mm"] for t in man_valid) if man_valid else 0

    max_ant = sum(t["width_mm"] for t in max_anterior) if max_anterior else 0
    man_ant = sum(t["width_mm"] for t in man_anterior) if man_anterior else 0

    overall_ratio = round((man_total / max_total) * 100, 2) if max_total > 0 else 0
    anterior_ratio = round((man_ant / max_ant) * 100, 2) if max_ant > 0 else 0

    return {
        "max_total": round(max_total, 2),
        "man_total": round(man_total, 2),
        "max_anterior": round(max_ant, 2),
        "man_anterior": round(man_ant, 2),
        "overall_ratio": overall_ratio,
        "anterior_ratio": anterior_ratio,
        "max_count": len(max_valid),
        "man_count": len(man_valid),
    }


# ============================================================
# رابط کاربری اصلی
# ============================================================

def render_tooth_measurement_tab(mesh_max=None, mesh_man=None, pixel_size_default=0.1):
    """رابط کاربری کامل اندازه‌گیری نقطه‌به‌نقطه"""
    
    # اگر مش‌ها پاس نشدند، از session_state بخوان
    if mesh_max is None:
        mesh_max = st.session_state.get("uploaded_mesh_max", None)
    if mesh_man is None:
        mesh_man = st.session_state.get("uploaded_mesh_man", None)
    
    st.header("📏 اندازه‌گیری نقطه‌به‌نقطه عرض دندان‌ها")
    st.info("""
    **راهنمای ترتیب علامت‌گذاری:**
    - از **آخرین دندان سمت راست** شروع کنید
    - برای هر دندان سمت راست: ابتدا **دیستال** (🔵) سپس **مزیال** (🔴)
    - برای هر دندان سمت چپ: ابتدا **مزیال** (🔴) سپس **دیستال** (🔵)
    """)

    arch = st.radio(
        "انتخاب فک:",
        ["🦷 فک بالا (Maxilla)", "🦷 فک پایین (Mandible)"],
        horizontal=True,
        key="measurement_arch"
    )
    is_maxilla = "بالا" in arch

    mesh = mesh_max if is_maxilla else mesh_man
    if mesh is None:
        st.warning(f"⚠️ فک {'بالا' if is_maxilla else 'پایین'} آپلود نشده است.")
        return None

    init_measurement_state(is_maxilla)
    key = "max" if is_maxilla else "man"

    col_px1, col_px2 = st.columns([1, 3])
    with col_px1:
        pixel_size_mm = st.number_input(
            "Pixel Size (mm/px):",
            min_value=0.01, max_value=1.0,
            value=pixel_size_default, step=0.01,
            format="%.3f",
            key=f"px_size_{key}"
        )

    with st.spinner("در حال رندر نمای اکلوزال..."):
        occ_img, transform_info = render_occlusal_view(mesh, img_size=900)

    if occ_img is None:
        st.error("❌ خطا در رندر نمای اکلوزال")
        return None

    teeth = get_arch_teeth(is_maxilla=is_maxilla)
    missing_teeth = st.session_state[f"missing_teeth_{key}"]

    # Missing teeth
    with st.expander("📋 مشخص کردن دندان‌های غایب (Missing)", expanded=False):
        st.markdown("دندان‌های غایب را تیک بزنید:")
        cols = st.columns(6)
        for idx, tooth in enumerate(teeth):
            with cols[idx % 6]:
                is_missing = tooth["id"] in missing_teeth
                checkbox = st.checkbox(
                    tooth["name"],
                    value=is_missing,
                    key=f"missing_{key}_{tooth['id']}"
                )
                if checkbox and tooth["id"] not in missing_teeth:
                    missing_teeth.add(tooth["id"])
                    st.rerun()
                elif not checkbox and tooth["id"] in missing_teeth:
                    missing_teeth.discard(tooth["id"])
