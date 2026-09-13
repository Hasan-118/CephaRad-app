"""
Aariz 3D Analysis Station
اپلیکیشن تحلیل سه‌بعدی اسکن داخل دهانی + ادغام با سفالومتری
نسخه: 2.2 - نمایش مستقیم نمای سه‌بعدی + اندازه‌گیری نقطه‌به‌نقطه
"""

import streamlit as st
import json
import os
import pickle
import hashlib
from datetime import datetime
from pathlib import Path

# --- تنظیمات صفحه ---
st.set_page_config(
    page_title="Aariz 3D Analysis Station",
    layout="wide",
    page_icon="🦷"
)

# --- بارگذاری ماژول ۳D ---
try:
    from intraoral_3d_module import (
        parse_mesh,
        create_3d_plotly_figure,
        safe_simplify_mesh,
    )
except ImportError as e:
    st.error(f"❌ ماژول `intraoral_3d_module.py` یافت نشد: {e}")
    st.stop()

# --- بارگذاری ماژول اندازه‌گیری ---
try:
    from tooth_measurement import (
        render_tooth_measurement_tab,
        compute_bolton_summary,
        get_arch_teeth,
    )
    MEASUREMENT_AVAILABLE = True
    MEASUREMENT_ERROR = None
except ImportError as e:
    MEASUREMENT_AVAILABLE = False
    MEASUREMENT_ERROR = str(e)

# --- استایل ---
st.markdown("""
<style>
    html, body, [class*="css"]  { font-size: 14px; }
    .stButton>button { padding: 0.3rem 0.6rem; font-size: 13px; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# توابع ذخیره‌سازی مش در فایل
# ============================================================

CACHE_DIR = Path("/tmp/aariz_mesh_cache")
try:
    CACHE_DIR.mkdir(exist_ok=True)
except Exception:
    pass


def save_mesh_to_cache(mesh, mesh_key):
    """ذخیره مش در فایل موقت روی سرور"""
    if mesh is None:
        return None
    try:
        cache_path = CACHE_DIR / f"{mesh_key}.pkl"
        with open(cache_path, 'wb') as f:
            pickle.dump(mesh, f)
        return str(cache_path)
    except Exception as e:
        st.warning(f"⚠️ خطا در ذخیره مش: {e}")
        return None


def load_mesh_from_cache(mesh_key):
    """بارگذاری مش از فایل موقت"""
    try:
        cache_path = CACHE_DIR / f"{mesh_key}.pkl"
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
    except Exception:
        pass
    return None


def get_file_hash(uploaded_file):
    """محاسبه hash فایل برای شناسایی یکتا"""
    if uploaded_file is None:
        return None
    try:
        uploaded_file.seek(0)
        content = uploaded_file.read()
        uploaded_file.seek(0)
        return hashlib.md5(content).hexdigest()[:12]
    except Exception:
        return None


# ============================================================
# عنوان
# ============================================================
st.title("🦷 ایستگاه تحلیل سه‌بعدی Aariz")
st.caption("Aariz 3D Analysis Station - Intraoral Scan")

# ============================================================
# سایدبار: آپلود JSON سفالومتری
# ============================================================
st.sidebar.header("📂 ادغام با تحلیل سفالومتری")
st.sidebar.markdown("""
اگر قبلاً تحلیل سفالومتری (۲D) را انجام داده‌اید،
فایل JSON دانلود شده را اینجا آپلود کنید تا گزارش نهایی یکپارچه شود.
""")

uploaded_json = st.sidebar.file_uploader(
    "آپلود نتایج سفالومتری (JSON):",
    type=['json'],
    key="ceph_json_upload_v3"
)

ceph_results = None
if uploaded_json is not None:
    try:
        uploaded_json.seek(0)
        ceph_results = json.load(uploaded_json)

        if 'version' not in ceph_results or 'measurements' not in ceph_results:
            st.sidebar.error("❌ فایل JSON نامعتبر است. (باید ساختار جدید داشته باشد)")
            st.sidebar.info("💡 لطفاً از اپ سفالومتری، فایل JSON **جدید** دانلود کنید.")
            ceph_results = None
        else:
            st.sidebar.success("✅ نتایج سفالومتری بارگذاری شد")
            st.sidebar.caption(f"تاریخ: {ceph_results.get('timestamp', 'نامشخص')[:10]}")

            with st.sidebar.expander("مشاهده خلاصه نتایج ۲D", expanded=False):
                m = ceph_results.get('measurements', {})
                st.write(f"**SNA:** {m.get('SNA', 'N/A')}°")
                st.write(f"**SNB:** {m.get('SNB', 'N/A')}°")
                st.write(f"**ANB:** {m.get('ANB', 'N/A')}°")
                st.write(f"**FMA:** {m.get('FMA', 'N/A')}°")
                st.write(f"**Wits:** {m.get('Wits', 'N/A')} mm")
    except json.JSONDecodeError:
        st.sidebar.error("❌ فایل JSON قابل خواندن نیست.")
        ceph_results = None
    except Exception as e:
        st.sidebar.error(f"❌ خطا: {e}")
        ceph_results = None
else:
    st.sidebar.info("ℹ️ بدون فایل JSON، فقط تحلیل ۳D نمایش داده می‌شود.")

# --- نمایش وضعیت ---
if ceph_results:
    st.success("🔗 **حالت یکپارچه فعال** — گزارش نهایی شامل هر دو تحلیل ۲D و ۳D خواهد بود.")
else:
    st.info("ℹ️ **حالت مستقل** — برای ادغام با سفالومتری، فایل JSON را در سایدبار آپلود کنید.")

st.divider()

# ============================================================
# بخش ۱: آپلود STL
# ============================================================
st.subheader("📤 آپلود اسکن‌های سه‌بعدی")

col_up1, col_up2 = st.columns(2)
with col_up1:
    stl_maxilla = st.file_uploader("آپلود اسکن فک بالا (Maxilla STL/OBJ):",
                                    type=['stl', 'obj'], key="max_stl_app3d_v2")
with col_up2:
    stl_mandible = st.file_uploader("آپلود اسکن فک پایین (Mandible STL/OBJ):",
                                     type=['stl', 'obj'], key="man_stl_app3d_v2")

# --- ذخیره‌سازی کلیدهای cache ---
if 'mesh_cache_key_max' not in st.session_state:
    st.session_state.mesh_cache_key_max = None
if 'mesh_cache_key_man' not in st.session_state:
    st.session_state.mesh_cache_key_man = None

# --- مش فک بالا ---
if stl_maxilla is not None:
    file_hash_max = get_file_hash(stl_maxilla)
    cache_key_max = f"max_{file_hash_max}"

    mesh_max_loaded = None

    if st.session_state.mesh_cache_key_max == cache_key_max:
        mesh_max_loaded = st.session_state.get("uploaded_mesh_max", None)

    if mesh_max_loaded is None:
        mesh_max_loaded = load_mesh_from_cache(cache_key_max)
        if mesh_max_loaded is not None:
            st.session_state["uploaded_mesh_max"] = mesh_max_loaded
            st.session_state.mesh_cache_key_max = cache_key_max

    if mesh_max_loaded is None:
        try:
            mesh_max_loaded = parse_mesh(stl_maxilla)
            if mesh_max_loaded is not None:
                st.session_state["uploaded_mesh_max"] = mesh_max_loaded
                st.session_state.mesh_cache_key_max = cache_key_max
                save_mesh_to_cache(mesh_max_loaded, cache_key_max)
        except Exception as e:
            st.error(f"❌ خطا در بارگذاری فک بالا: {e}")

    if mesh_max_loaded is not None:
        st.success(f"✅ فک بالا: {len(mesh_max_loaded.vertices)} رأس")

# --- مش فک پایین ---
if stl_mandible is not None:
    file_hash_man = get_file_hash(stl_mandible)
    cache_key_man = f"man_{file_hash_man}"

    mesh_man_loaded = None

    if st.session_state.mesh_cache_key_man == cache_key_man:
        mesh_man_loaded = st.session_state.get("uploaded_mesh_man", None)

    if mesh_man_loaded is None:
        mesh_man_loaded = load_mesh_from_cache(cache_key_man)
        if mesh_man_loaded is not None:
            st.session_state["uploaded_mesh_man"] = mesh_man_loaded
            st.session_state.mesh_cache_key_man = cache_key_man

    if mesh_man_loaded is None:
        try:
            mesh_man_loaded = parse_mesh(stl_mandible)
            if mesh_man_loaded is not None:
                st.session_state["uploaded_mesh_man"] = mesh_man_loaded
                st.session_state.mesh_cache_key_man = cache_key_man
                save_mesh_to_cache(mesh_man_loaded, cache_key_man)
        except Exception as e:
            st.error(f"❌ خطا در بارگذاری فک پایین: {e}")

    if mesh_man_loaded is not None:
        st.success(f"✅ فک پایین: {len(mesh_man_loaded.vertices)} رأس")

# ============================================================
# بخش ۲: نمایش مستقیم نمای سه‌بعدی (Plotly)
# ============================================================
mesh_max_current = st.session_state.get("uploaded_mesh_max", None)
mesh_man_current = st.session_state.get("uploaded_mesh_man", None)

if mesh_max_current is not None or mesh_man_current is not None:
    st.divider()
    st.subheader("🖼 نمای سه‌بعدی (قابل چرخش)")

    view_col1, view_col2 = st.columns(2)
    with view_col1:
        if mesh_max_current is not None:
            st.markdown("**فک بالا (Maxilla)**")
            try:
                mesh_simple = safe_simplify_mesh(mesh_max_current, target_faces=20000)
                fig_max = create_3d_plotly_figure(mesh_simple, "Maxillary Arch")
                st.plotly_chart(fig_max, use_container_width=True, key="plotly_max_app3d")
            except Exception as e:
                st.error(f"خطا در نمایش فک بالا: {e}")

    with view_col2:
        if mesh_man_current is not None:
            st.markdown("**فک پایین (Mandible)**")
            try:
                mesh_simple = safe_simplify_mesh(mesh_man_current, target_faces=20000)
                fig_man = create_3d_plotly_figure(mesh_simple, "Mandibular Arch")
                st.plotly_chart(fig_man, use_container_width=True, key="plotly_man_app3d")
            except Exception as e:
                st.error(f"خطا در نمایش فک پایین: {e}")
else:
    st.info("💡 برای شروع، لطفاً حداقل یک فایل STL/OBJ آپلود کنید.")

# ============================================================
# بخش ۳: اندازه‌گیری نقطه‌به‌نقطه
# ============================================================
st.divider()

if MEASUREMENT_AVAILABLE:
    if mesh_max_current is not None or mesh_man_current is not None:
        st.header("🎯 اندازه‌گیری نقطه‌به‌نقطه (ترتیب هوشمند)")
        st.caption("""
        این بخش روی **نمای اکلوزال** (از بالا) کار می‌کند. برای هر دندان:
        - **دندان‌های سمت راست:** ابتدا **دیستال** (🔵)، سپس **مزیال** (🔴)
        - **دندان‌های سمت چپ:** ابتدا **مزیال** (🔴)، سپس **دیستال** (🔵)
        """)

        try:
            widths_max = render_tooth_measurement_tab(mesh_max_current, mesh_man_current)

            widths_man_stored = st.session_state.get("measured_widths_man", None)
            widths_max_stored = st.session_state.get("measured_widths_max", None)

            if widths_man_stored is not None and widths_max_stored is not None:
                st.divider()
                st.subheader("🔢 نسبت‌های بولتون (بر اساس اندازه‌گیری واقعی)")

                bolton = compute_bolton_summary(widths_max_stored, widths_man_stored)

                col1, col2 = st.columns(2)
                with col1:
                    diff = round(bolton["overall_ratio"] - 91.3, 2)
                    st.metric("Overall Bolton", f"{bolton['overall_ratio']}%", f"{diff}%")
                with col2:
                    diff_ant = round(bolton["anterior_ratio"] - 77.2, 2)
                    st.metric("Anterior Bolton", f"{bolton['anterior_ratio']}%", f"{diff_ant}%")

                st.caption(f"📊 تعداد دندان‌های اندازه‌گیری‌شده: فک بالا {bolton['max_count']} | فک پایین {bolton['man_count']}")
                st.session_state["bolton_auto"] = bolton
        except Exception as e:
            st.error(f"❌ خطا در بخش اندازه‌گیری: {type(e).__name__}: {e}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.header("🎯 اندازه‌گیری نقطه‌به‌نقطه (ترتیب هوشمند)")
        st.info("برای فعال شدن این بخش، ابتدا اسکن‌های STL را در بخش بالا آپلود کنید.")
else:
    st.header("🎯 اندازه‌گیری نقطه‌به‌نقطه (ترتیب هوشمند)")
    st.error(f"❌ ماژول `tooth_measurement.py` یافت نشد: {MEASUREMENT_ERROR}")

# ============================================================
# بخش ۴: گزارش یکپارچه PDF
# ============================================================
if ceph_results:
    st.divider()
    st.header("📄 گزارش نهایی یکپارچه")
    st.markdown("**گزارش PDF یکپارچه** شامل هر دو تحلیل سفالومتری (۲D) و اسکن داخل دهانی (۳D).")

    if st.button("🖨 تولید گزارش یکپارچه PDF", use_container_width=True, key="gen_pdf_btn_v3"):
        try:
            from ceph_reporter import generate_unified_report

            with st.spinner("در حال تولید گزارش..."):
                ceph_results["intraoral_3d"] = {
                    "bolton": st.session_state.get("bolton_3d", {}),
                    "bolton_auto": st.session_state.get("bolton_auto", {}),
                    "clicks_max": st.session_state.get("clicks_max", []),
                    "clicks_man": st.session_state.get("clicks_man", []),
                }
                pdf_bytes = generate_unified_report(ceph_results)

            st.download_button(
                label="📥 دانلود گزارش یکپارچه (PDF)",
                data=pdf_bytes,
                file_name=f"Aariz_Unified_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="download_unified_pdf_v3"
            )
            st.success("✅ گزارش آماده دانلود است.")
        except ImportError:
            st.error("❌ ماژول `ceph_reporter.py` یافت نشد.")
        except Exception as e:
            st.error(f"❌ خطا در تولید گزارش: {type(e).__name__}: {e}")
            import traceback
            st.code(traceback.format_exc())
else:
    st.divider()
    st.header("📄 گزارش نهایی یکپارچه")
    st.info("برای فعال شدن این بخش، فایل JSON سفالومتری را در **سایدبار چپ** آپلود کنید.")
