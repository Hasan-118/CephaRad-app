"""
Aariz 3D Analysis Station
اپلیکیشن تحلیل سه‌بعدی اسکن داخل دهانی + ادغام با سفالومتری
نسخه: 2.0 - با اندازه‌گیری نقطه‌به‌نقطه
"""

import streamlit as st
import json
from datetime import datetime

# --- تنظیمات صفحه ---
st.set_page_config(
    page_title="Aariz 3D Analysis Station",
    layout="wide",
    page_icon="🦷"
)

# --- بارگذاری ماژول ۳D ---
try:
    from intraoral_3d_module import render_intraoral_3d_tab
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

# --- عنوان ---
st.title("🦷 ایستگاه تحلیل سه‌بعدی Aariz")
st.caption("Aariz 3D Analysis Station - Intraoral Scan")

# --- بارگذاری JSON سفالومتری در سایدبار ---
st.sidebar.header("📂 ادغام با تحلیل سفالومتری")
st.sidebar.markdown("""
اگر قبلاً تحلیل سفالومتری (۲D) را انجام داده‌اید،
فایل JSON دانلود شده را اینجا آپلود کنید تا گزارش نهایی یکپارچه شود.
""")

uploaded_json = st.sidebar.file_uploader(
    "آپلود نتایج سفالومتری (JSON):",
    type=['json'],
    key="ceph_json_upload"
)

ceph_results = None
if uploaded_json is not None:
    try:
        ceph_results = json.load(uploaded_json)

        if 'version' not in ceph_results or 'measurements' not in ceph_results:
            st.sidebar.error("❌ فایل JSON نامعتبر است.")
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

# --- نمایش وضعیت در تب اصلی ---
if ceph_results:
    st.success("🔗 **حالت یکپارچه فعال** — گزارش نهایی شامل هر دو تحلیل ۲D و ۳D خواهد بود.")
else:
    st.info("ℹ️ **حالت مستقل** — فقط تحلیل سه‌بعدی انجام می‌شود. برای ادغام، فایل JSON سفالومتری را در سایدبار آپلود کنید.")

st.divider()

# ============================================================
# بخش ۱: ماژول اصلی ۳D (نمای سه‌بعدی + کلیک اکلوزال + Bolton دستی)
# ============================================================
with st.spinner("در حال بارگذاری ماژول تحلیل سه‌بعدی..."):
    try:
        render_intraoral_3d_tab()
    except Exception as e:
        st.error(f"❌ خطا در اجرای ماژول ۳D: {type(e).__name__}: {e}")
        import traceback
        st.code(traceback.format_exc())

# ============================================================
# بخش ۲: اندازه‌گیری نقطه‌به‌نقطه با ترتیب هوشمند
# ============================================================
st.divider()

mesh_max = st.session_state.get("uploaded_mesh_max", None)
mesh_man = st.session_state.get("uploaded_mesh_man", None)

if MEASUREMENT_AVAILABLE:
    if mesh_max is not None or mesh_man is not None:
        st.header("🎯 اندازه‌گیری نقطه‌به‌نقطه (ترتیب هوشمند)")
        st.caption("""
        این بخش روی **نمای اکلوزال** (از بالا) کار می‌کند. برای هر دندان:
        - **دندان‌های سمت راست:** ابتدا **دیستال** (🔵)، سپس **مزیال** (🔴)
        - **دندان‌های سمت چپ:** ابتدا **مزیال** (🔴)، سپس **دیستال** (🔵)
        
        این ترتیب، یک مسیر پیوسته از آخرین دندان راست تا آخرین دندان چپ ایجاد می‌کند.
        """)

        try:
            widths_max = render_tooth_measurement_tab(mesh_max, mesh_man)

            # --- Bolton خودکار بر اساس اندازه‌گیری واقعی ---
            widths_man_stored = st.session_state.get("measured_widths_man", None)
            widths_max_stored = st.session_state.get("measured_widths_max", None)

            if widths_man_stored is not None and widths_max_stored is not None:
                st.divider()
                st.subheader("🔢 نسبت‌های بولتون (بر اساس اندازه‌گیری واقعی نقطه‌به‌نقطه)")

                bolton = compute_bolton_summary(widths_max_stored, widths_man_stored)

                col1, col2 = st.columns(2)
                with col1:
                    diff = round(bolton["overall_ratio"] - 91.3, 2)
                    st.metric(
                        "Overall Bolton",
                        f"{bolton['overall_ratio']}%",
                        f"{diff}%",
                        help="نرمال: 91.3%"
                    )
                    if bolton["overall_ratio"] > 92.5:
                        st.warning("⚠️ اضافه حجم دندانی در فک پایین")
                    elif 0 < bolton["overall_ratio"] < 90.0:
                        st.info("ℹ️ اضافه حجم دندانی در فک بالا")
                    elif bolton["overall_ratio"] > 0:
                        st.success("✅ نسبت کلی متوازن است")

                with col2:
                    diff_ant = round(bolton["anterior_ratio"] - 77.2, 2)
                    st.metric(
                        "Anterior Bolton",
                        f"{bolton['anterior_ratio']}%",
                        f"{diff_ant}%",
                        help="نرمال: 77.2%"
                    )
                    if bolton["anterior_ratio"] > 78.5:
                        st.warning("⚠️ اضافه حجم دندان‌های قدامی فک پایین")
                    elif 0 < bolton["anterior_ratio"] < 75.5:
                        st.info("ℹ️ اضافه حجم دندان‌های قدامی فک بالا")
                    elif bolton["anterior_ratio"] > 0:
                        st.success("✅ نسبت قدامی متوازن است")

                st.caption(
                    f"📊 تعداد دندان‌های اندازه‌گیری‌شده: "
                    f"فک بالا {bolton['max_count']} | فک پایین {bolton['man_count']}"
                )

                # ذخیره در session_state برای استفاده در PDF
                st.session_state["bolton_auto"] = bolton

        except Exception as e:
            st.error(f"❌ خطا در بخش اندازه‌گیری: {type(e).__name__}: {e}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.header("🎯 اندازه‌گیری نقطه‌به‌نقطه (ترتیب هوشمند)")
        st.info("""
        برای فعال شدن این بخش، ابتدا باید اسکن‌های STL را در بخش بالا آپلود کنید.
        پس از آپلود، نمای اکلوزال و ابزار اندازه‌گیری نقطه‌به‌نقطه ظاهر می‌شود.
        """)
else:
    st.header("🎯 اندازه‌گیری نقطه‌به‌نقطه (ترتیب هوشمند)")
    st.error(f"❌ ماژول `tooth_measurement.py` یافت نشد یا خطا داد: {MEASUREMENT_ERROR if 'MEASUREMENT_ERROR' in dir() else 'نامشخص'}")
    st.info("""
    برای رفع این مشکل، فایل `tooth_measurement.py` را در کنار `app_3d.py` قرار دهید
    و مطمئن شوید که کد آن کامل و بدون خطا باشد.
    """)

# ============================================================
# بخش ۳: گزارش یکپارچه PDF
# ============================================================
if ceph_results:
    st.divider()
    st.header("📄 گزارش نهایی یکپارچه")
    st.markdown("""
    در این بخش می‌توانید **گزارش PDF یکپارچه** شامل هر دو تحلیل سفالومتری (۲D)
    و اسکن داخل دهانی (۳D) را دانلود کنید.
    """)

    if st.button("🖨 تولید گزارش یکپارچه PDF", use_container_width=True, key="gen_pdf_btn"):
        try:
            from ceph_reporter import generate_unified_report

            with st.spinner("در حال تولید گزارش..."):
                # ادغام نتایج ۳D در ceph_results
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
                key="download_unified_pdf"
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
    st.info("""
    برای فعال شدن این بخش، فایل JSON سفالومتری را در **سایدبار چپ** آپلود کنید.
    پس از آپلود، دکمه تولید گزارش یکپارچه ظاهر می‌شود.
    """)
