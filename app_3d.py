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

# --- فراخوانی تب ۳D ---
with st.spinner("در حال بارگذاری ماژول تحلیل سه‌بعدی..."):
    try:
        render_intraoral_3d_tab()
    except Exception as e:
        st.error(f"❌ خطا در اجرای ماژول ۳D: {type(e).__name__}: {e}")
        import traceback
        st.code(traceback.format_exc())

# --- بخش پایانی: دانلود گزارش یکپارچه ---
if ceph_results:
    st.divider()
    st.header("📄 گزارش نهایی یکپارچه")
    st.markdown("""
    در این بخش می‌توانید **گزارش PDF یکپارچه** شامل هر دو تحلیل سفالومتری (۲D)
    و اسکن داخل دهانی (۳D) را دانلود کنید.
    """)

    if st.button("🖨 تولید گزارش یکپارچه PDF", use_container_width=True):
        try:
            from ceph_reporter import generate_unified_report

            with st.spinner("در حال تولید گزارش..."):
                pdf_bytes = generate_unified_report(ceph_results)

            st.download_button(
                label="📥 دانلود گزارش یکپارچه (PDF)",
                data=pdf_bytes,
                file_name=f"Aariz_Unified_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            st.success("✅ گزارش آماده دانلود است.")
        except ImportError:
            st.error("❌ ماژول `ceph_reporter.py` یافت نشد.")
        except Exception as e:
            st.error(f"❌ خطا در تولید گزارش: {type(e).__name__}: {e}")
            import traceback
            st.code(traceback.format_exc())
