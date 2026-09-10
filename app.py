import streamlit as st

# ۱. تنظیمات اولیه صفحه (در صورت وجود)
st.set_page_config(
    page_title="Aariz Precision Station",
    layout="wide"
)

# ۲. تزریق CSS برای اصلاح جهت متون فارسی و انگلیسی (RTL)
st.markdown("""
    <style>
    /* تنظیم جهت کل صفحه و فونت‌ها */
    .stApp {
        direction: rtl;
        text-align: right;
    }
    
    /* تنظیم تمامی عناوین، پاراگراف‌ها، لیبل‌ها و فرم‌ها */
    h1, h2, h3, h4, h5, h6, p, div, label, span, .stMarkdown {
        direction: rtl;
        text-align: right;
    }

    /* استثنا: نگه‌داشتن نمایش کدها و فرمول‌ها به صورت چپ‌به‌راست */
    code, pre, .stCodeBlock {
        direction: ltr !important;
        text-align: left !important;
    }
    </style>
""", unsafe_allow_html=True)

# ۳. ادامه کدهای اصلی برنامه شما...
