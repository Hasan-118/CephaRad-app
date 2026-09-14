def render_tooth_measurement_tab(mesh_max=None, mesh_man=None, pixel_size_default=0.1):
    """رابط کاربری کامل اندازه‌گیری نقطه‌به‌نقطه"""

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
        key="measurement_arch_v3"
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
            key=f"px_size_{key}_v3"
        )

    # --- کش کردن تصویر اکلوزال در session_state ---
    img_cache_key = f"occlusal_img_{key}"
    if img_cache_key not in st.session_state:
        with st.spinner("در حال رندر نمای اکلوزال..."):
            occ_img, transform_info = render_occlusal_view(mesh, img_size=900)
            st.session_state[img_cache_key] = (occ_img, transform_info)
    else:
        occ_img, transform_info = st.session_state[img_cache_key]

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
                    key=f"missing_{key}_v3_{tooth['id']}"
                )
                if checkbox and tooth["id"] not in missing_teeth:
                    missing_teeth.add(tooth["id"])
                    st.rerun()
                elif not checkbox and tooth["id"] in missing_teeth:
                    missing_teeth.discard(tooth["id"])
                    st.rerun()

        st.info(f"📊 غایب: **{len(missing_teeth)}** | موجود: **{len(teeth) - len(missing_teeth)}**")

    available_teeth = [t for t in teeth if t["id"] not in missing_teeth]
    if not available_teeth:
        st.warning("⚠️ همه دندان‌ها غایب هستند.")
        return None

    # --- بررسی اتمام فک ---
    tooth_points = st.session_state[f"tooth_points_{key}"]
    completed_count = sum(
        1 for t in available_teeth
        if t["id"] in tooth_points
        and "mesial" in tooth_points[t["id"]]
        and "distal" in tooth_points[t["id"]]
    )
    total_count = len(available_teeth)
    is_completed = (completed_count == total_count)

    # --- اگر کامل شده ---
    if is_completed:
        st.success(f"✅ فک {'بالا' if is_maxilla else 'پایین'} کامل شد! ({total_count} / {total_count})")

        widths = compute_tooth_widths(tooth_points, teeth, missing_teeth, pixel_size_mm)
        total_width = sum(w["width_mm"] for w in widths if w["width_mm"] is not None)
        st.metric("مجموع عرض دندان‌های علامت‌گذاری‌شده", f"{round(total_width, 2)} mm")

        # نمایش جدول
        import pandas as pd
        table_data = []
        for w in widths:
            width_str = f"{w['width_mm']} mm" if w["width_mm"] else "—"
            space_str = f"{w['space_before_mm']} mm" if w["space_before_mm"] is not None else "—"
            table_data.append({
                "دندان": f"{w['tooth_name']} ({w['tooth_id']})",
                "نوع": w["type"],
                "عرض (mm)": width_str,
                "فاصله با قبلی (mm)": space_str,
                "وضعیت": w["status"]
            })
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.session_state[f"measured_widths_{key}"] = widths

        # دکمه برای شروع مجدد (Reset)
        if st.button("🔄 شروع مجدد این فک", key=f"reset_arch_{key}"):
            for t in available_teeth:
                if t["id"] in tooth_points:
                    del tooth_points[t["id"]]
            st.session_state[f"current_tooth_idx_{key}"] = 0
            st.session_state.pop(f"occlusal_img_{key}", None)
            st.rerun()

        return widths

    # --- ادامه اندازه‌گیری ---
    current_idx = min(st.session_state[f"current_tooth_idx_{key}"], len(available_teeth) - 1)
    current_tooth = available_teeth[current_idx]

    # تعیین نقطه فعلی
    if current_tooth["id"] not in tooth_points:
        expected_type = get_expected_point_type(current_tooth, is_maxilla)
    else:
        pts = tooth_points[current_tooth["id"]]
        if "distal" not in pts:
            expected_type = "distal"
        elif "mesial" not in pts:
            expected_type = "mesial"
        else:
            # هر دو نقطه ثبت شده → برو به دندان بعدی
            if current_idx < len(available_teeth) - 1:
                st.session_state[f"current_tooth_idx_{key}"] = current_idx + 1
                st.rerun()
            else:
                # آخرین دندان کامل شد
                st.rerun()
            expected_type = "distal"

    if f"override_point_type_{key}" in st.session_state:
        expected_type = st.session_state.pop(f"override_point_type_{key}")

    current_type = expected_type

    # نمایش وضعیت
    st.markdown("### 🎯 علامت‌گذاری")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("دندان فعلی", f"{current_tooth['name']}")
    with col2:
        side_label = "🟢 سمت راست" if current_tooth["side"] == "right" else "🔵 سمت چپ"
        st.metric("سمت", side_label)
    with col3:
        type_label = "🔵 دیستال" if current_type == "distal" else "🔴 مزیال"
        st.metric("نقطه فعلی", type_label)
    with col4:
        st.metric("پیشرفت", f"{completed_count} / {total_count}")

    # دکمه‌های ناوبری
    col_nav1, col_nav2, col_nav3, col_nav4, col_nav5 = st.columns(5)
    with col_nav1:
        if st.button("◀ قبلی", use_container_width=True, key=f"prev_{key}_v3"):
            if current_idx > 0:
                st.session_state[f"current_tooth_idx_{key}"] = current_idx - 1
                st.session_state.pop(f"override_point_type_{key}", None)
                st.rerun()
            else:
                st.toast("این اولین دندان است.", icon="⚠️")

    with col_nav2:
        if st.button("🔄 پاک کردن این دندان", use_container_width=True, key=f"clear_{key}_v3"):
            tid = current_tooth["id"]
            if tid in tooth_points:
                del tooth_points[tid]
                st.toast(f"نقاط دندان {current_tooth['name']} پاک شد.", icon="✅")
            st.session_state.pop(f"override_point_type_{key}", None)
            st.rerun()

    with col_nav3:
        if st.button("🔵 دیستال", use_container_width=True, key=f"set_d_{key}_v3"):
            st.session_state[f"override_point_type_{key}"] = "distal"
            st.rerun()

    with col_nav4:
        if st.button("🔴 مزیال", use_container_width=True, key=f"set_m_{key}_v3"):
            st.session_state[f"override_point_type_{key}"] = "mesial"
            st.rerun()

    with col_nav5:
        if st.button("⏭ بعدی", use_container_width=True, key=f"next_{key}_v3"):
            if current_idx < len(available_teeth) - 1:
                st.session_state[f"current_tooth_idx_{key}"] = current_idx + 1
                st.session_state.pop(f"override_point_type_{key}", None)
                st.rerun()
            else:
                st.toast("این آخرین دندان است.", icon="⚠️")

    # رسم و نمایش
    img_with_points = draw_points_on_image(
        occ_img, tooth_points, teeth, missing_teeth,
        current_tooth_id=current_tooth["id"],
        current_point_type=current_type,
        is_maxilla=is_maxilla
    )

    st.markdown(f"**👆 کلیک کنید تا نقطه {current_type} دندان {current_tooth['name']} ثبت شود:**")

    clicked = streamlit_image_coordinates(
        img_with_points,
        key=f"occlusal_click_{key}_{current_tooth['id']}_{current_type}_v3"
    )

    if clicked:
        cx, cy = clicked["x"], clicked["y"]
        tid = current_tooth["id"]

        if tid not in tooth_points:
            tooth_points[tid] = {}

        tooth_points[tid][current_type] = (cx, cy)

        next_type = get_next_point_type(current_type, current_tooth)

        if next_type is not None:
            st.session_state[f"override_point_type_{key}"] = next_type
        else:
            # آخرین نقطه این دندان ثبت شد
            if current_idx < len(available_teeth) - 1:
                st.session_state[f"current_tooth_idx_{key}"] = current_idx + 1
                st.session_state.pop(f"override_point_type_{key}", None)
            else:
                # این آخرین دندان بود → فک کامل شد
                st.session_state.pop(f"override_point_type_{key}", None)

        st.rerun()

    # نمایش نتایج (اگر نقاطی ثبت شده)
    st.divider()
    st.markdown("### 📊 نتایج اندازه‌گیری")

    if len(tooth_points) == 0:
        st.info("ℹ️ هنوز هیچ دندانی علامت‌گذاری نشده است.")
        return None

    widths = compute_tooth_widths(tooth_points, teeth, missing_teeth, pixel_size_mm)

    import pandas as pd
    table_data = []
    for w in widths:
        width_str = f"{w['width_mm']} mm" if w["width_mm"] else "—"
        space_str = f"{w['space_before_mm']} mm" if w["space_before_mm"] is not None else "—"
        table_data.append({
            "دندان": f"{w['tooth_name']} ({w['tooth_id']})",
            "نوع": w["type"],
            "عرض (mm)": width_str,
            "فاصله با قبلی (mm)": space_str,
            "وضعیت": w["status"]
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    total_width = sum(w["width_mm"] for w in widths if w["width_mm"] is not None)
    st.metric("مجموع عرض دندان‌های علامت‌گذاری‌شده", f"{round(total_width, 2)} mm")

    st.session_state[f"measured_widths_{key}"] = widths

    return widths
