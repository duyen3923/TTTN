import streamlit as st
import os
import time
from prediction import predict_video

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

_, _, _, col_notify = st.columns([1, 1, 1, 1])
notify_placeholder = col_notify.empty()

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

st.set_page_config(page_title="Safety construction", layout="wide")

st.title("An toàn lao động")

uploaded_file = st.file_uploader("", type=["mp4"])

col1, col2 = st.columns(2)

if uploaded_file is not None:
    video_input_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(video_input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    notify_placeholder.success("Video đã được tải lên thành công!")
    time.sleep(5)
    notify_placeholder.empty()

    with col1:
        st.subheader("Video Gốc")
        st.video(video_input_path)


    if st.button("Dự đoán"):
        with st.spinner("Đang xử lý video..."):
            video_output_path = os.path.join(OUTPUT_DIR, "predicted_" + uploaded_file.name)

        # Gọi hàm dự đoán và nhận alert_info
            alert_info = predict_video(video_input_path, video_output_path)

            notify_placeholder.success("Dự đoán thành công!")
            time.sleep(5)
            notify_placeholder.empty()
        
    # Hiển thị video kết quả
        with col2:
            st.subheader("Kết quả Dự đoán")
            st.video(video_output_path)

        st.markdown("## ⚠️ Thống kê vi phạm an toàn")

        if alert_info:
            total_violations = sum(alert["num_no_helmet"] for alert in alert_info)
            st.warning(f"🔴 Tổng số lượt vi phạm được phát hiện: {total_violations}")
    
            with st.expander("📋 Xem bảng thống kê chi tiết"):
                st.dataframe(alert_info, use_container_width=True)

        st.markdown("""
        <style>
        .alert-toast {
            position: fixed;
            right: 20px;
            background-color: #dc3545; /* đỏ cảnh báo */
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 15px;
            z-index: 9999;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.25);
            opacity: 1;
            animation: fadeOut 1s ease-in-out forwards;
            animation-delay: 5s;
        }

        @keyframes fadeOut {
            from {opacity: 1; transform: translateY(0);}
            to {opacity: 0; transform: translateY(10px);}
        }
        </style>
    """, unsafe_allow_html=True)
        for i, alert in enumerate(alert_info[-5:]):  
            alert_text = f"{alert['num_no_helmet']} người không đội mũ tại {alert['timestamp']} giây (frame {alert['frame_number']})"
            st.markdown(f"""
            <div class="alert-toast" style="bottom: {20 + i * 65}px;">
                {alert_text}
            </div>
        """, unsafe_allow_html=True)




