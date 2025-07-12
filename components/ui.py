# components/ui.py
import streamlit as st
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List

def on_audio_recorded():
    """Callback khi ghi âm hoàn tất."""
    if st.session_state.recorder_output:
        st.session_state.audio_recorded = True

def audio_source_selector(audio_ref_dir: Path, temp_dir: Path) -> Tuple[str, Optional[str]]:
    """Component để chọn nguồn âm thanh tham chiếu."""
    from utils.audio_utils import handle_audio_input, handle_recorded_audio
    from utils.model_utils import get_reference_audio_files
    
    st.header("Âm Thanh Tham Chiếu")
    
    # Chọn nguồn âm thanh
    audio_source = st.radio(
        "Nguồn âm thanh tham chiếu",
        options=["Tải tệp lên", "Ghi âm trực tiếp", "Chọn từ thư mục"],
        horizontal=True
    )
    
    ref_audio_path = None
    
    if audio_source == "Tải tệp lên":
        uploaded_file = st.file_uploader(
            "Tải lên âm thanh tham chiếu", 
            type=["wav", "mp3", "flac", "ogg", "m4a"]
        )
        if uploaded_file:
            ref_audio_path = handle_audio_input(uploaded_file, temp_dir)
    
    elif audio_source == "Ghi âm trực tiếp":
        try:
            from streamlit_mic_recorder import mic_recorder
            
            # Khởi tạo session state
            if "audio_recorded" not in st.session_state:
                st.session_state.audio_recorded = False
                
            st.info("**Ghi âm tham chiếu** (khuyến nghị: 5-10 giây)")
            
            # Giao diện ghi âm
            audio_data = mic_recorder(
                start_prompt="⏺️ Bắt đầu ghi âm",
                stop_prompt="⏹️ Dừng ghi âm",
                key="recorder",
                callback=on_audio_recorded
            )
            
            # Xử lý khi ghi âm thành công
            if st.session_state.get("audio_recorded"):
                recorder_output = st.session_state.get("recorder_output")
                
                if recorder_output and "bytes" in recorder_output:
                    st.success(f"✅ Đã ghi âm thành công! (tỷ lệ mẫu: {recorder_output.get('sample_rate', 'N/A')}Hz)")
                    
                    # Lưu âm thanh và lấy đường dẫn
                    ref_audio_path = handle_recorded_audio(recorder_output, temp_dir)
                    
                    # Nút quản lý âm thanh
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("💾 Lưu vào thư mục"):
                            try:
                                timestamp = time.strftime("%Y%m%d-%H%M%S")
                                save_path = audio_ref_dir / f"recording_{timestamp}.wav"
                                
                                with open(ref_audio_path, "rb") as f_in:
                                    with open(save_path, "wb") as f_out:
                                        f_out.write(f_in.read())
                                st.success(f"Đã lưu âm thanh vào {save_path}")
                            except Exception as e:
                                st.error(f"Lỗi lưu âm thanh: {e}")
                    
                    with col2:
                        if st.button("🗑️ Xóa ghi âm"):
                            try:
                                if "recorder_output" in st.session_state:
                                    del st.session_state.recorder_output
                                st.session_state.audio_recorded = False
                                if ref_audio_path and Path(ref_audio_path).exists():
                                    Path(ref_audio_path).unlink(missing_ok=True)
                                ref_audio_path = None
                                st.rerun()
                            except Exception as e:
                                st.error(f"Lỗi xóa ghi âm: {e}")
        except ImportError:
            st.error("Thư viện streamlit-mic-recorder chưa được cài đặt")
            if st.button("Cài đặt streamlit-mic-recorder"):
                import subprocess, sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit-mic-recorder"])
                st.success("Đã cài đặt thư viện. Vui lòng khởi động lại ứng dụng.")
                st.rerun()
    else:
        # Chọn từ thư mục
        reference_files = get_reference_audio_files(audio_ref_dir)
        if not reference_files:
            st.warning("Không tìm thấy tệp âm thanh trong thư mục ./audio_ref")
        else:
            selected_audio = st.selectbox(
                "Chọn âm thanh tham chiếu",
                options=reference_files
            )
            if selected_audio:
                ref_audio_path = str(audio_ref_dir / selected_audio)
                with open(ref_audio_path, "rb") as file:
                    st.audio(file, format=f"audio/{selected_audio.split('.')[-1]}")
    
    return audio_source, ref_audio_path

def model_config_sidebar(model_dir: Path) -> Dict[str, Any]:
    """Component cấu hình mô hình trong sidebar."""
    from utils.model_utils import get_available_models
    
    st.sidebar.header("Cấu hình mô hình")
    
    # Trạng thái thiết bị
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.info(f"Thiết bị: {device.upper()}")
    
    if device == "cuda":
        st.sidebar.success(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    
    # Chọn mô hình
    available_models = get_available_models(model_dir)
    selected_model = st.sidebar.selectbox("Chọn mô hình", options=available_models)
    model_path = str(model_dir / selected_model)
    
    # Cấu hình thêm
    vocab_file = st.sidebar.text_input("Tệp từ vựng", value=str(model_dir / "vocab.txt"))
    vocoder_name = st.sidebar.selectbox("Bộ tạo giọng nói", options=["vocos", "bigvgan"])
    use_ema = st.sidebar.checkbox("Sử dụng trọng số EMA", value=True)
    
    # Cache control
    st.sidebar.divider()
    
    # Nút làm mới
    if st.sidebar.button("🔄 Làm mới"):
        st.cache_data.clear()
        st.rerun()
        
    # Nút xóa cache
    if st.sidebar.button("Xóa bộ nhớ đệm"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.sidebar.success("Đã xóa bộ nhớ đệm")
        st.rerun()
    
    return {
        "model_path": model_path,
        "vocab_file": vocab_file,
        "vocoder_name": vocoder_name,
        "use_ema": use_ema,
        "device": device
    }

def generation_parameters() -> Dict[str, Any]:
    """Component các tham số tạo giọng nói."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Tham Số Tạo Giọng Nói")
        nfe_step = st.slider(
            "Số bước khử nhiễu",
            min_value=8,
            max_value=50,
            value=20,
            help="Giá trị cao hơn = chất lượng tốt hơn nhưng chậm hơn"
        )
        
        cfg_strength = st.slider(
            "Độ mạnh hướng dẫn",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.1,
            help="Kiểm soát độ trung thành với văn bản"
        )
        
        speed = st.slider(
            "Tốc độ giọng nói",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Điều chỉnh tốc độ nói"
        )
    
    with col2:
        st.header("Xử Lý Âm Thanh")
        cross_fade_duration = st.slider(
            "Thời gian cross-fade",
            min_value=0.0,
            max_value=0.5,
            value=0.15,
            step=0.01,
            help="Thời gian chuyển tiếp giữa các phân đoạn"
        )
        
        show_spectrogram = st.checkbox(
            "Hiển thị spectrogram",
            value=True,
            help="Hiển thị biểu diễn hình ảnh của âm thanh"
        )
        
        use_cache = st.checkbox(
            "Lưu kết quả vào bộ nhớ đệm",
            value=True,
            help="Tăng tốc khi tạo lại cùng một văn bản"
        )
    
    return {
        "nfe_step": nfe_step,
        "cfg_strength": cfg_strength,
        "speed": speed,
        "cross_fade_duration": cross_fade_duration,
        "show_spectrogram": show_spectrogram,
        "use_cache": use_cache
    }

def audio_player(audio_array, sample_rate, output_filename, generation_time=None):
    """Component phát âm thanh với tải xuống."""
    from utils.audio_utils import play_audio_from_array
    
    st.subheader("Âm Thanh Đã Tạo")
    
    # Phát âm thanh
    audio_buffer = play_audio_from_array(audio_array, sample_rate)
    
    # Thông báo thành công
    if generation_time:
        st.success(f"✅ Tạo giọng nói thành công trong {generation_time:.2f} giây!")
    else:
        st.success("✅ Tạo giọng nói thành công!")
    
    # Nút tải xuống
    if audio_buffer:
        st.download_button(
            label="Tải Xuống",
            data=audio_buffer,
            file_name=output_filename,
            mime="audio/wav"
        )
    
    return audio_buffer
