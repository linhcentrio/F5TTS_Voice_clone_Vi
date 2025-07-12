# voice_clone.py - Phiên bản siêu tối ưu 2.0
import os
import sys
import warnings
from pathlib import Path
import streamlit as st

# Bỏ qua tất cả cảnh báo không cần thiết
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Torchaudio's I/O functions")
warnings.filterwarnings("ignore", message="local_dir_use_symlinks")

# Tắt tính năng theo dõi của Streamlit
os.environ.update({
    'STREAMLIT_SERVER_RUN_ON_SAVE': 'false',
    "STREAMLIT_WATCHER_WARNING_DISABLED": "true",
    "STREAMLIT_SERVER_WATCHER_TYPE": "none",
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"
})

# Khắc phục event loop Windows
if sys.platform.startswith('win'):
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass
    
    # Monkey patch để ngăn lỗi torch._classes.__path__._path
    try:
        class PathProtector:
            def __getattr__(self, name):
                if name == '_path':
                    return []
                raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
                
        import types
        import torch
        if hasattr(torch, '_classes') and hasattr(torch._classes, '__path__'):
            torch._classes.__path__ = PathProtector()
    except:
        pass

# Đường dẫn
DIRS = {d: Path(f"./{d}").resolve() for d in ["model", "output_audio", "audio_ref", "temp_audio"]}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# Import các module
from utils.text_utils import setup_ttsnorm
from utils.audio_utils import generate_filename, save_audio
from utils.model_utils import load_tts_model, preload_vocoder
from components.ui import audio_source_selector, generation_parameters, audio_player, model_config_sidebar

# Thiết lập TTSnorm
TTSnorm = setup_ttsnorm()

# Cấu hình UI
st.set_page_config(page_title="F5TTS Voice Clone", page_icon="🎤", layout="wide")
st.title("F5TTS - Nhân Bản Giọng Nói")

# Tải vocoder cho Windows
if sys.platform.startswith('win'):
    vocos_path = preload_vocoder(DIRS["model"])
else:
    vocos_path = None

# Sidebar
model_config = model_config_sidebar(DIRS["model"])

# Tabs chính
tab1, tab2 = st.tabs(["Tạo Giọng Nói", "Cài Đặt Nâng Cao"])

with tab1:
    # Layout 2 cột
    col1, col2 = st.columns(2)
    
    with col1:
        # Chọn âm thanh tham chiếu
        _, ref_audio_path = audio_source_selector(DIRS["audio_ref"], DIRS["temp_audio"])
        
        # Văn bản tham chiếu
        st.subheader("Văn Bản Tham Chiếu")
        ref_text = st.text_area("Văn bản tham chiếu", height=100)
        clip_short = st.checkbox("Cắt âm thanh dài", value=True)
    
    with col2:
        # Văn bản đầu vào
        st.header("Văn Bản Cần Tạo")
        input_text = st.text_area("Văn bản cần chuyển thành giọng nói", height=200)
        
        # Tệp đầu ra
        custom_filename = st.text_input("Tên tệp tùy chỉnh (để trống = tự động)", value="")
        
        # Nút tạo
        col1, col2 = st.columns(2)
        with col1:
            generate_button = st.button("🔊 Tạo Giọng Nói", type="primary", use_container_width=True)
        with col2:
            quick_button = st.button("🚀 Tạo Nhanh", use_container_width=True)

with tab2:
    # Các tham số nâng cao
    params = generation_parameters()

# Container kết quả
result_container = st.container()

# Tải mô hình
@st.cache_resource
def get_model():
    return load_tts_model(
        model_config["model_path"], 
        model_config["vocab_file"], 
        model_config["vocoder_name"], 
        model_config["use_ema"],
        vocos_path
    )

tts_model, error = get_model()
model_loaded = tts_model is not None

if error:
    st.sidebar.error(error)
elif model_loaded:
    st.sidebar.success("✅ Mô hình đã được khởi tạo!")

# Xử lý tạo giọng nói
def process_generation(input_text, ref_audio_path=None, ref_text=None, quick_mode=False):
    if not model_loaded:
        st.error("Mô hình chưa được tải")
        return
    
    if not input_text:
        st.error("Vui lòng nhập văn bản")
        return
    
    # Chuẩn hóa văn bản
    text_norm = TTSnorm(input_text)
    
    # Tên tệp đầu ra
    output_filename = custom_filename if custom_filename else generate_filename(input_text, params["nfe_step"])
    output_path = str(DIRS["output_audio"] / output_filename)
    
    # Xử lý tham chiếu (nếu không phải chế độ nhanh)
    if not quick_mode and ref_audio_path:
        with st.spinner("Đang xử lý âm thanh tham chiếu..."):
            tts_model.preprocess_reference(
                ref_audio_path=ref_audio_path,
                ref_text=ref_text,
                clip_short=clip_short
            )
            
            # Hiển thị thời lượng
            ref_duration = tts_model.get_current_audio_length()
            st.info(f"Thời lượng tham chiếu: {ref_duration:.2f} giây")
    
    # Tạo giọng nói
    with st.spinner("Đang tạo giọng nói..."):
        import time
        start_time = time.time()
        
        # Tham số
        generate_params = {
            "text": text_norm,
            "output_path": output_path,
            "nfe_step": params["nfe_step"],
            "cfg_strength": params["cfg_strength"],
            "speed": params["speed"],
            "cross_fade_duration": params["cross_fade_duration"],
            "return_numpy": True
        }
        
        if params["show_spectrogram"]:
            generate_params["return_spectrogram"] = True
            result = tts_model.generate(**generate_params)
            audio_array, sample_rate, spectrogram = result
            
            # Hiển thị spectrogram
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(spectrogram, origin="lower", aspect="auto")
            ax.set_title("Spectrogram")
            st.pyplot(fig)
        else:
            audio_array, sample_rate = tts_model.generate(**generate_params)
        
        generation_time = time.time() - start_time
    
    # Hiển thị kết quả
    audio_player(audio_array, sample_rate, output_filename, generation_time)
    
    # Lưu tệp
    save_audio(audio_array, sample_rate, output_path)

# Xử lý nút tạo
if generate_button and ref_audio_path:
    with result_container:
        process_generation(input_text, ref_audio_path, ref_text, quick_mode=False)
elif quick_button and input_text:
    with result_container:
        if not hasattr(tts_model, 'ref_audio_processed') or tts_model.ref_audio_processed is None:
            st.error("Chưa có tham chiếu. Vui lòng sử dụng 'Tạo Giọng Nói' trước.")
        else:
            process_generation(input_text, quick_mode=True)
elif generate_button and not ref_audio_path:
    st.error("Vui lòng chọn âm thanh tham chiếu")

# Giải quyết vấn đề với torch._classes.__path__._path trong Streamlit
def patch_streamlit_watcher():
    """Patch cho vấn đề Streamlit với torch._classes.__path__._path"""
    try:
        import streamlit.watcher.local_sources_watcher as watcher
        
        # Lưu hàm extract_paths gốc
        original_extract_paths = watcher.extract_paths
        
        # Tạo phiên bản an toàn của extract_paths
        def safe_extract_paths(module):
            try:
                # Bỏ qua các module torch để tránh lỗi
                if module.__name__.startswith('torch'):
                    return []
                return original_extract_paths(module)
            except Exception:
                return []
        
        # Thay thế hàm extract_paths
        watcher.extract_paths = safe_extract_paths
    except:
        pass
