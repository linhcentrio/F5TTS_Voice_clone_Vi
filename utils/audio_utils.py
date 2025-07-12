# utils/audio_utils.py
import io
import time
import hashlib
import soundfile as sf
import streamlit as st
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Optional, Tuple, Union, BinaryIO

def handle_audio_input(uploaded_file, temp_dir: Path) -> Optional[str]:
    """Xử lý tệp âm thanh đã tải lên."""
    if not uploaded_file:
        return None
    
    try:
        # Lưu vào thư mục tạm
        temp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = temp_dir / f"upload_{time.strftime('%Y%m%d%H%M%S')}_{uploaded_file.name}"
        
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Hiển thị âm thanh
        try:
            st.audio(uploaded_file)
        except Exception as e:
            st.warning(f"Không thể hiển thị âm thanh (nhưng đã lưu thành công): {e}")
        
        return str(tmp_path)
    except Exception as e:
        st.error(f"Lỗi xử lý tệp âm thanh: {e}")
        return None

def handle_recorded_audio(recorder_output, temp_dir: Path) -> Optional[str]:
    """Xử lý âm thanh được ghi trực tiếp."""
    if not recorder_output or "bytes" not in recorder_output:
        return None
        
    try:
        # Lưu âm thanh tạm
        temp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = temp_dir / f"recording_{time.strftime('%Y%m%d%H%M%S')}.wav"
        
        with open(tmp_path, "wb") as f:
            f.write(recorder_output['bytes'])
            
        # Hiển thị âm thanh
        try:
            st.audio(recorder_output['bytes'], format="audio/wav")
        except Exception as e:
            st.warning(f"Không thể hiển thị âm thanh (nhưng đã lưu thành công): {e}")
            
        return str(tmp_path)
    except Exception as e:
        st.error(f"Lỗi lưu âm thanh ghi: {e}")
        return None

def play_audio_from_array(audio_array: np.ndarray, sample_rate: int) -> Optional[BinaryIO]:
    """Phát âm thanh trực tiếp từ mảng numpy."""
    try:
        buffer = io.BytesIO()
        sf.write(buffer, audio_array, sample_rate, format='WAV')
        buffer.seek(0)
        st.audio(buffer, format="audio/wav")
        return buffer
    except Exception as e:
        st.error(f"Lỗi phát âm thanh: {e}")
        return None

def save_audio(audio_array: np.ndarray, sample_rate: int, output_path: str) -> bool:
    """Lưu âm thanh với nhiều phương pháp dự phòng."""
    # Đảm bảo thư mục tồn tại
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Thử với soundfile
    try:
        sf.write(output_path, audio_array, sample_rate)
        return True
    except Exception:
        pass
    
    # Thử với scipy
    try:
        from scipy.io import wavfile
        wavfile.write(output_path, sample_rate, audio_array)
        return True
    except Exception:
        pass
    
    # Thử với torchaudio
    try:
        torchaudio.save(
            output_path, 
            torch.tensor(audio_array).unsqueeze(0), 
            sample_rate
        )
        return True
    except Exception:
        return False

def generate_filename(text: str, nfe_step: int) -> str:
    """Tạo tên tệp dựa trên nội dung văn bản và thời gian."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"voice_{text_hash}_{nfe_step}steps_{timestamp}.wav"

def get_audio_length(audio_path: str) -> float:
    """Lấy thời lượng của tệp âm thanh (giây)."""
    try:
        info = sf.info(audio_path)
        return info.duration
    except Exception:
        try:
            import wave
            with wave.open(audio_path, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return frames / float(rate)
        except Exception:
            return 0.0
