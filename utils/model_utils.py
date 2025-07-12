# utils/model_utils.py
import os
import sys
import torch
import streamlit as st
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any, Callable, Union

def safe_execute(func: Callable, fallback=None, error_msg: Optional[str] = None):
    """Thực thi hàm an toàn, trả về fallback nếu gặp lỗi."""
    try:
        return func()
    except Exception as e:
        if error_msg:
            st.error(f"{error_msg}: {e}")
        return fallback

def get_available_models(model_dir: Path) -> List[str]:
    """Lấy danh sách mô hình có sẵn."""
    models = []
    for ext in ["*.safetensors", "*.pt"]:
        models.extend([p.name for p in model_dir.glob(ext)])
    return models or ["model_486000.safetensors"]

def get_reference_audio_files(audio_dir: Path) -> List[str]:
    """Lấy danh sách tệp âm thanh tham chiếu."""
    files = []
    for ext in ["wav", "mp3", "flac", "ogg", "m4a"]:
        files.extend([p.name for p in audio_dir.glob(f"*.{ext}")])
    return files

def preload_vocoder(model_dir: Path, repo_id: str = "charactr/vocos-mel-24khz") -> Optional[str]:
    """Tải trước vocoder từ Hugging Face Hub."""
    try:
        from huggingface_hub import snapshot_download
        
        # Tạo thư mục đích
        local_dir = model_dir / Path(repo_id).name
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # Tải mô hình
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        
        return str(local_dir)
    except Exception as e:
        st.error(f"Lỗi tải vocoder: {e}")
        return None

def enhance_model_generate(model):
    """Cải thiện method generate của mô hình với xử lý dấu câu tốt hơn."""
    import re
    
    # Lưu hàm gốc
    original_generate = model.generate
    
    # Định nghĩa hàm mới
    def enhanced_generate(text, *args, **kwargs):
        # Cải thiện xử lý dấu câu
        text = re.sub(r'([.,!?;:])(?=[^\s])', r'\1 ', text)
        text = re.sub(r',\s*', ', ', text)
        text = re.sub(r'\.\s*', '. ', text)
        text = re.sub(r'\s+', ' ', text)
        return original_generate(text, *args, **kwargs)
    
    # Thay thế hàm
    model.generate = enhanced_generate
    return model

def load_tts_model(
    model_path: str, 
    vocab_file: str, 
    vocoder_name: str, 
    use_ema: bool, 
    vocos_local_path: Optional[str] = None
) -> Tuple[Any, Optional[str]]:
    """Tải mô hình F5TTS với xử lý lỗi tốt hơn."""
    from f5tts_wrapper import F5TTSWrapper
    
    # Kiểm tra tệp
    if not Path(model_path).exists():
        return None, f"Không tìm thấy tệp mô hình: {model_path}"
    if not Path(vocab_file).exists():
        return None, f"Không tìm thấy tệp từ vựng: {vocab_file}"
    
    # Thiết lập tham số
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {"device": device}
    
    # Thiết lập vocoder local cho Windows
    if sys.platform.startswith('win') and vocoder_name == "vocos" and vocos_local_path:
        if Path(vocos_local_path).exists():
            kwargs.update({
                "use_local_vocoder": True,
                "vocoder_path": vocos_local_path
            })
    
    # Khởi tạo mô hình
    try:
        model = F5TTSWrapper(
            vocoder_name=vocoder_name,
            ckpt_path=model_path,
            vocab_file=vocab_file,
            use_ema=use_ema,
            **kwargs
        )
        
        # Nâng cao mô hình
        model = enhance_model_generate(model)
        return model, None
    except Exception as e:
        # Thử lại với phiên bản API khác
        try:
            # Bỏ tham số device nếu không hỗ trợ
            if "device" in kwargs:
                del kwargs["device"]
            
            model = F5TTSWrapper(
                vocoder_name=vocoder_name,
                ckpt_path=model_path,
                vocab_file=vocab_file,
                use_ema=use_ema,
                **kwargs
            )
            
            # Nâng cao mô hình
            model = enhance_model_generate(model)
            return model, None
        except Exception as e2:
            return None, f"Lỗi khởi tạo mô hình: {e2}"
