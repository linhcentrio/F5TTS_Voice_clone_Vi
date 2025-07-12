# utils/text_utils.py
import re
import sys
import types
import streamlit as st

def windows_ttsnorm(text):
    """Chuẩn hóa văn bản tiếng Việt đơn giản cho Windows."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([.,!?;:])(?=[^\s])', r'\1 ', text)
    text = re.sub(r'([.!?])\s+([a-zA-ZÀ-ỹ])', lambda m: f"{m.group(1)} {m.group(2).upper()}", text)
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    if text and not text.rstrip().endswith(('.', '?', '!')):
        text += '.'
    return text

def enhance_text(text):
    """Cải thiện văn bản với xử lý dấu câu tốt hơn."""
    # Đảm bảo dấu câu có khoảng trắng sau
    text = re.sub(r'([.,!?;:])(?=[^\s])', r'\1 ', text)
    
    # Tăng cường khoảng nghỉ cho dấu phẩy
    text = re.sub(r',\s*', ', ', text)
    
    # Tăng cường khoảng nghỉ cho dấu chấm
    text = re.sub(r'\.\s*', '. ', text)
    
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text)
    
    return text

def split_long_text(text, max_chars=150):
    """Chia nhỏ văn bản dài theo dấu câu."""
    # Các pattern ngắt câu
    patterns = [
        r'(?<=[.!?])\s+',      # Ưu tiên dấu câu mạnh
        r'(?<=[;:])\s+',       # Dấu chấm phẩy, hai chấm
        r'(?<=[,])\s+'         # Dấu phẩy
    ]
    
    # Thử chia theo mức độ ưu tiên
    chunks = [text]
    for pattern in patterns:
        if all(len(c.encode('utf-8')) <= max_chars for c in chunks):
            break
            
        new_chunks = []
        for chunk in chunks:
            if len(chunk.encode('utf-8')) <= max_chars:
                new_chunks.append(chunk)
            else:
                parts = re.split(pattern, chunk)
                new_chunks.extend(parts)
        chunks = new_chunks
    
    # Xử lý đoạn vẫn còn quá dài
    result = []
    for chunk in chunks:
        if len(chunk.encode('utf-8')) <= max_chars:
            result.append(chunk)
        else:
            # Chia theo từ nếu cần
            words = chunk.split()
            current = ""
            for word in words:
                if len((current + " " + word).encode('utf-8')) <= max_chars:
                    current = (current + " " + word).strip()
                else:
                    if current:
                        result.append(current)
                    current = word
            if current:
                result.append(current)
    
    # Đảm bảo dấu câu hợp lý khi nối lại
    for i in range(len(result) - 1):
        if not result[i].rstrip().endswith(('.', ',', ':', ';', '!', '?')):
            result[i] = result[i].rstrip() + ','
    
    return result

def setup_ttsnorm():
    """Thiết lập TTSnorm tương thích với platform."""
    if sys.platform.startswith('win'):
        # Tạo module giả cho vinorm.TTSnorm
        if 'vinorm' not in sys.modules:
            vinorm_module = types.ModuleType('vinorm')
            sys.modules['vinorm'] = vinorm_module
            sys.modules['vinorm'].TTSnorm = windows_ttsnorm
        # Import từ module giả
        from vinorm import TTSnorm
    else:
        # Thử import từ module thật
        try:
            from vinorm import TTSnorm
        except ImportError:
            # Fallback nếu không có
            TTSnorm = lambda text: text
    
    return TTSnorm
