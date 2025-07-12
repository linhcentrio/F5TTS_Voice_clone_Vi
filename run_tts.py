from pathlib import Path
import tempfile
import traceback
from datetime import timedelta
from functools import lru_cache
from typing import Optional, Dict, List, Any, Tuple

import gradio as gr
import torch
import pysrt
from pydub import AudioSegment
from faster_whisper import WhisperModel

# Thiết lập môi trường
MODEL_DIR = Path(__file__).parent / "EraX-WoW-Turbo-V1.1-CT2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "float32"

# Danh sách ngôn ngữ: Map từ tên hiển thị -> mã
LANGUAGE_MAP = {
    "Tự động nhận diện": "auto",
    "Tiếng Việt": "vi",
    "Tiếng Anh": "en",
    "Tiếng Pháp": "fr", 
    "Tiếng Đức": "de",
    "Tiếng Tây Ban Nha": "es",
    "Tiếng Trung Quốc": "zh",
    "Tiếng Nhật": "ja",
    "Tiếng Hàn Quốc": "ko",
    "Tiếng Nga": "ru",
    "Tiếng Thái": "th",
    **{f"Tiếng {code}": code for code in "af am ar as az ba be bg bn bo br bs ca cs cy da el et eu fa fi fo gl gu ha haw he hi hr ht hu hy id is it jw ka kk km kn la lb ln lo lt lv mg mi mk ml mn mr ms mt my ne nl nn no oc pa pl ps pt ro sa sd si sk sl sn so sq sr su sv sw ta te tg tk tl tr tt uk ur uz yi yo yue".split()}
}

@lru_cache(maxsize=1)
def load_model() -> WhisperModel:
    """Tải model một lần và cache lại kết quả"""
    try:
        # Chỉ sử dụng các tham số được hỗ trợ bởi CTranslate2
        model = WhisperModel(
            str(MODEL_DIR), 
            device=DEVICE, 
            compute_type=COMPUTE_TYPE
        )
        return model
    except Exception as e:
        if DEVICE == "cuda":
            print(f"Lỗi khi tải model trên GPU: {e}")
            print("Chuyển sang CPU")
            return WhisperModel(str(MODEL_DIR), device="cpu", compute_type="float32")
        raise

def convert_audio(audio_path: str) -> str:
    """Chuyển đổi âm thanh sang mono 16kHz"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(temp_file.name, format="wav")
        return temp_file.name

def create_srt(segments: List[Dict[str, Any]]) -> str:
    """Tạo nội dung SRT từ các đoạn"""
    subs = pysrt.SubRipFile()
    
    for i, segment in enumerate(segments, start=1):
        start_time = timedelta(seconds=segment["bắt_đầu"])
        end_time = timedelta(seconds=segment["kết_thúc"])
        
        item = pysrt.SubRipItem(
            index=i,
            start=pysrt.SubRipTime(
                hours=start_time.seconds // 3600,
                minutes=(start_time.seconds % 3600) // 60,
                seconds=start_time.seconds % 60,
                milliseconds=int(start_time.microseconds / 1000)
            ),
            end=pysrt.SubRipTime(
                hours=end_time.seconds // 3600,
                minutes=(end_time.seconds % 3600) // 60,
                seconds=end_time.seconds % 60,
                milliseconds=int(end_time.microseconds / 1000)
            ),
            text=segment["văn_bản"]
        )
        subs.append(item)
    
    return "".join(f"{sub.index}\n{sub.start} --> {sub.end}\n{sub.text}\n\n" for sub in subs)

def transcribe_audio(
    audio_file: Optional[str], 
    display_language: str = "Tiếng Việt", 
    beam_size: int = 5, 
    vad_filter: bool = True,
    temperature: float = 0.0, 
    word_timestamps: bool = False, 
    *_
) -> Tuple[str, str, List[Dict[str, Any]], str, str]:
    """Phiên âm và tạo phụ đề từ file audio"""
    if not audio_file:
        return "Không có file âm thanh.", "Vui lòng tải lên hoặc ghi âm.", [], "", "Sẵn sàng"
    
    processed_audio = None
    
    try:
        # Lấy mã ngôn ngữ từ tên hiển thị
        lang_code = LANGUAGE_MAP.get(display_language, "auto")
        lang_code = None if lang_code == "auto" else lang_code
        
        # Chuẩn bị và xử lý audio
        model = load_model()
        processed_audio = convert_audio(audio_file)
        
        # Đo thời gian xử lý
        import time
        start_time = time.time()
        
        # Phiên âm
        segments, info = model.transcribe(
            audio=processed_audio,
            beam_size=beam_size,
            language=lang_code,
            temperature=temperature,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps
        )
        
        # Xử lý kết quả
        segments_list = list(segments)
        segment_data = [
            {
                "bắt_đầu": segment.start,
                "kết_thúc": segment.end,
                "văn_bản": segment.text,
                **({"các_từ": [{"từ": w.word, "bắt_đầu": w.start, "kết_thúc": w.end} 
                   for w in segment.words]} if word_timestamps and hasattr(segment, 'words') else {})
            }
            for segment in segments_list
        ]
        
        # Tạo đầu ra
        full_text = " ".join(segment.text for segment in segments_list)
        srt_content = create_srt(segment_data)
        
        # Thông tin hiệu suất
        processing_time = time.time() - start_time
        detected_language = (
            f"Ngôn ngữ phát hiện: {info.language} (độ tin cậy: {info.language_probability:.2f}) | "
            f"Thời gian xử lý: {processing_time:.2f} giây"
        )
        
        return full_text, detected_language, segment_data, srt_content, "Đã hoàn thành"
        
    except Exception as e:
        print(f"Lỗi: {e}")
        print(traceback.format_exc())
        return f"Lỗi khi phiên âm: {str(e)}", "", [], "", f"Gặp lỗi: {str(e)}"
        
    finally:
        # Xóa file tạm
        if processed_audio and Path(processed_audio).exists() and processed_audio != audio_file:
            try:
                Path(processed_audio).unlink()
            except:
                pass

def save_srt(srt_content: str, audio_path: str) -> str:
    """Lưu nội dung SRT vào file"""
    if not srt_content or not audio_path:
        return "Không có nội dung SRT để tải xuống"
    
    try:
        # Tạo tên file SRT từ tên file âm thanh
        audio_path = Path(audio_path)
        srt_path = audio_path.with_suffix('.srt')
        
        # Lưu file
        srt_path.write_text(srt_content, encoding="utf-8")
        return f"Đã lưu file SRT tại: {srt_path}"
    except Exception as e:
        return f"Lỗi khi lưu file SRT: {e}"

def create_interface() -> gr.Blocks:
    """Tạo giao diện Gradio"""
    with gr.Blocks(title="EraX-WoW-Turbo STT", theme=gr.themes.Soft()) as app:
        gr.Markdown("# EraX-WoW-Turbo Nhận Diện Giọng Nói")
        
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath")
                
                with gr.Row():
                    language = gr.Dropdown(
                        choices=list(LANGUAGE_MAP.keys()),
                        value="Tiếng Việt",
                        label="Ngôn ngữ"
                    )
                    beam_size = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Beam Size")
                
                with gr.Row():
                    vad_filter = gr.Checkbox(value=True, label="Lọc VAD")
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, label="Nhiệt độ")
                    word_timestamps = gr.Checkbox(value=False, label="Timestamp từng từ")
                
                transcribe_btn = gr.Button("Phiên Âm", variant="primary")
            
            with gr.Column(scale=1):
                full_text = gr.Textbox(label="Văn bản phiên âm", lines=10)
                detected_lang = gr.Textbox(label="Thông tin")
                
                with gr.Tabs():
                    with gr.TabItem("JSON"):
                        segments_json = gr.JSON()
                    with gr.TabItem("SRT"):
                        srt_output = gr.Textbox(lines=10)
                        download_btn = gr.Button("Tải xuống SRT")
        
        status = gr.Textbox(label="Trạng thái", value="Sẵn sàng")
        
        # Kết nối các hàm với UI
        transcribe_btn.click(
            fn=transcribe_audio,
            inputs=[audio_input, language, beam_size, vad_filter, temperature, word_timestamps],
            outputs=[full_text, detected_lang, segments_json, srt_output, status]
        )
        
        download_btn.click(
            fn=save_srt,
            inputs=[srt_output, audio_input],
            outputs=[status]
        )
        
    return app

if __name__ == "__main__":
    app = create_interface()
    app.queue()  # Thêm queue để tránh lỗi HTTP
    app.launch(share=True)
