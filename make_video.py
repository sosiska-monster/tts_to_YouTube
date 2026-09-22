import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, AudioFileClip

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
ASSETS_DIR = BASE_DIR / "assets"

W, H = 1080, 1920
FPS = 30

FONT_PATH = str(ASSETS_DIR / "font.ttf") if (ASSETS_DIR / "font.ttf").exists() else "arial.ttf"

def get_font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()

FONT_NAME = get_font(28)
FONT_TEXT = get_font(34)

COLOR_BG_RIGHT = (0, 122, 255)       # Синий iMessage (Jess)
COLOR_BG_LEFT = (44, 44, 46)         # Темно-серый iMessage (Остальные)
COLOR_TEXT_MAIN = (255, 255, 255)
COLOR_NAME_TEXT = (160, 160, 165)

def wrap_text(text, font, max_width, draw):
    words = text.split()
    lines = []
    current_line = []
    for w in words:
        test_line = " ".join(current_line + [w])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            current_line.append(w)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [w]
    if current_line:
        lines.append(" ".join(current_line))
    return lines

def render_bubble(item, max_bubble_w=760):
    dummy_img = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(dummy_img)

    lines = wrap_text(item["text"], FONT_TEXT, max_bubble_w - 60, draw)
    line_h = 44
    text_block_h = len(lines) * line_h

    pad_x, pad_y = 28, 20
    name_h = 36 if item["side"] == "left" else 0

    bubble_w = min(max_bubble_w, max(draw.textbbox((0, 0), l, font=FONT_TEXT)[2] for l in lines) + pad_x * 2 + 20)
    bubble_w = max(bubble_w, 240)
    bubble_h = text_block_h + pad_y * 2 + name_h

    img = Image.new("RGBA", (int(bubble_w), int(bubble_h)), (0, 0, 0, 0))
    bdraw = ImageDraw.Draw(img)

    bg_color = COLOR_BG_RIGHT if item["side"] == "right" else COLOR_BG_LEFT
    bdraw.rounded_rectangle([(0, 0), (bubble_w, bubble_h)], radius=28, fill=bg_color)

    y_offset = pad_y
    if item["side"] == "left":
        bdraw.text((pad_x, y_offset - 4), item["display_name"], font=FONT_NAME, fill=COLOR_NAME_TEXT)
        y_offset += name_h

    for line in lines:
        bdraw.text((pad_x, y_offset), line, font=FONT_TEXT, fill=COLOR_TEXT_MAIN)
        y_offset += line_h

    return img

def create_chat_overlay(current_ms, timestamps):
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    active_items = [item for item in timestamps if current_ms >= item["start_ms"]]
    if not active_items:
        return canvas

    bubbles = [render_bubble(it) for it in active_items]
    gap = 24
    total_chat_h = sum(b.height for b in bubbles) + gap * (len(bubbles) - 1)

    bottom_anchor = H - 320
    start_y = bottom_anchor - total_chat_h

    cur_y = start_y
    for item, b_img in zip(active_items, bubbles):
        if item["side"] == "right":
            cur_x = W - b_img.width - 40
        else:
            cur_x = 40

        if cur_y + b_img.height > 120 and cur_y < H - 100:
            canvas.paste(b_img, (int(cur_x), int(cur_y)), b_img)
        cur_y += b_img.height + gap

    return canvas

def make_video():
    timestamps_path = OUTPUT_DIR / "dialog_timestamps.json"
    audio_path = OUTPUT_DIR / "final_voiceover.wav"
    gameplay_path = ASSETS_DIR / "gameplay.mp4"
    out_video_path = OUTPUT_DIR / "final_reddit_chat_short.mp4"

    if not timestamps_path.exists() or not audio_path.exists():
        print("[!] Ошибка: сначала выполните синтез через python voice_engine.py")
        return

    with open(timestamps_path, "r", encoding="utf-8") as f:
        timestamps = json.load(f)

    voice_clip = AudioFileClip(str(audio_path))
    duration = voice_clip.duration

    print(f"[*] Длительность аудио: {duration:.2f} сек. Открытие фонового видео...")
    bg_clip = VideoFileClip(str(gameplay_path))
    if bg_clip.duration < duration:
        bg_clip = bg_clip.loop(duration=duration)
    else:
        bg_clip = bg_clip.subclip(0, duration)

    bg_w, bg_h = bg_clip.size
    target_ratio = 9 / 16
    curr_ratio = bg_w / bg_h

    if curr_ratio > target_ratio:
        new_w = int(bg_h * target_ratio)
        x_center = bg_w // 2
        bg_clip = bg_clip.crop(x1=x_center - new_w // 2, width=new_w, height=bg_h)
    bg_clip = bg_clip.resize((W, H))

    print("[*] Рендеринг анимации переписки...")
    def process_frame(get_frame, t):
        frame = get_frame(t)
        current_ms = int(t * 1000)
        overlay = create_chat_overlay(current_ms, timestamps)
        pil_frame = Image.fromarray(frame).convert("RGBA")
        combined = Image.alpha_composite(pil_frame, overlay)
        return np.array(combined.convert("RGB"))

    final_clip = bg_clip.fl(process_frame).set_audio(voice_clip)
    final_clip.write_videofile(
        str(out_video_path),
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        preset="fast",
        threads=4
    )
    print(f"\n[✓] Видео успешно собрано: {out_video_path}")

if __name__ == "__main__":
    make_video()