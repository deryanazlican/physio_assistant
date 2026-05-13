import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def put_text_tr(img, text, pos, font_size=24, color=(255, 255, 255), bold=False):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font_name = "arialbd.ttf" if bold else "arial.ttf"
        font = ImageFont.truetype(font_name, font_size)
    except:
        font = ImageFont.load_default()

    draw.text(pos, str(text), font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def wrap_text(text, max_width, font_size=24, bold=False):
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    img_pil = Image.fromarray(cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font_name = "arialbd.ttf" if bold else "arial.ttf"
        font = ImageFont.truetype(font_name, font_size)
    except:
        font = ImageFont.load_default()

    text = str(text).replace("\r", "")
    paragraphs = text.split("\n")
    lines = []

    for para in paragraphs:
        if not para.strip():
            lines.append("")
            continue

        words = para.split()
        current = ""

        for word in words:
            test = word if not current else current + " " + word
            try:
                bbox = draw.textbbox((0, 0), test, font=font)
                w = bbox[2] - bbox[0]
            except:
                w = len(test) * font_size * 0.55

            if w <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word

        if current:
            lines.append(current)

    return lines


def add_chat_message(messages: list, role: str, text: str):
    messages.append({
        "role": role,
        "text": text
    })


def draw_chat_panel(frame, messages, input_text="", scroll_offset=0):
    h, w = frame.shape[:2]

    panel_x = int(w * 0.16)
    panel_y = int(h * 0.10)
    panel_w = int(w * 0.68)
    panel_h = int(h * 0.80)

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)

    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (255, 255, 0), 2)

    frame = put_text_tr(frame, "AI ASİSTAN", (panel_x + 330, panel_y + 20), 34, (0, 255, 255), True)
    frame = put_text_tr(frame, "Komutlar: /plan  /rapor  /bitir   | ESC kapat", (panel_x + 100, panel_y + 60), 18, (220, 220, 220))

    line_height = 30
    content_x = panel_x + 22
    content_y = panel_y + 120
    content_w = panel_w - 90
    content_h = panel_h - 190

    # Sağ tarafta butonlar
    btn_w = 42
    btn_h = 42
    btn_x = panel_x + panel_w - 58

    up_btn = (btn_x, content_y, btn_x + btn_w, content_y + btn_h)
    down_btn = (btn_x, content_y + content_h - btn_h, btn_x + btn_w, content_y + content_h)

    # Buton çiz
    cv2.rectangle(frame, (up_btn[0], up_btn[1]), (up_btn[2], up_btn[3]), (70, 70, 70), -1)
    cv2.rectangle(frame, (up_btn[0], up_btn[1]), (up_btn[2], up_btn[3]), (0, 255, 255), 2)
    frame = put_text_tr(frame, "▲", (up_btn[0] + 10, up_btn[1] + 5), 24, (255, 255, 255), True)

    cv2.rectangle(frame, (down_btn[0], down_btn[1]), (down_btn[2], down_btn[3]), (70, 70, 70), -1)
    cv2.rectangle(frame, (down_btn[0], down_btn[1]), (down_btn[2], down_btn[3]), (0, 255, 255), 2)
    frame = put_text_tr(frame, "▼", (down_btn[0] + 10, down_btn[1] + 5), 24, (255, 255, 255), True)

    # Mesajlar
    all_lines = []
    for msg in messages:
        role = msg["role"]
        text = msg["text"]
        prefix = "Sen: " if role == "user" else "AI: "
        wrapped_lines = wrap_text(prefix + text, content_w - 10, font_size=24)

        color = (80, 255, 80) if role == "user" else (240, 240, 240)
        for line in wrapped_lines:
            all_lines.append((line, color))
        all_lines.append(("", color))

    max_visible_lines = max(1, content_h // line_height)
    max_scroll = max(0, len(all_lines) - max_visible_lines)

    scroll_offset = max(0, min(scroll_offset, max_scroll))
    start_idx = max(0, len(all_lines) - max_visible_lines - scroll_offset)
    end_idx = min(len(all_lines), start_idx + max_visible_lines)

    visible_lines = all_lines[start_idx:end_idx]

    y = content_y
    for line, color in visible_lines:
        if line:
            frame = put_text_tr(frame, line, (content_x, y - 22), 24, color)
        y += line_height

    # Scroll bar
    bar_x = btn_x + 15
    bar_y1 = content_y + btn_h + 10
    bar_y2 = content_y + content_h - btn_h - 10
    bar_h = max(1, bar_y2 - bar_y1)

    cv2.rectangle(frame, (bar_x, bar_y1), (bar_x + 6, bar_y2), (70, 70, 70), -1)

    if max_scroll > 0:
        thumb_h = max(36, int((max_visible_lines / max(1, len(all_lines))) * bar_h))
        thumb_range = max(1, bar_h - thumb_h)
        thumb_y = bar_y1 + int((scroll_offset / max_scroll) * thumb_range)
    else:
        thumb_h = bar_h
        thumb_y = bar_y1

    cv2.rectangle(frame, (bar_x, thumb_y), (bar_x + 6, thumb_y + thumb_h), (0, 255, 255), -1)

    # Input
    input_y = panel_y + panel_h - 72
    cv2.rectangle(frame, (panel_x + 20, input_y), (panel_x + panel_w - 20, input_y + 50), (0, 255, 255), 2)
    frame = put_text_tr(frame, input_text[-90:] + "|", (panel_x + 30, input_y + 10), 24, (255, 255, 255))

    ui_info = {
        "up_btn": up_btn,
        "down_btn": down_btn,
        "max_scroll": max_scroll
    }

    return frame, scroll_offset, max_scroll, ui_info