from PIL import Image, ImageDraw, ImageFont

img1_path = "/root/CVpart3/results/part3/wildvideo/20260418_143539/outputs/enhanced_frames/000011.png"
img2_path = "/root/CVpart3/results/part3/wildvideo/20260419_120446/outputs/enhanced_frames/000011.png"
out_path = "/root/CVpart3/results/part3/wildvideo/blending_comparison_labeled_horizontal.png"

try:
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    w, h = img1.size
    # 依然裁剪出各自的右半部分 (w//2 到 w)
    crop_box = (w // 2, 0, w, h)
    i1 = img1.crop(crop_box)
    i2 = img2.crop(crop_box)

    cw, ch = i1.size
    
    # 左右拼接：宽 * 2，高度不变
    combined = Image.new('RGB', (cw * 2, ch))
    combined.paste(i1, (0, 0))
    combined.paste(i2, (cw, 0))

    draw = ImageDraw.Draw(combined)

    # 尝试加载加大字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
    except IOError:
        font = ImageFont.load_default()

    def draw_text_with_outline(draw, pos, text, font):
        x, y = pos
        shadowcol = (0, 0, 0)
        draw.text((x-2, y-2), text, font=font, fill=shadowcol)
        draw.text((x+2, y-2), text, font=font, fill=shadowcol)
        draw.text((x-2, y+2), text, font=font, fill=shadowcol)
        draw.text((x+2, y+2), text, font=font, fill=shadowcol)
        draw.text((x, y), text, font=font, fill=(255, 255, 255))

    # 添加文字标注：左侧左上角是 Before，右侧左上角是 After
    draw_text_with_outline(draw, (20, 20), "Before", font)
    draw_text_with_outline(draw, (cw + 20, 20), "After", font)

    combined.save(out_path)
    print(f"Horizontal labeled image successfully saved to: {out_path}")
except Exception as e:
    print(f"Error: {e}")
