from PIL import Image, ImageDraw, ImageFont

in_path = "/root/CVpart3/results/part3/wildvideo/blending_comparison.png"
out_path = "/root/CVpart3/results/part3/wildvideo/blending_comparison_labeled.png"

try:
    img = Image.open(in_path)
    draw = ImageDraw.Draw(img)

    cw, total_h = img.size
    ch = total_h // 2

    # 尝试加载更大的系统字体，如果失败则使用默认字体并尝试放大
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
    except IOError:
        font = ImageFont.load_default()

    # 绘制带黑边的白字，保证在任何背景下都能看清
    def draw_text_with_outline(draw, pos, text, font):
        x, y = pos
        shadowcol = (0, 0, 0)
        # 粗描边
        draw.text((x-2, y-2), text, font=font, fill=shadowcol)
        draw.text((x+2, y-2), text, font=font, fill=shadowcol)
        draw.text((x-2, y+2), text, font=font, fill=shadowcol)
        draw.text((x+2, y+2), text, font=font, fill=shadowcol)
        # 实体白字
        draw.text((x, y), text, font=font, fill=(255, 255, 255))

    # 添加文字标注
    draw_text_with_outline(draw, (20, 20), "Before", font)
    draw_text_with_outline(draw, (20, ch + 20), "After", font)

    img.save(out_path)
    print(f"Labeled image successfully saved to: {out_path}")
except Exception as e:
    print(f"Error: {e}")
