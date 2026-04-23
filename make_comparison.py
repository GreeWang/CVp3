from PIL import Image

img1_path = "/root/CVpart3/results/part3/wildvideo/20260418_143539/outputs/enhanced_frames/000011.png"
img2_path = "/root/CVpart3/results/part3/wildvideo/20260419_120446/outputs/enhanced_frames/000011.png"
out_path = "/root/CVpart3/results/part3/wildvideo/blending_comparison.png"

try:
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    w, h = img1.size
    # 裁剪右半部分 (w//2 到 w)
    crop_box = (w // 2, 0, w, h)
    i1 = img1.crop(crop_box)
    i2 = img2.crop(crop_box)

    cw, ch = i1.size
    # 竖向拼接：宽不变，高度 * 2
    combined = Image.new('RGB', (cw, ch * 2))
    combined.paste(i1, (0, 0))
    combined.paste(i2, (0, ch))

    combined.save(out_path)
    print(f"Combined image successfully saved to: {out_path}")
except Exception as e:
    print(f"Error: {e}")
