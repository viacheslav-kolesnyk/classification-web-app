"""
Generates clean sample test images for the 6 waste categories.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def generate_sample_images(samples_dir: Path):
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    categories = [
        ("cardboard_sample.jpg", "Cardboard Box", (194, 153, 107), (120, 80, 40), "📦"),
        ("glass_sample.jpg", "Glass Bottle", (178, 223, 219), (0, 105, 92), "🍾"),
        ("metal_sample.jpg", "Aluminium Can", (207, 216, 220), (69, 90, 100), "🥫"),
        ("paper_sample.jpg", "Printed Paper", (245, 245, 245), (33, 33, 33), "📄"),
        ("plastic_sample.jpg", "Plastic Bottle", (255, 224, 178), (230, 81, 0), "🧴"),
        ("trash_sample.jpg", "Organic Residual", (189, 189, 189), (33, 33, 33), "🗑️")
    ]
    
    for filename, title, bg_color, text_color, emoji in categories:
        img = Image.new("RGB", (400, 400), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Draw decorative border & shapes
        draw.rectangle([(20, 20), (380, 380)], outline=text_color, width=4)
        draw.rectangle([(40, 40), (360, 360)], fill=(255, 255, 255, 128), outline=text_color, width=2)
        
        # Draw central circle
        draw.ellipse([(140, 110), (260, 230)], fill=bg_color, outline=text_color, width=3)
        
        # Draw text labels
        draw.text((200, 260), title, fill=text_color, anchor="mm")
        draw.text((200, 300), "EcoSort Test Sample", fill=text_color, anchor="mm")
        draw.text((200, 330), "(Click to analyze)", fill=(100, 100, 100), anchor="mm")
        
        img.save(samples_dir / filename, quality=95)
        print(f"Generated {filename}")

if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "static" / "samples"
    generate_sample_images(out_dir)
