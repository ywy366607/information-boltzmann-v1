"""
Create side-by-side comparison image:
Original User Sketch vs 3D Procedural Mesh
"""
import os
from PIL import Image, ImageOps

sketch_path = "C:/Users/85750/.gemini/antigravity/brain/8c845106-7bc4-4dfe-a086-5d162932c955/.user_uploaded/media_1790666515231.jpg"
render_sketch_path = "present/robot_pet/sketch_wire_refined.png"
render_clay_path = "present/robot_pet/sketch_match_refined.png"
out_path = "present/robot_pet/side_by_side_comparison.png"

# Load images
img_user = Image.open(sketch_path)
# User sketch is vertical (tall tablet photo, e.g. 1080x2400 or similar)
# Let's crop user sketch around the robot character
w, h = img_user.size
# Robot in sketch is centered roughly between y=0.20*h and y=0.75*h, x=0.10*w and 0.85*w
# Let's find crop box or use smart crop
crop_box = (int(w * 0.10), int(h * 0.22), int(w * 0.85), int(h * 0.72))
img_user_crop = img_user.crop(crop_box)

target_size = (700, 700)
img_user_crop = ImageOps.fit(img_user_crop, target_size, method=Image.Resampling.LANCZOS)

img_render_wire = Image.open(render_sketch_path)
img_render_wire = ImageOps.fit(img_render_wire, target_size, method=Image.Resampling.LANCZOS)

img_render_clay = Image.open(render_clay_path)
img_render_clay = ImageOps.fit(img_render_clay, target_size, method=Image.Resampling.LANCZOS)

# Create 3-panel comparison: [User Sketch] | [3D Wireframe Match] | [3D Shaded Solid]
total_w = target_size[0] * 3 + 40
total_h = target_size[1] + 60

canvas = Image.new("RGB", (total_w, total_h), color=(20, 22, 28))

canvas.paste(img_user_crop, (10, 40))
canvas.paste(img_render_wire, (target_size[0] + 20, 40))
canvas.paste(img_render_clay, (target_size[0] * 2 + 30, 40))

canvas.save(out_path)
print("Saved side-by-side comparison:", out_path)
