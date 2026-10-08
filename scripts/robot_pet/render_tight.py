"""
Tight zoom render matching the exact framing of the user's sketch.
"""
import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageOps
from generate_model import build_robot_pet

out_dir = "present/robot_pet"
robot = build_robot_pet()

def render_tight(mesh, azimuth=64, elevation=15, span=0.68, center=(0, 0.04, 1.00), filename="sketch_tight.png", style="sketch"):
    fig = plt.figure(figsize=(8, 8), dpi=140)
    ax = fig.add_subplot(111, projection='3d')
    
    bg_color = '#1b1d24' if style == "sketch" else '#181a20'
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    verts = mesh.vertices - np.array(center)
    faces = mesh.faces
    
    az_rad = np.radians(azimuth + 20)
    el_rad = np.radians(elevation + 22)
    light = np.array([np.cos(el_rad)*np.cos(az_rad), np.cos(el_rad)*np.sin(az_rad), np.sin(el_rad)])
    light /= np.linalg.norm(light)
    
    normals = mesh.face_normals
    dots = np.clip(np.dot(normals, light), 0.22, 1.0)
    
    raw_colors = mesh.visual.vertex_colors[faces].mean(axis=1)[:, :3] / 255.0
    is_eye = np.all(raw_colors > 0.98, axis=1)
    
    shaded = raw_colors * dots[:, None]
    shaded[is_eye] = [1.0, 1.0, 1.0]
    shaded[~is_eye] += np.array([0.04, 0.06, 0.09]) * (1.0 - dots[~is_eye, None])
    shaded = np.clip(shaded, 0.0, 1.0)
    
    if style == "sketch":
        edge_c = '#b0d0f0'
        lw = 0.42
        face_rgba = np.column_stack([shaded * 0.72, np.ones(len(faces)) * 0.92])
        face_rgba[is_eye] = [1.0, 1.0, 1.0, 1.0]
    else:
        edge_c = '#252a36'
        lw = 0.12
        face_rgba = np.column_stack([shaded, np.ones(len(faces))])
        face_rgba[is_eye] = [1.0, 1.0, 1.0, 1.0]
        
    poly = Poly3DCollection(verts[faces], facecolors=face_rgba, edgecolors=edge_c, linewidths=lw, antialiased=True)
    ax.add_collection3d(poly)
    
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_zlim(-span, span)
    
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_axis_off()
    
    plt.tight_layout()
    plt.savefig(filename, facecolor=bg_color, edgecolor='none', pad_inches=0)
    plt.close(fig)

tight_wire = os.path.join(out_dir, "tight_wire.png")
tight_clay = os.path.join(out_dir, "tight_clay.png")
render_tight(robot, filename=tight_wire, style="sketch")
render_tight(robot, filename=tight_clay, style="shaded")

# Composite side-by-side with original sketch
sketch_path = "C:/Users/85750/.gemini/antigravity/brain/8c845106-7bc4-4dfe-a086-5d162932c955/.user_uploaded/media_1790666515231.jpg"
img_user = Image.open(sketch_path)
w, h = img_user.size
crop_box = (int(w * 0.12), int(h * 0.22), int(w * 0.85), int(h * 0.72))
img_user_crop = img_user.crop(crop_box)

target_size = (800, 800)
img_user_crop = ImageOps.fit(img_user_crop, target_size, method=Image.Resampling.LANCZOS)
img_render_wire = ImageOps.fit(Image.open(tight_wire), target_size, method=Image.Resampling.LANCZOS)
img_render_clay = ImageOps.fit(Image.open(tight_clay), target_size, method=Image.Resampling.LANCZOS)

total_w = target_size[0] * 3 + 40
total_h = target_size[1] + 40
canvas = Image.new("RGB", (total_w, total_h), color=(22, 25, 32))
canvas.paste(img_user_crop, (10, 20))
canvas.paste(img_render_wire, (target_size[0] + 20, 20))
canvas.paste(img_render_clay, (target_size[0] * 2 + 30, 20))

comp_path = os.path.join(out_dir, "final_side_by_side.png")
canvas.save(comp_path)
print("Saved final comparison to:", comp_path)
