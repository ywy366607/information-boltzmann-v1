"""
Robot Pet 3D Model Generator - Round 4
Updates:
1. ARMS: Significantly thicker, rounded, organic capsule limbs (radius ~0.052).
   Capsule geometry provides:
   - Analytical closed-form mass properties & inertia tensor.
   - O(1) segment-to-segment distance for collision and contact force.
   - Smooth, chunky, toy-like rounded aesthetic matching the original sketch.
2. FACE SCREEN: Inset dark CRT/monitor screen panel with bezel.
3. EXPRESSION SYSTEM: Screen displays selectable facial expression tokens:
   - "default": classic vertical pill eyes (||)
   - "happy": upward curved crescent arcs (^^)
   - "surprised": wide luminous circles (OO)
   - "sleepy": calm horizontal resting bars (--)
   - "curious": asymmetric tilted eyes (oO)
"""

import os
import math
import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix, translation_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageOps

def create_beveled_box(extents, bevel=0.035):
    """Creates a smooth box with beveled corners and edges."""
    w, d, h = extents
    r = min(bevel, w * 0.16, d * 0.16, h * 0.16)
    hw, hd, hh = w / 2 - r, d / 2 - r, h / 2 - r
    
    corners = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                s = trimesh.creation.icosphere(subdivisions=2, radius=r)
                s.apply_translation([sx * hw, sy * hd, sz * hh])
                corners.append(s)
                
    combined = trimesh.util.concatenate(corners)
    return combined.convex_hull

def create_capsule_between_points(p0, p1, radius):
    """
    Creates a rounded capsule (cylinder + 2 hemispherical caps)
    between two 3D points p0 and p1.
    """
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    vec = p1 - p0
    length = np.linalg.norm(vec)
    if length < 1e-5:
        return trimesh.creation.uv_sphere(radius=radius, count=[14, 14])
        
    direction = vec / length
    # Capsule along Z
    cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=16)
    
    z_axis = np.array([0.0, 0.0, 1.0])
    dot = np.dot(z_axis, direction)
    if np.abs(dot - 1.0) < 1e-6:
        rot = np.eye(4)
    elif np.abs(dot + 1.0) < 1e-6:
        rot = rotation_matrix(np.pi, [1.0, 0.0, 0.0])
    else:
        rot_axis = np.cross(z_axis, direction)
        rot_axis /= np.linalg.norm(rot_axis)
        rot = rotation_matrix(np.arccos(np.clip(dot, -1.0, 1.0)), rot_axis)
        
    cyl.apply_transform(rot)
    cyl.apply_translation((p0 + p1) / 2.0)
    
    # Hemispherical caps
    sph0 = trimesh.creation.uv_sphere(radius=radius, count=[14, 14])
    sph0.apply_translation(p0)
    sph1 = trimesh.creation.uv_sphere(radius=radius, count=[14, 14])
    sph1.apply_translation(p1)
    
    return trimesh.util.concatenate([cyl, sph0, sph1])

def create_loop_antenna_z(width=0.22, height=0.74, tube_r=0.016, segments=48):
    """Tapered wire loop antenna."""
    t = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    width_mod = 1.0 + 0.28 * np.sin(t / 2.0)
    xs = (width / 2.0) * np.sin(t) * width_mod
    zs = (height / 2.0) * (1.0 - np.cos(t))
    ys = np.zeros_like(t)
    points = np.stack([xs, ys, zs], axis=-1)
    
    parts = []
    for i in range(segments):
        p0 = points[i]
        p1 = points[(i + 1) % segments]
        vec = p1 - p0
        l = np.linalg.norm(vec)
        if l < 1e-6:
            continue
        cyl = trimesh.creation.cylinder(radius=tube_r, height=l, sections=10)
        z_axis = np.array([0.0, 0.0, 1.0])
        d = vec / l
        dot = np.dot(z_axis, d)
        if np.abs(dot - 1.0) < 1e-6:
            rot = np.eye(4)
        elif np.abs(dot + 1.0) < 1e-6:
            rot = rotation_matrix(np.pi, [1.0, 0.0, 0.0])
        else:
            ax = np.cross(z_axis, d)
            ax /= np.linalg.norm(ax)
            rot = rotation_matrix(np.arccos(np.clip(dot, -1.0, 1.0)), ax)
        cyl.apply_transform(rot)
        cyl.apply_translation((p0 + p1) / 2.0)
        parts.append(cyl)
        sph = trimesh.creation.uv_sphere(radius=tube_r, count=[8, 8])
        sph.apply_translation(p0)
        parts.append(sph)
        
    return trimesh.util.concatenate(parts)

def create_torus_ring(major_r, minor_r, axis='z', segments=24):
    t = trimesh.creation.torus(major_radius=major_r, minor_radius=minor_r, major_sections=segments, minor_sections=12)
    if axis == 'x':
        t.apply_transform(rotation_matrix(np.pi / 2, [0, 1, 0]))
    elif axis == 'y':
        t.apply_transform(rotation_matrix(np.pi / 2, [1, 0, 0]))
    return t

def create_expression_eyes(expression="default", head_z_center=1.03, head_d=0.66):
    """
    Generates 3D mesh for the facial expression displayed on the screen.
    """
    eye_parts = []
    eye_y = head_d / 2.0 + 0.016  # slightly in front of the screen glass
    spacing_x = 0.11
    
    if expression == "default":
        # Classic vertical pill capsules (||)
        eye_h = 0.22
        eye_r = 0.040
        for sx in [-1, 1]:
            eye = trimesh.creation.capsule(height=eye_h, radius=eye_r)
            eye.apply_translation([sx * spacing_x, eye_y, head_z_center + 0.04])
            eye.visual.vertex_colors = [255, 255, 255, 255]
            eye_parts.append(eye)
            
    elif expression == "happy":
        # Upward curved crescent arcs (^^)
        arc_pts = 16
        theta = np.linspace(np.radians(30), np.radians(150), arc_pts)
        r_arc = 0.10
        xs = r_arc * np.cos(theta)
        zs = r_arc * np.sin(theta)
        for sx in [-1, 1]:
            pts = np.stack([sx * spacing_x + xs, np.full_like(xs, eye_y), head_z_center + 0.02 + zs], axis=-1)
            for i in range(arc_pts - 1):
                seg = create_capsule_between_points(pts[i], pts[i+1], radius=0.032)
                seg.visual.vertex_colors = [255, 255, 255, 255]
                eye_parts.append(seg)
                
    elif expression == "surprised":
        # Big glowing round circles (OO)
        for sx in [-1, 1]:
            ring = create_torus_ring(major_r=0.08, minor_r=0.025, axis='y')
            ring.apply_translation([sx * spacing_x, eye_y, head_z_center + 0.04])
            ring.visual.vertex_colors = [255, 255, 255, 255]
            eye_parts.append(ring)
            
    elif expression == "sleepy":
        # Horizontal resting bars (--)
        bar_len = 0.18
        bar_r = 0.034
        for sx in [-1, 1]:
            p0 = [sx * spacing_x - bar_len / 2, eye_y, head_z_center + 0.03]
            p1 = [sx * spacing_x + bar_len / 2, eye_y, head_z_center + 0.03]
            bar = create_capsule_between_points(p0, p1, radius=bar_r)
            bar.visual.vertex_colors = [255, 255, 255, 255]
            eye_parts.append(bar)
            
    elif expression == "curious":
        # Asymmetric (left eye big O, right eye pill |)
        ring = create_torus_ring(major_r=0.08, minor_r=0.025, axis='y')
        ring.apply_translation([-spacing_x, eye_y, head_z_center + 0.04])
        ring.visual.vertex_colors = [255, 255, 255, 255]
        eye_parts.append(ring)
        
        eye = trimesh.creation.capsule(height=0.20, radius=0.038)
        eye.apply_translation([spacing_x, eye_y, head_z_center + 0.04])
        eye.visual.vertex_colors = [255, 255, 255, 255]
        eye_parts.append(eye)

    return eye_parts

def build_robot_pet(expression="default"):
    parts = []
    
    # -------------------------------------------------------------
    # 1. BOOTS / FEET
    # -------------------------------------------------------------
    boot_w, boot_d, boot_h = 0.22, 0.30, 0.42
    leg_x_spacing = 0.165
    
    for sx in [-1, 1]:
        boot = create_beveled_box([boot_w, boot_d, boot_h], bevel=0.016)
        boot.apply_translation([sx * leg_x_spacing, 0.03, boot_h / 2.0])
        boot.visual.vertex_colors = [232, 236, 244, 255]
        parts.append(boot)
        
    # -------------------------------------------------------------
    # 2. LEGS
    # -------------------------------------------------------------
    leg_z_bottom = boot_h * 0.92
    leg_z_top = 0.66
    leg_r = 0.042
    for sx in [-1, 1]:
        p0 = [sx * leg_x_spacing, 0.02, leg_z_bottom]
        p1 = [sx * leg_x_spacing, 0.02, leg_z_top]
        leg = create_capsule_between_points(p0, p1, radius=leg_r)
        leg.visual.vertex_colors = [190, 196, 208, 255]
        parts.append(leg)
        
    # -------------------------------------------------------------
    # 3. HEAD / TORSO
    # -------------------------------------------------------------
    head_w, head_d, head_h = 0.76, 0.66, 0.74
    head_z_center = leg_z_top + head_h / 2.0  # 1.03
    
    head = create_beveled_box([head_w, head_d, head_h], bevel=0.042)
    head.apply_translation([0.0, 0.0, head_z_center])
    head.visual.vertex_colors = [240, 243, 248, 255]
    parts.append(head)
    
    # -------------------------------------------------------------
    # 4. FACE SCREEN (Inset display bezel)
    # The front is a display screen visor where expressions appear
    # -------------------------------------------------------------
    screen_w = head_w * 0.82   # 0.62
    screen_h = head_h * 0.76   # 0.56
    screen_d = 0.025
    screen_bevel = 0.025
    screen = create_beveled_box([screen_w, screen_d, screen_h], bevel=screen_bevel)
    screen_y = head_d / 2.0 + 0.006
    screen.apply_translation([0.0, screen_y, head_z_center + 0.02])
    # Dark glass monitor face
    screen.visual.vertex_colors = [45, 48, 56, 255]
    parts.append(screen)
    
    # -------------------------------------------------------------
    # 5. EXPRESSIONS (Luminous eyes on the screen)
    # -------------------------------------------------------------
    eye_parts = create_expression_eyes(expression=expression, head_z_center=head_z_center, head_d=head_d)
    parts.extend(eye_parts)
        
    # -------------------------------------------------------------
    # 6. TOP SOCKETS & LOOP ANTENNAS
    # -------------------------------------------------------------
    top_z = leg_z_top + head_h
    socket_x_spacing = 0.18
    socket_y = 0.0
    socket_r = 0.088
    socket_h = 0.026
    
    for sx in [-1, 1]:
        socket = trimesh.creation.cylinder(radius=socket_r, height=socket_h, sections=24)
        socket.apply_translation([sx * socket_x_spacing, socket_y, top_z + socket_h / 2.0])
        socket.visual.vertex_colors = [185, 192, 202, 255]
        parts.append(socket)
        
        inner = trimesh.creation.cylinder(radius=socket_r * 0.72, height=socket_h * 1.15, sections=24)
        inner.apply_translation([sx * socket_x_spacing, socket_y, top_z + socket_h / 2.0])
        inner.visual.vertex_colors = [140, 148, 160, 255]
        parts.append(inner)
        
        antenna = create_loop_antenna_z(width=0.22, height=0.74, tube_r=0.016, segments=48)
        rot_pitch = rotation_matrix(np.radians(-14), [1, 0, 0])
        rot_roll = rotation_matrix(sx * np.radians(10), [0, 1, 0])
        rot_combined = rot_roll @ rot_pitch
        antenna.apply_transform(rot_combined)
        antenna.apply_translation([sx * socket_x_spacing, socket_y - 0.01, top_z + socket_h])
        antenna.visual.vertex_colors = [245, 248, 252, 255]
        parts.append(antenna)
        
    # -------------------------------------------------------------
    # 7. ROUNDED, CHUNKY ARMS & JOINTS (Substantially thickened!)
    # Capsule radius increased to 0.052 (organic, toy-like roundness)
    # -------------------------------------------------------------
    shoulder_x = head_w / 2.0
    shoulder_y = 0.02
    shoulder_z = head_z_center + 0.06
    arm_radius = 0.052  # Chunky and rounded!
    
    for sx in [-1, 1]:
        # 7a. Shoulder concentric circular ring/bearing
        pivot_ring = create_torus_ring(major_r=0.068, minor_r=0.015, axis='x', segments=24)
        pivot_ring.apply_translation([sx * (shoulder_x + 0.038), shoulder_y, shoulder_z])
        pivot_ring.visual.vertex_colors = [180, 186, 198, 255]
        parts.append(pivot_ring)
        
        joint_hub = trimesh.creation.uv_sphere(radius=0.048, count=[14, 14])
        joint_hub.apply_translation([sx * (shoulder_x + 0.038), shoulder_y, shoulder_z])
        joint_hub.visual.vertex_colors = [160, 168, 180, 255]
        parts.append(joint_hub)
        
        # 7b. Upper Arm: Chunky Capsule
        # Origin at shoulder, extends down and slightly forward/outward
        p_shoulder = [sx * (shoulder_x + 0.045), shoulder_y, shoulder_z]
        p_elbow = [
            sx * (shoulder_x + 0.095),
            shoulder_y + 0.08,
            shoulder_z - 0.23
        ]
        upper_arm = create_capsule_between_points(p_shoulder, p_elbow, radius=arm_radius)
        upper_arm.visual.vertex_colors = [228, 233, 242, 255]
        parts.append(upper_arm)
        
        # Elbow joint pivot ring/sphere
        elbow_joint = trimesh.creation.uv_sphere(radius=arm_radius * 1.05, count=[14, 14])
        elbow_joint.apply_translation(p_elbow)
        elbow_joint.visual.vertex_colors = [170, 176, 188, 255]
        parts.append(elbow_joint)
        
        # 7c. Forearm: Chunky Capsule, bent forward
        p_wrist = [
            sx * (shoulder_x + 0.075),
            shoulder_y + 0.23,
            shoulder_z - 0.38
        ]
        fore_arm = create_capsule_between_points(p_elbow, p_wrist, radius=arm_radius * 0.94)
        fore_arm.visual.vertex_colors = [228, 233, 242, 255]
        parts.append(fore_arm)
        
        # 7d. Hand: Chunky rounded loop/ring
        hand_ring = create_torus_ring(major_r=0.055, minor_r=0.015, axis='x', segments=24)
        rot_hand = rotation_matrix(np.radians(35), [1, 0, 0]) @ rotation_matrix(sx * np.radians(22), [0, 0, 1])
        hand_ring.apply_transform(rot_hand)
        hand_ring.apply_translation(p_wrist)
        hand_ring.visual.vertex_colors = [242, 246, 252, 255]
        parts.append(hand_ring)
        
    robot = trimesh.util.concatenate(parts)
    return robot

def render_scene(mesh, azimuth=64, elevation=15, span=0.72, center=(0, 0.04, 1.01), filename="out.png", style="shaded"):
    fig = plt.figure(figsize=(9, 9), dpi=140)
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
    is_screen = (raw_colors[:, 0] < 0.25) & (raw_colors[:, 1] < 0.25)
    
    shaded = raw_colors * dots[:, None]
    # Pure emissive eyes
    shaded[is_eye] = [1.0, 1.0, 1.0]
    # Screen glass soft specular
    shaded[is_screen] = raw_colors[is_screen] * (0.8 + 0.4 * dots[is_screen, None])
    
    shaded[~is_eye & ~is_screen] += np.array([0.04, 0.06, 0.09]) * (1.0 - dots[~is_eye & ~is_screen, None])
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

def render_expression_sheet(out_path="present/robot_pet/expressions_sheet.png"):
    """
    Renders 4 facial expressions side by side to showcase the screen expression system.
    """
    exprs = [
        ("Default (||)", "default"),
        ("Happy (^^)", "happy"),
        ("Surprised (OO)", "surprised"),
        ("Sleepy (--)", "sleepy")
    ]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=120, subplot_kw={'projection': '3d'})
    bg_color = '#16181f'
    fig.patch.set_facecolor(bg_color)
    center = np.array([0, 0.05, 1.03])
    span = 0.72
    
    for i, (title, expr_name) in enumerate(exprs):
        ax = axes[i]
        ax.set_facecolor(bg_color)
        m = build_robot_pet(expression=expr_name)
        verts = m.vertices - center
        faces = m.faces
        normals = m.face_normals
        
        az_rad = np.radians(85)
        el_rad = np.radians(20)
        light = np.array([np.cos(el_rad)*np.cos(az_rad), np.cos(el_rad)*np.sin(az_rad), np.sin(el_rad)])
        light /= np.linalg.norm(light)
        
        dots = np.clip(np.dot(normals, light), 0.25, 1.0)
        raw_colors = m.visual.vertex_colors[faces].mean(axis=1)[:, :3] / 255.0
        is_eye = np.all(raw_colors > 0.98, axis=1)
        is_screen = (raw_colors[:, 0] < 0.25) & (raw_colors[:, 1] < 0.25)
        
        shaded = raw_colors * dots[:, None]
        shaded[is_eye] = [1.0, 1.0, 1.0]
        shaded[is_screen] = raw_colors[is_screen] * (0.8 + 0.4 * dots[is_screen, None])
        shaded[~is_eye & ~is_screen] += np.array([0.04, 0.06, 0.09]) * (1.0 - dots[~is_eye & ~is_screen, None])
        shaded = np.clip(shaded, 0.0, 1.0)
        
        face_rgba = np.column_stack([shaded, np.ones(len(faces))])
        face_rgba[is_eye] = [1.0, 1.0, 1.0, 1.0]
        
        poly = Poly3DCollection(verts[faces], facecolors=face_rgba, edgecolors='#252a36', linewidths=0.1, antialiased=True)
        ax.add_collection3d(poly)
        
        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
        ax.set_zlim(-span, span)
        ax.view_init(elev=6, azim=80)
        ax.set_axis_off()
        ax.set_title(title, color='#cfd6e6', fontsize=13, pad=8, y=0.96)
        
    plt.tight_layout()
    plt.savefig(out_path, facecolor=bg_color, edgecolor='none', pad_inches=0.05)
    plt.close(fig)
    print(f"Rendered expression sheet: {out_path}")

if __name__ == "__main__":
    out_dir = "present/robot_pet"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Building Round 4 Model (Chunky rounded arms + Screen face)...")
    robot = build_robot_pet(expression="default")
    
    # Save standard 3D formats
    obj_p = os.path.join(out_dir, "robot_pet.obj")
    glb_p = os.path.join(out_dir, "robot_pet.glb")
    robot.export(obj_p)
    robot.export(glb_p)
    print(f"Saved: {obj_p}, {glb_p}")
    
    # 1. New perspective renders
    match_shaded = os.path.join(out_dir, "round4_match_shaded.png")
    match_wire = os.path.join(out_dir, "round4_match_wire.png")
    render_scene(robot, azimuth=64, elevation=15, span=0.72, filename=match_shaded, style="shaded")
    render_scene(robot, azimuth=64, elevation=15, span=0.72, filename=match_wire, style="sketch")
    
    # 2. Expression sheet
    expr_sheet = os.path.join(out_dir, "expressions_sheet.png")
    render_expression_sheet(expr_sheet)
    
    # 3. 1:1 Side by Side Comparison
    sketch_path = "C:/Users/85750/.gemini/antigravity/brain/8c845106-7bc4-4dfe-a086-5d162932c955/.user_uploaded/media_1790666515231.jpg"
    img_user = Image.open(sketch_path)
    w, h = img_user.size
    crop_box = (int(w * 0.12), int(h * 0.22), int(w * 0.85), int(h * 0.72))
    img_user_crop = img_user.crop(crop_box)
    
    target_size = (800, 800)
    img_user_crop = ImageOps.fit(img_user_crop, target_size, method=Image.Resampling.LANCZOS)
    img_wire = ImageOps.fit(Image.open(match_wire), target_size, method=Image.Resampling.LANCZOS)
    img_shaded = ImageOps.fit(Image.open(match_shaded), target_size, method=Image.Resampling.LANCZOS)
    
    total_w = target_size[0] * 3 + 40
    total_h = target_size[1] + 40
    canvas = Image.new("RGB", (total_w, total_h), color=(22, 25, 32))
    canvas.paste(img_user_crop, (10, 20))
    canvas.paste(img_wire, (target_size[0] + 20, 20))
    canvas.paste(img_shaded, (target_size[0] * 2 + 30, 20))
    
    comp_path = os.path.join(out_dir, "round4_side_by_side.png")
    canvas.save(comp_path)
    print("Saved Round 4 Comparison:", comp_path)
