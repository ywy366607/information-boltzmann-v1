"""
Robot Pet Dynamic Visor Facial Expression & Affect Hierarchy System.

Theoretical Foundation:
- Grounded in Paul Ekman's 6 Primary Emotions & Shaver/Parrott's Affect Tree:
  1. JOY (喜悦 / 快乐)
  2. CURIOSITY (好奇 / 探索 - Active Inference Epistemic Engine)
  3. SURPRISE (惊讶 / 错愕 - Surprisal / Prediction Error Spike)
  4. SADNESS (悲伤 / 沮丧 - Persistent Error / Physical Failure)
  5. FEAR (恐惧 / 警惕 - Pragmatic Risk / Loss of Equilibrium)
  6. ANGER (愤怒 / 抵触 - Environmental Resistance / Obstacle)
  Plus NEUTRAL (平静 / 稳态).
- Grounded in Karl Friston's Active Inference Affective Dynamics:
  - Valence: dF/dt rate of free energy minimization.
  - Arousal: Kinetic velocity + audio/tactile sensory influx.
  - Epistemic Salience: Expected information gain driving inquisitive focus.
  - Pragmatic Risk: Distance from homeostatic balance envelope.

Visual Themes (4 Selectable Styles):
- CYBER_CHIBI: Cute anime robot expressions with glowing eyes, blush, curved smiling mouths.
- SCI_FI_HUD: Cyberpunk tactical visor with holographic rings, scanning arcs, telemetry reticles.
- RETRO_PIXEL: Nostalgic 8-bit Tamagotchi dot matrix.
- MINIMALIST: Clean, organic Apple/Eve-style luminous ellipses.
"""

from enum import IntEnum, Enum
from typing import Dict, Any, Tuple, Optional
import numpy as np
import math
import mujoco
from PIL import Image, ImageDraw

WIDTH = 256
HEIGHT = 256

class ThemeStyle(IntEnum):
    CYBER_CHIBI = 0  # 萌系赛博 / 灵动动漫
    SCI_FI_HUD = 1   # 未来科幻 / 全息战术目镜
    RETRO_PIXEL = 2  # 复古像素 / 8-Bit 点阵 LED
    MINIMALIST = 3   # 极简流光 / 苹果&皮克斯有机机甲

class PrimaryEmotion(IntEnum):
    NEUTRAL = 0      # 平静 / 稳态
    JOY = 1          # 喜悦 / 快乐
    CURIOSITY = 2    # 好奇 / 探索
    SURPRISE = 3     # 惊讶 / 错愕
    SADNESS = 4      # 悲伤 / 沮丧
    FEAR = 5         # 恐惧 / 警惕
    ANGER = 6        # 愤怒 / 抵触

class SubEmotion(str, Enum):
    # NEUTRAL
    IDLE_CALM = "IDLE_CALM"          # 平静待机
    
    # JOY
    CHEERFUL = "CHEERFUL"            # 欣喜开朗 (^^ 弯月微笑)
    ECSTATIC = "ECSTATIC"            # 狂喜兴奋 (> < 欢呼)
    CONTENT = "CONTENT"              # 惬意满足 (◠‿◠ 温和)
    AFFECTIONATE = "AFFECTIONATE"    # 喜爱亲昵 (♡ ♡ 心心眼)
    PLAYFUL = "PLAYFUL"              # 顽皮嬉戏 (^_- 眨眼眨舌)
    
    # CURIOSITY
    INQUISITIVE = "INQUISITIVE"      # 探究思索 (•ิ_•ิ)? 挑眉侧视
    FOCUS_SCAN = "FOCUS_SCAN"        # 专注扫描 (准星聚焦)
    AWE_WONDER = "AWE_WONDER"        # 惊奇赞叹 (★_★ 星星眼)
    EUREKA = "EUREKA"                # 灵光一闪 (感叹号 / 灯泡)
    
    # SURPRISE
    STARTLED = "STARTLED"            # 大吃一惊 (O_O 震颤)
    BLANK_DOTS = "BLANK_DOTS"        # 呆愣错愕 (·_· 愣神)
    PUZZLED = "PUZZLED"              # 困惑迷茫 (?_o 歪头困惑)
    
    # SADNESS
    POUTY = "POUTY"                  # 委屈失落 (｡•́︿•̀｡ 撇嘴)
    DIZZY_FALLEN = "DIZZY_FALLEN"    # 跌倒眩晕 (@_@ 蚊香圈)
    SLEEPY = "SLEEPY"                # 困倦疲乏 (-_- 睡意朦胧)
    WEEPING = "WEEPING"              # 心碎悲伤 (T_T 泪汪汪)
    
    # FEAR
    PANICKED = "PANICKED"            # 慌张惊恐 (>_< 滴汗发抖)
    ALERT_GUARD = "ALERT_GUARD"      # 高度戒备 (°ロ° 警戒框)
    TIMID = "TIMID"                  # 怯懦畏缩 (斜眼游移)
    
    # ANGER
    ANNOYED = "ANNOYED"              # 恼怒不爽 (¬_¬ 倒八字眉)
    RESISTING = "RESISTING"          # 坚韧顽抗 (｀へ´ 咬牙发力)
    OVERHEATED = "OVERHEATED"        # 过载暴躁 (冒烟警告)

# Legacy compatibility token mapping
class ExpressionToken(IntEnum):
    DEFAULT = 0
    HAPPY = 1
    SLEEPY = 2
    SURPRISED = 3
    CURIOUS = 4

class ExpressionSystem:
    """
    Unified Hierarchical Affect & Multi-Style Procedural Digital Visor System.
    """
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        
        # Texture memory address binding for screen_face_tex
        self.tex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, "screen_face_tex")
        self.tex_adr = model.tex_adr[self.tex_id] if self.tex_id != -1 else None
        
        # Legacy joint IDs
        self.jl_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "eye_roll_l")
        self.jr_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "eye_roll_r")
        self.qpos_jl = model.jnt_qposadr[self.jl_id] if self.jl_id != -1 else None
        self.qpos_jr = model.jnt_qposadr[self.jr_id] if self.jr_id != -1 else None
        
        # Current active states
        self.theme_style = ThemeStyle.CYBER_CHIBI
        self.primary_emotion = PrimaryEmotion.JOY
        self.sub_emotion = SubEmotion.CHEERFUL
        self.valence = 0.65       # [-1.0 (pain/distress) to +1.0 (joy/relief)]
        self.arousal = 0.40       # [0.0 (asleep/calm) to 1.0 (hyper-active/shock)]
        self.manual_override = False
        self.manual_override_timer = 0.0
        
        # Internal affect dynamics
        self.prev_vfe = 0.0
        self.vfe_velocity = 0.0
        self.shock_cooldown = 0.0
        self.curiosity_cooldown = 0.0
        self.petted_cooldown = 0.0
        self.blink_timer = 0.0
        self.is_blinking = False
        self.life_time = 0.0
        
        # Current procedural image array (256, 256, 3)
        self.canvas_arr = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        self._needs_upload = True
        
        # Render initial face
        self._render_current_face()

    def set_theme(self, theme: ThemeStyle):
        """Switches visual theme style."""
        self.theme_style = ThemeStyle(theme)
        self._needs_upload = True

    def set_emotion_explicit(self, primary: PrimaryEmotion, sub: Optional[SubEmotion] = None, hold_duration: float = 3.0):
        """Allows manual testing / audition of any emotion from UI."""
        self.primary_emotion = PrimaryEmotion(primary)
        if sub is not None:
            self.sub_emotion = SubEmotion(sub)
        else:
            # Pick canonical sub-emotion for primary
            canonical = {
                PrimaryEmotion.NEUTRAL: SubEmotion.IDLE_CALM,
                PrimaryEmotion.JOY: SubEmotion.CHEERFUL,
                PrimaryEmotion.CURIOSITY: SubEmotion.INQUISITIVE,
                PrimaryEmotion.SURPRISE: SubEmotion.STARTLED,
                PrimaryEmotion.SADNESS: SubEmotion.POUTY,
                PrimaryEmotion.FEAR: SubEmotion.PANICKED,
                PrimaryEmotion.ANGER: SubEmotion.RESISTING,
            }
            self.sub_emotion = canonical.get(self.primary_emotion, SubEmotion.IDLE_CALM)
            
        self.manual_override = True
        self.manual_override_timer = hold_duration
        self._needs_upload = True

    def set_expression(self, data: mujoco.MjData, token: ExpressionToken):
        """Legacy compatibility method."""
        token_map = {
            ExpressionToken.DEFAULT: (PrimaryEmotion.NEUTRAL, SubEmotion.IDLE_CALM),
            ExpressionToken.HAPPY: (PrimaryEmotion.JOY, SubEmotion.CHEERFUL),
            ExpressionToken.SLEEPY: (PrimaryEmotion.SADNESS, SubEmotion.SLEEPY),
            ExpressionToken.SURPRISED: (PrimaryEmotion.SURPRISE, SubEmotion.STARTLED),
            ExpressionToken.CURIOUS: (PrimaryEmotion.CURIOSITY, SubEmotion.INQUISITIVE),
        }
        p, s = token_map.get(token, (PrimaryEmotion.NEUTRAL, SubEmotion.IDLE_CALM))
        self.primary_emotion = p
        self.sub_emotion = s
        self._needs_upload = True

    def select_autonomous_token(
        self,
        dt: float,
        audio_amplitude: float,
        pitch_rad: float,
        roll_rad: float,
        forward_speed: float,
        steering_rate: float,
        vfe: float = 0.0,
        curiosity: float = 0.0,
        risk: float = 0.0,
        is_fallen: bool = False,
        is_petted: bool = False
    ) -> ExpressionToken:
        """
        Continuous Active Inference Emotion Emergence Engine:
        Maps variational thermodynamics and sensorimotor state into the 6 Primary Emotions:
        - Rate of Free Energy change dF/dt -> Valence
        - Acoustic wavefronts & impacts -> Surprisal (Shock)
        - Epistemic curiosity & exploration -> Curiosity
        - Loss of balance & tumbling -> Fear & Sadness (Dizzy/Fallen)
        - External disturbance push resistance -> Anger (Resisting)
        - Caretaker touch & steady gait -> Joy
        """
        self.life_time += dt
        
        # Free energy velocity: dF/dt
        self.vfe_velocity = (vfe - self.prev_vfe) / max(dt, 1e-4)
        self.prev_vfe = vfe
        
        # Natural periodic blinking (every ~3.5s for 0.12s)
        self.blink_timer += dt
        if self.blink_timer > 3.5:
            self.is_blinking = True
            if self.blink_timer > 3.65:
                self.is_blinking = False
                self.blink_timer = 0.0
                self._needs_upload = True

        # Handle manual override hold timer
        if self.manual_override:
            self.manual_override_timer -= dt
            if self.manual_override_timer <= 0:
                self.manual_override = False
            else:
                return self._to_legacy_token()

        tilt_deg = math.degrees(math.sqrt(pitch_rad**2 + roll_rad**2))

        # Check explicit caretaker pet event
        if is_petted:
            self.petted_cooldown = 2.5
            self.primary_emotion = PrimaryEmotion.JOY
            self.sub_emotion = SubEmotion.AFFECTIONATE
            self.valence = 0.95
            self.arousal = 0.50
            self._needs_upload = True
            return ExpressionToken.HAPPY

        if self.petted_cooldown > 0:
            self.petted_cooldown -= dt
            return ExpressionToken.HAPPY

        # 1. Physical Shock & Acoustic Wave Surprisal (> 25 Pa or severe stumble)
        if self.shock_cooldown > 0:
            self.shock_cooldown -= dt
        elif audio_amplitude > 25.0 or (tilt_deg > 32.0 and not is_fallen):
            self.shock_cooldown = 0.8
            self.primary_emotion = PrimaryEmotion.SURPRISE
            self.sub_emotion = SubEmotion.STARTLED
            self.valence = -0.2
            self.arousal = 0.95
            self._needs_upload = True
            return ExpressionToken.SURPRISED

        # 2. Catastrophic Fall or Fallen State
        if is_fallen or tilt_deg > 45.0:
            self.primary_emotion = PrimaryEmotion.SADNESS
            self.sub_emotion = SubEmotion.DIZZY_FALLEN
            self.valence = -0.85
            self.arousal = 0.40
            self._needs_upload = True
            return ExpressionToken.SLEEPY

        # 3. High Pragmatic Risk (imminent loss of balance)
        if risk > 0.65 or tilt_deg > 18.0:
            self.primary_emotion = PrimaryEmotion.FEAR
            self.sub_emotion = SubEmotion.PANICKED if tilt_deg > 25.0 else SubEmotion.ALERT_GUARD
            self.valence = -0.60
            self.arousal = 0.85
            self._needs_upload = True
            return ExpressionToken.DEFAULT

        # 4. External Force Disturbance Resistance (Anger / Resisting)
        if abs(steering_rate) > 0.60 and forward_speed < 0.05:
            self.primary_emotion = PrimaryEmotion.ANGER
            self.sub_emotion = SubEmotion.RESISTING
            self.valence = -0.40
            self.arousal = 0.75
            self._needs_upload = True
            return ExpressionToken.DEFAULT

        # 5. Epistemic Curiosity (Active Inference Salience Exploration)
        if curiosity > 0.28 or abs(steering_rate) > 0.25:
            self.primary_emotion = PrimaryEmotion.CURIOSITY
            if curiosity > 0.45:
                self.sub_emotion = SubEmotion.AWE_WONDER
            elif audio_amplitude > 2.0:
                self.sub_emotion = SubEmotion.FOCUS_SCAN
            else:
                self.sub_emotion = SubEmotion.INQUISITIVE
            self.valence = 0.50
            self.arousal = 0.60
            self._needs_upload = True
            return ExpressionToken.CURIOUS

        # 6. Joy / Flow State (Stable forward limit-cycle walking, low VFE)
        if forward_speed > 0.12 and tilt_deg < 10.0:
            self.primary_emotion = PrimaryEmotion.JOY
            if forward_speed > 0.22:
                self.sub_emotion = SubEmotion.ECSTATIC
            else:
                self.sub_emotion = SubEmotion.CHEERFUL
            self.valence = 0.80
            self.arousal = 0.65
            self._needs_upload = True
            return ExpressionToken.HAPPY

        # 7. Peaceful Homeostatic Equilibrium (Standing upright, low error)
        if tilt_deg < 6.0 and abs(forward_speed) < 0.06:
            self.primary_emotion = PrimaryEmotion.JOY
            self.sub_emotion = SubEmotion.CONTENT
            self.valence = 0.55
            self.arousal = 0.25
            self._needs_upload = True
            return ExpressionToken.HAPPY

        # Default Neutral
        self.primary_emotion = PrimaryEmotion.NEUTRAL
        self.sub_emotion = SubEmotion.IDLE_CALM
        self.valence = 0.10
        self.arousal = 0.20
        self._needs_upload = True
        return ExpressionToken.DEFAULT

    def _to_legacy_token(self) -> ExpressionToken:
        if self.primary_emotion == PrimaryEmotion.JOY:
            return ExpressionToken.HAPPY
        elif self.primary_emotion == PrimaryEmotion.CURIOSITY:
            return ExpressionToken.CURIOUS
        elif self.primary_emotion == PrimaryEmotion.SURPRISE:
            return ExpressionToken.SURPRISED
        elif self.primary_emotion == PrimaryEmotion.SADNESS:
            return ExpressionToken.SLEEPY
        return ExpressionToken.DEFAULT

    def upload_to_renderer(self, mjr_context):
        """Uploads current procedural facial expression texture to GPU VRAM."""
        if self.tex_adr is None or self.tex_id == -1:
            return
            
        if self._needs_upload:
            self._render_current_face()
            # Copy into MjModel tex_data
            size = WIDTH * HEIGHT * 3
            self.model.tex_data[self.tex_adr : self.tex_adr + size] = self.canvas_arr.flatten()
            # Upload to OpenGL texture via MuJoCo C-API wrapper
            mujoco.mjr_uploadTexture(self.model, mjr_context, self.tex_id)
            self._needs_upload = False

    def _render_current_face(self):
        """Procedurally draws the active emotion on the 256x256 RGB canvas."""
        img = Image.new("RGB", (WIDTH, HEIGHT), color=(14, 18, 26))
        draw = ImageDraw.Draw(img)
        # Screen border
        draw.rounded_rectangle([4, 4, WIDTH-5, HEIGHT-5], radius=16, outline=(35, 45, 65), width=2)
        
        # Handle blinking
        if self.is_blinking:
            draw.line([(55, 138), (101, 138)], fill=(0, 240, 255), width=4)
            draw.line([(155, 138), (201, 138)], fill=(0, 240, 255), width=4)
            self.canvas_arr = np.array(img, dtype=np.uint8)
            return

        t = self.life_time
        th = self.theme_style
        p = self.primary_emotion
        s = self.sub_emotion

        if th == ThemeStyle.CYBER_CHIBI:
            self._paint_cyber_chibi(draw, p, s, t)
        elif th == ThemeStyle.SCI_FI_HUD:
            self._paint_sci_fi_hud(draw, p, s, t)
        elif th == ThemeStyle.RETRO_PIXEL:
            self._paint_retro_pixel(draw, p, s, t)
        elif th == ThemeStyle.MINIMALIST:
            self._paint_minimalist(draw, p, s, t)

        self.canvas_arr = np.array(img, dtype=np.uint8)

    # ------------------ THEME 1: CYBER CHIBI ------------------
    def _paint_cyber_chibi(self, draw: ImageDraw.ImageDraw, p: PrimaryEmotion, s: SubEmotion, t: float):
        CYAN = (0, 240, 255)
        GOLD = (255, 200, 50)
        ROSE_PINK = (255, 90, 140)
        WHITE = (245, 250, 255)
        PURPLE = (195, 110, 255)
        AMBER = (255, 150, 30)
        RED = (255, 60, 60)
        
        # JOY
        if p == PrimaryEmotion.JOY:
            if s == SubEmotion.CHEERFUL:
                # Upward smiling crescent arcs ^ ^
                for x in range(50, 105):
                    y = int(115 + 0.038 * (x - 77)**2)
                    draw.line([(x, y-4), (x, y+4)], fill=CYAN, width=3)
                for x in range(151, 206):
                    y = int(115 + 0.038 * (x - 178)**2)
                    draw.line([(x, y-4), (x, y+4)], fill=CYAN, width=3)
                # Blush
                draw.ellipse([42, 142, 68, 160], fill=(255, 80, 130))
                draw.ellipse([188, 142, 214, 160], fill=(255, 80, 130))
                # Smiling mouth
                draw.arc([114, 138, 142, 162], start=10, end=170, fill=GOLD, width=3)

            elif s == SubEmotion.ECSTATIC: # > <
                draw.line([(55, 116), (95, 136)], fill=CYAN, width=5)
                draw.line([(55, 156), (95, 136)], fill=CYAN, width=5)
                draw.line([(201, 116), (161, 136)], fill=CYAN, width=5)
                draw.line([(201, 156), (161, 136)], fill=CYAN, width=5)
                draw.chord([110, 142, 146, 180], start=0, end=180, fill=GOLD, outline=GOLD)
                draw.ellipse([38, 150, 68, 175], fill=ROSE_PINK)
                draw.ellipse([188, 150, 218, 175], fill=ROSE_PINK)
                # Sparkles
                draw.polygon([(128, 88), (131, 96), (139, 99), (131, 102), (128, 110), (125, 102), (117, 99), (125, 96)], fill=WHITE)

            elif s == SubEmotion.CONTENT: # (◠‿◠)
                for x in range(54, 102):
                    y = int(122 + 0.026 * (x - 78)**2)
                    draw.line([(x, y-3), (x, y+3)], fill=(50, 255, 180), width=2)
                for x in range(154, 202):
                    y = int(122 + 0.026 * (x - 178)**2)
                    draw.line([(x, y-3), (x, y+3)], fill=(50, 255, 180), width=2)
                draw.ellipse([46, 146, 68, 162], fill=(255, 100, 140))
                draw.ellipse([188, 146, 210, 162], fill=(255, 100, 140))
                draw.arc([116, 142, 140, 160], start=10, end=170, fill=(50, 255, 180), width=2)

            elif s == SubEmotion.AFFECTIONATE: # Heart eyes ♡ ♡
                # Left Heart
                draw.polygon([(78, 155), (55, 130), (55, 115), (68, 110), (78, 120), (88, 110), (101, 115), (101, 130)], fill=ROSE_PINK)
                draw.pieslice([55, 108, 78, 126], start=180, end=360, fill=ROSE_PINK)
                draw.pieslice([78, 108, 101, 126], start=180, end=360, fill=ROSE_PINK)
                # Right Heart
                draw.polygon([(178, 155), (155, 130), (155, 115), (168, 110), (178, 120), (188, 110), (201, 115), (201, 130)], fill=ROSE_PINK)
                draw.pieslice([155, 108, 178, 126], start=180, end=360, fill=ROSE_PINK)
                draw.pieslice([178, 108, 201, 126], start=180, end=360, fill=ROSE_PINK)
                draw.arc([114, 148, 142, 168], start=10, end=170, fill=ROSE_PINK, width=3)

            elif s == SubEmotion.PLAYFUL: # Wink (^_-)
                for x in range(50, 105):
                    y = int(115 + 0.038 * (x - 77)**2)
                    draw.line([(x, y-4), (x, y+4)], fill=CYAN, width=3)
                draw.ellipse([160, 112, 200, 152], fill=CYAN)
                draw.ellipse([168, 117, 182, 131], fill=WHITE)
                draw.arc([112, 142, 144, 168], start=20, end=160, fill=GOLD, width=3)

        # CURIOSITY
        elif p == PrimaryEmotion.CURIOSITY:
            if s == SubEmotion.INQUISITIVE:
                # Tilted left brow
                draw.line([(52, 95), (102, 108)], fill=CYAN, width=4)
                draw.ellipse([58, 118, 96, 158], fill=CYAN)
                draw.ellipse([68, 124, 82, 138], fill=WHITE)
                # Raised right brow
                draw.arc([150, 85, 196, 110], start=190, end=350, fill=CYAN, width=4)
                draw.ellipse([152, 112, 202, 162], fill=CYAN)
                draw.ellipse([162, 118, 182, 138], fill=WHITE)
                draw.ellipse([184, 142, 194, 152], fill=WHITE)
                # Question spark
                draw.arc([210, 95, 224, 112], start=180, end=360, fill=GOLD, width=3)
                draw.line([(224, 105), (218, 116)], fill=GOLD, width=3)
                draw.point([(218, 122)], fill=GOLD)
                draw.ellipse([122, 150, 134, 164], outline=GOLD, width=3)

            elif s == SubEmotion.AWE_WONDER: # Starry eyes
                for cx in [78, 178]:
                    draw.ellipse([cx-26, 110, cx+26, 162], fill=PURPLE)
                    star_pts = []
                    for i in range(8):
                        r = 18 if i % 2 == 0 else 8
                        ang = i * math.pi / 4
                        star_pts.append((cx + r*math.cos(ang), 136 + r*math.sin(ang)))
                    draw.polygon(star_pts, fill=WHITE)
                draw.ellipse([120, 150, 136, 168], fill=GOLD)

            elif s == SubEmotion.FOCUS_SCAN:
                draw.ellipse([56, 118, 100, 154], fill=CYAN)
                draw.ellipse([156, 118, 200, 154], fill=CYAN)
                draw.line([(40, 136), (216, 136)], fill=(0, 255, 200), width=2)
                draw.line([(116, 152), (140, 152)], fill=CYAN, width=3)

            elif s == SubEmotion.EUREKA:
                draw.ellipse([56, 116, 100, 160], fill=CYAN)
                draw.ellipse([156, 116, 200, 160], fill=CYAN)
                draw.line([(128, 85), (128, 108)], fill=GOLD, width=4)
                draw.ellipse([126, 113, 130, 117], fill=GOLD)
                draw.chord([114, 148, 142, 172], start=0, end=180, fill=GOLD)

        # SURPRISE
        elif p == PrimaryEmotion.SURPRISE:
            if s == SubEmotion.STARTLED:
                draw.ellipse([50, 105, 106, 161], outline=WHITE, width=6)
                draw.ellipse([64, 119, 92, 147], fill=CYAN)
                draw.ellipse([150, 105, 206, 161], outline=WHITE, width=6)
                draw.ellipse([164, 119, 192, 147], fill=CYAN)
                draw.ellipse([116, 150, 140, 180], outline=WHITE, width=4)
                for p1, p2 in [((36, 125), (44, 130)), ((36, 142), (44, 137)), ((220, 125), (212, 130)), ((220, 142), (212, 137))]:
                    draw.line([p1, p2], fill=WHITE, width=3)

            elif s == SubEmotion.BLANK_DOTS:
                draw.ellipse([70, 128, 84, 142], fill=WHITE)
                draw.ellipse([172, 128, 186, 142], fill=WHITE)
                draw.line([(116, 154), (140, 154)], fill=WHITE, width=3)

            elif s == SubEmotion.PUZZLED:
                draw.ellipse([54, 110, 102, 158], fill=CYAN)
                draw.ellipse([162, 124, 192, 154], fill=CYAN)
                draw.arc([114, 150, 142, 166], start=180, end=360, fill=GOLD, width=3)

        # SADNESS
        elif p == PrimaryEmotion.SADNESS:
            if s == SubEmotion.POUTY:
                for x in range(52, 102):
                    y = int(136 - 0.028 * (x - 77)**2)
                    draw.line([(x, y-3), (x, y+3)], fill=(120, 180, 255), width=3)
                for x in range(154, 204):
                    y = int(136 - 0.028 * (x - 179)**2)
                    draw.line([(x, y-3), (x, y+3)], fill=(120, 180, 255), width=3)
                draw.arc([114, 150, 142, 168], start=180, end=360, fill=(120, 180, 255), width=3)

            elif s == SubEmotion.DIZZY_FALLEN: # Spiral eyes
                for cx in [78, 178]:
                    for theta_deg in range(0, 720, 10):
                        th = math.radians(theta_deg)
                        r = 0.035 * theta_deg
                        px = int(cx + r * math.cos(th))
                        py = int(135 + r * math.sin(th))
                        draw.point([(px, py), (px+1, py), (px, py+1)], fill=PURPLE)
                draw.line([(114, 160), (122, 154), (132, 162), (142, 156)], fill=PURPLE, width=3)

            elif s == SubEmotion.SLEEPY:
                draw.line([(55, 136), (101, 136)], fill=(100, 160, 200), width=4)
                draw.line([(155, 136), (201, 136)], fill=(100, 160, 200), width=4)
                draw.line([(120, 156), (136, 156)], fill=(100, 160, 200), width=2)
                draw.line([(195, 95), (205, 95), (195, 107), (205, 107)], fill=WHITE, width=2)

            elif s == SubEmotion.WEEPING:
                draw.line([(55, 126), (101, 126)], fill=(80, 160, 255), width=4)
                draw.line([(155, 126), (201, 126)], fill=(80, 160, 255), width=4)
                draw.line([(78, 130), (78, 175)], fill=(0, 200, 255), width=4)
                draw.line([(178, 130), (178, 175)], fill=(0, 200, 255), width=4)
                draw.arc([114, 156, 142, 174], start=180, end=360, fill=(80, 160, 255), width=3)

        # FEAR
        elif p == PrimaryEmotion.FEAR:
            if s == SubEmotion.PANICKED:
                draw.line([(55, 122), (95, 138)], fill=AMBER, width=4)
                draw.line([(55, 154), (95, 138)], fill=AMBER, width=4)
                draw.line([(201, 122), (161, 138)], fill=AMBER, width=4)
                draw.line([(201, 154), (161, 138)], fill=AMBER, width=4)
                draw.line([(114, 156), (120, 152), (126, 158), (132, 152), (138, 158), (144, 154)], fill=AMBER, width=3)
                draw.polygon([(215, 110), (208, 125), (222, 125)], fill=(0, 180, 255))
                draw.pieslice([208, 116, 222, 130], start=0, end=180, fill=(0, 180, 255))

            elif s == SubEmotion.ALERT_GUARD:
                draw.ellipse([58, 118, 98, 158], outline=AMBER, width=4)
                draw.ellipse([74, 134, 82, 142], fill=WHITE)
                draw.ellipse([158, 118, 198, 158], outline=AMBER, width=4)
                draw.ellipse([174, 134, 182, 142], fill=WHITE)
                draw.rectangle([118, 148, 138, 168], outline=AMBER, width=3)

            elif s == SubEmotion.TIMID:
                draw.ellipse([58, 120, 98, 152], fill=CYAN)
                draw.ellipse([82, 128, 94, 144], fill=WHITE)
                draw.ellipse([158, 120, 198, 152], fill=CYAN)
                draw.ellipse([182, 128, 194, 144], fill=WHITE)
                draw.line([(120, 156), (136, 156)], fill=CYAN, width=2)

        # ANGER
        elif p == PrimaryEmotion.ANGER:
            if s == SubEmotion.ANNOYED:
                draw.line([(50, 112), (105, 130)], fill=RED, width=5)
                draw.line([(206, 112), (151, 130)], fill=RED, width=5)
                draw.ellipse([64, 132, 92, 156], fill=RED)
                draw.ellipse([164, 132, 192, 156], fill=RED)
                draw.line([(116, 158), (140, 152)], fill=RED, width=3)

            elif s == SubEmotion.RESISTING:
                draw.line([(50, 108), (105, 128)], fill=RED, width=5)
                draw.line([(206, 108), (151, 128)], fill=RED, width=5)
                draw.rectangle([60, 128, 96, 148], fill=WHITE)
                draw.rectangle([160, 128, 196, 148], fill=WHITE)
                draw.rectangle([114, 150, 142, 164], outline=WHITE, width=2)
                draw.line([(123, 150), (123, 164)], fill=WHITE, width=2)
                draw.line([(133, 150), (133, 164)], fill=WHITE, width=2)

            elif s == SubEmotion.OVERHEATED:
                draw.line([(48, 105), (108, 128)], fill=(255, 30, 30), width=6)
                draw.line([(208, 105), (148, 128)], fill=(255, 30, 30), width=6)
                draw.ellipse([60, 125, 96, 155], fill=(255, 30, 30))
                draw.ellipse([160, 125, 196, 155], fill=(255, 30, 30))
                draw.arc([118, 80, 128, 96], start=0, end=180, fill=WHITE, width=2)
                draw.arc([128, 80, 138, 96], start=180, end=360, fill=WHITE, width=2)
                draw.rectangle([112, 148, 144, 166], fill=(255, 30, 30))

        # NEUTRAL
        else:
            draw.rounded_rectangle([62, 110, 94, 160], radius=16, fill=CYAN)
            draw.rounded_rectangle([162, 110, 194, 160], radius=16, fill=CYAN)
            draw.ellipse([70, 118, 82, 130], fill=WHITE)
            draw.ellipse([170, 118, 182, 130], fill=WHITE)
            draw.line([(120, 156), (136, 156)], fill=CYAN, width=3)

    # ------------------ THEME 2: SCI-FI HUD ------------------
    def _paint_sci_fi_hud(self, draw: ImageDraw.ImageDraw, p: PrimaryEmotion, s: SubEmotion, t: float):
        draw.line([(20, 40), (20, 20), (40, 20)], fill=(0, 180, 255), width=2)
        draw.line([(236, 40), (236, 20), (216, 20)], fill=(0, 180, 255), width=2)
        draw.line([(20, 216), (20, 236), (40, 236)], fill=(0, 180, 255), width=2)
        draw.line([(236, 216), (236, 236), (216, 236)], fill=(0, 180, 255), width=2)
        
        color = (0, 240, 255)
        if p == PrimaryEmotion.JOY:
            color = (50, 255, 160)
        elif p == PrimaryEmotion.ANGER:
            color = (255, 60, 60)
        elif p == PrimaryEmotion.FEAR:
            color = (255, 160, 30)
        elif p == PrimaryEmotion.SADNESS:
            color = (190, 110, 255)

        for cx in [78, 178]:
            draw.ellipse([cx-36, 102, cx+36, 174], outline=color, width=2)
            draw.arc([cx-42, 96, cx+42, 180], start=30, end=150, fill=color, width=3)
            draw.arc([cx-42, 96, cx+42, 180], start=210, end=330, fill=color, width=3)
            
            if p == PrimaryEmotion.JOY:
                draw.arc([cx-24, 114, cx+24, 162], start=20, end=160, fill=(245, 250, 255), width=4)
            elif p == PrimaryEmotion.SURPRISE:
                draw.ellipse([cx-26, 112, cx+26, 164], fill=(245, 250, 255))
            elif p == PrimaryEmotion.ANGER:
                draw.polygon([(cx-26, 134), (cx+26, 138), (cx, 144)], fill=(255, 60, 60))
            elif p == PrimaryEmotion.SADNESS:
                draw.line([(cx-24, 142), (cx+24, 142)], fill=(190, 110, 255), width=3)
            else:
                draw.ellipse([cx-14, 124, cx+14, 152], fill=(245, 250, 255))
                draw.line([(cx-20, 138), (cx+20, 138)], fill=color, width=1)

        # Center waveform
        for i in range(11):
            x = 108 + i * 4
            h = 6 + 10 * math.sin(i * 0.8 + t * 5)
            draw.line([(x, 138 - h), (x, 138 + h)], fill=color, width=2)

    # ------------------ THEME 3: RETRO PIXEL ------------------
    def _paint_retro_pixel(self, draw: ImageDraw.ImageDraw, p: PrimaryEmotion, s: SubEmotion, t: float):
        pix_color = (80, 255, 120)
        def pbox(gx, gy, w=1, h=1, col=pix_color):
            ox = 32 + gx * 12
            oy = 40 + gy * 12
            draw.rectangle([ox, oy, ox + w*12 - 2, oy + h*12 - 2], fill=col)

        if p == PrimaryEmotion.JOY:
            for gx, gy in [(2, 8), (3, 7), (4, 7), (5, 8), (10, 8), (11, 7), (12, 7), (13, 8)]:
                pbox(gx, gy, 1, 1)
            for gx in range(6, 10):
                pbox(gx, 10, 1, 1)
            pbox(5, 9, 1, 1)
            pbox(10, 9, 1, 1)

        elif p == PrimaryEmotion.CURIOSITY:
            pbox(2, 7, 3, 2)
            pbox(11, 6, 2, 3)
            pbox(14, 5, 1, 1)
            pbox(7, 10, 2, 1)

        elif p == PrimaryEmotion.SURPRISE:
            pbox(2, 6, 4, 4)
            pbox(10, 6, 4, 4)
            pbox(7, 10, 2, 2)

        elif p == PrimaryEmotion.SADNESS:
            for gx, gy in [(2, 7), (3, 8), (4, 8), (5, 7), (10, 7), (11, 8), (12, 8), (13, 7)]:
                pbox(gx, gy, 1, 1)
            pbox(7, 11, 2, 1)

        elif p == PrimaryEmotion.ANGER:
            pbox(2, 6, 4, 2)
            pbox(10, 7, 4, 2)
            pbox(6, 11, 4, 1)

        else: # NEUTRAL
            pbox(3, 7, 2, 3)
            pbox(11, 7, 2, 3)
            pbox(7, 11, 2, 1)

    # ------------------ THEME 4: MINIMALIST ------------------
    def _paint_minimalist(self, draw: ImageDraw.ImageDraw, p: PrimaryEmotion, s: SubEmotion, t: float):
        color = (220, 245, 255)
        if p == PrimaryEmotion.JOY:
            draw.chord([54, 110, 102, 158], start=180, end=360, fill=color)
            draw.chord([154, 110, 202, 158], start=180, end=360, fill=color)
        elif p == PrimaryEmotion.CURIOSITY:
            draw.ellipse([58, 120, 98, 160], fill=color)
            draw.ellipse([154, 110, 202, 166], fill=color)
        elif p == PrimaryEmotion.SURPRISE:
            draw.ellipse([50, 108, 106, 164], fill=color)
            draw.ellipse([150, 108, 206, 164], fill=color)
        elif p == PrimaryEmotion.SADNESS:
            draw.chord([54, 118, 102, 166], start=0, end=180, fill=(160, 180, 220))
            draw.chord([154, 118, 202, 166], start=0, end=180, fill=(160, 180, 220))
        elif p == PrimaryEmotion.ANGER:
            draw.polygon([(52, 120), (102, 140), (70, 160)], fill=(255, 140, 140))
            draw.polygon([(204, 120), (154, 140), (186, 160)], fill=(255, 140, 140))
        else: # NEUTRAL
            draw.ellipse([60, 115, 96, 155], fill=color)
            draw.ellipse([160, 115, 196, 155], fill=color)

    @property
    def current_label(self) -> str:
        icons = {
            SubEmotion.CHEERFUL: "(◠‿◠) 欣喜开朗",
            SubEmotion.ECSTATIC: "(> <) 狂喜兴奋",
            SubEmotion.CONTENT: "(˘‿˘) 惬意满足",
            SubEmotion.AFFECTIONATE: "(♡ ♡) 喜爱亲昵",
            SubEmotion.PLAYFUL: "(^_-) 顽皮嬉戏",
            SubEmotion.INQUISITIVE: "(•ิ_•ิ)? 探究好奇",
            SubEmotion.FOCUS_SCAN: "[⊙_⊙] 专注扫描",
            SubEmotion.AWE_WONDER: "(★_★) 惊奇赞叹",
            SubEmotion.EUREKA: "(!_!) 灵光一闪",
            SubEmotion.STARTLED: "(O_O) 大吃一惊",
            SubEmotion.BLANK_DOTS: "(·_·) 呆愣错愕",
            SubEmotion.PUZZLED: "(?_o) 困惑迷茫",
            SubEmotion.POUTY: "(｡•́︿•̀｡) 委屈失落",
            SubEmotion.DIZZY_FALLEN: "(@_@) 跌倒眩晕",
            SubEmotion.SLEEPY: "(-_-) zZ 困倦疲乏",
            SubEmotion.WEEPING: "(T_T) 心碎悲伤",
            SubEmotion.PANICKED: "(>_<) 慌张惊恐",
            SubEmotion.ALERT_GUARD: "(°ロ°) 高度戒备",
            SubEmotion.TIMID: "(•.•) 怯懦畏缩",
            SubEmotion.ANNOYED: "(¬_¬) 恼怒不爽",
            SubEmotion.RESISTING: "(｀へ´) 坚韧顽抗",
            SubEmotion.OVERHEATED: "(♨_♨) 过载暴躁",
            SubEmotion.IDLE_CALM: "(||) 平静稳态",
        }
        return icons.get(self.sub_emotion, f"{self.primary_emotion.name}: {self.sub_emotion.value}")
