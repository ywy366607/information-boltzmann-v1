"""
Unit tests for the Robot Pet Dynamic Visor Facial Expression & Affect Hierarchy System.
Verifies the 6 Primary Emotions, hierarchical sub-branches, 4 visual themes,
and GPU texture upload via MuJoCo.
"""

import pytest
import numpy as np
import mujoco
from information_boltzmann.sandbox.expression_system import (
    ExpressionSystem,
    ThemeStyle,
    PrimaryEmotion,
    SubEmotion,
    ExpressionToken
)
from information_boltzmann.sandbox.ether_sandbox import EtherSandbox

def test_all_emotions_and_themes_render():
    """Verify that all combinations of (Theme, Primary, Sub) render valid 256x256 RGB images."""
    sb = EtherSandbox()
    expr = sb.expressions
    
    # Check all 4 themes
    for theme in ThemeStyle:
        expr.set_theme(theme)
        assert expr.theme_style == theme
        
        # Check all 6 primary emotions + neutral
        for primary in PrimaryEmotion:
            expr.set_emotion_explicit(primary)
            assert expr.primary_emotion == primary
            
            # Verify canvas array shape and validity
            expr._render_current_face()
            assert expr.canvas_arr.shape == (256, 256, 3)
            assert expr.canvas_arr.dtype == np.uint8
            # Must not be completely black (should have face graphics and borders)
            assert expr.canvas_arr.max() > 0

def test_sub_emotions_coverage():
    """Verify that all specific sub-emotions can be explicitly set and produce non-empty art."""
    sb = EtherSandbox()
    expr = sb.expressions
    
    for sub in SubEmotion:
        # Determine parent primary
        p = PrimaryEmotion.NEUTRAL
        if sub.value in ["CHEERFUL", "ECSTATIC", "CONTENT", "AFFECTIONATE", "PLAYFUL"]:
            p = PrimaryEmotion.JOY
        elif sub.value in ["INQUISITIVE", "FOCUS_SCAN", "AWE_WONDER", "EUREKA"]:
            p = PrimaryEmotion.CURIOSITY
        elif sub.value in ["STARTLED", "BLANK_DOTS", "PUZZLED"]:
            p = PrimaryEmotion.SURPRISE
        elif sub.value in ["POUTY", "DIZZY_FALLEN", "SLEEPY", "WEEPING"]:
            p = PrimaryEmotion.SADNESS
        elif sub.value in ["PANICKED", "ALERT_GUARD", "TIMID"]:
            p = PrimaryEmotion.FEAR
        elif sub.value in ["ANNOYED", "RESISTING", "OVERHEATED"]:
            p = PrimaryEmotion.ANGER
            
        expr.set_emotion_explicit(p, sub)
        expr._render_current_face()
        assert expr.canvas_arr.shape == (256, 256, 3)
        assert len(expr.current_label) > 0

def test_texture_upload_to_renderer():
    """Verify that upload_to_renderer successfully writes to MjModel.tex_data and calls mjr_uploadTexture."""
    sb = EtherSandbox()
    expr = sb.expressions
    r = mujoco.Renderer(sb.model, 64, 64)
    
    # Upload cheerful face
    expr.set_emotion_explicit(PrimaryEmotion.JOY, SubEmotion.CHEERFUL)
    expr.upload_to_renderer(r._mjr_context)
    
    # Verify tex_data was modified
    tex_slice = sb.model.tex_data[expr.tex_adr : expr.tex_adr + 256*256*3]
    assert np.array_equal(tex_slice, expr.canvas_arr.flatten())
    assert tex_slice.max() > 0

def test_active_inference_affect_mapping():
    """Verify autonomous selection logic under various embodied situations."""
    sb = EtherSandbox()
    expr = sb.expressions
    
    # 1. Acoustic wave shock -> SURPRISE (Startled)
    tok1 = expr.select_autonomous_token(
        dt=0.002, audio_amplitude=60.0, pitch_rad=0.0, roll_rad=0.0,
        forward_speed=0.0, steering_rate=0.0, vfe=0.1, curiosity=0.1, risk=0.0
    )
    assert expr.primary_emotion == PrimaryEmotion.SURPRISE
    assert expr.sub_emotion == SubEmotion.STARTLED
    assert tok1 == ExpressionToken.SURPRISED
    
    # Clear cooldown
    expr.shock_cooldown = 0.0
    
    # 2. Catastrophic fall -> SADNESS (Dizzy Fallen)
    tok2 = expr.select_autonomous_token(
        dt=0.002, audio_amplitude=0.0, pitch_rad=1.0, roll_rad=0.0,
        forward_speed=0.0, steering_rate=0.0, vfe=0.5, curiosity=0.1, risk=0.9, is_fallen=True
    )
    assert expr.primary_emotion == PrimaryEmotion.SADNESS
    assert expr.sub_emotion == SubEmotion.DIZZY_FALLEN
    assert tok2 == ExpressionToken.SLEEPY
    
    # 3. High curiosity -> CURIOSITY
    tok3 = expr.select_autonomous_token(
        dt=0.002, audio_amplitude=0.0, pitch_rad=0.0, roll_rad=0.0,
        forward_speed=0.05, steering_rate=0.3, vfe=0.05, curiosity=0.35, risk=0.0
    )
    assert expr.primary_emotion == PrimaryEmotion.CURIOSITY
    assert tok3 == ExpressionToken.CURIOUS
    
    # 4. Confident forward walk -> JOY (Cheerful / Ecstatic)
    tok4 = expr.select_autonomous_token(
        dt=0.002, audio_amplitude=0.0, pitch_rad=0.02, roll_rad=0.01,
        forward_speed=0.18, steering_rate=0.0, vfe=0.02, curiosity=0.1, risk=0.0
    )
    assert expr.primary_emotion == PrimaryEmotion.JOY
    assert expr.sub_emotion == SubEmotion.CHEERFUL
    assert tok4 == ExpressionToken.HAPPY
