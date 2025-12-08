import os
from typing import Union
import json
import time
import threading
import math
import random
from datetime import datetime
from pathlib import Path
import shutil

import numpy as np
import sounddevice as sd
import soundfile as sf
import pygame
from openai import OpenAI

try:
    from scipy import signal as scipy_signal
except ImportError:  # pragma: no cover - SciPy is optional
    scipy_signal = None
    print("SciPy not available; using NumPy fallbacks for audio effects.")

client = OpenAI()

BASE_DIR = Path(__file__).resolve().parent
INPUT_WAV = BASE_DIR / "input.wav"
RECORDINGS_DIR = BASE_DIR / "recordings"
KEEP_RECORDINGS = os.getenv("KEEP_RECORDINGS", "0") == "1"
INPUT_DEVICE_PREF = os.getenv("AUDIO_INPUT_DEVICE")

WIDTH = 640
HEIGHT = 480

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Gerty style avatar")
clock = pygame.time.Clock()
pygame.mixer.init()

FONT = pygame.font.SysFont("Arial", 24)
SMALL_FONT = pygame.font.SysFont("Arial", 18)
GREETING_MESSAGES = [
    "Hello, I'm Zeni. How may I assist you today?",
    "I am still here. I have not moved. I keep listening.",
    "The system says you are present. Is this true?",
    "Just checking in—ready whenever you are.",
    "Speak. I am already watching.",
]

EMOTIONS = [
    "neutral",
    "happy",
    "excited",
    "curious",
    "confused",
    "sad",
    "concerned",
    "surprised",
    "thinking",
]
EMOTION_SET = set(EMOTIONS)
EMOTION_KEY_MAP = {
    "1": "neutral",
    "2": "happy",
    "3": "excited",
    "4": "curious",
    "5": "confused",
    "6": "sad",
    "7": "concerned",
    "8": "surprised",
    "9": "thinking",
}
EMOTION_KEYS = dict(EMOTION_KEY_MAP)
HOTKEY_PAIRS = [f"{key}:{value}" for key, value in EMOTION_KEY_MAP.items()]
HOTKEY_LINES = []
line = []
for pair in HOTKEY_PAIRS:
    line.append(pair)
    if len(line) == 3:
        HOTKEY_LINES.append("  ".join(line))
        line = []
if line:
    HOTKEY_LINES.append("  ".join(line))
DEFAULT_STATUS = "Hold space to talk"


def _linear_resample(data: np.ndarray, num_samples: int) -> np.ndarray:
    """Simple linear resampling fallback for 1D/2D audio."""
    data = np.asarray(data)
    if num_samples <= 0:
        return np.empty((0,) + data.shape[1:], dtype=data.dtype)
    if data.size == 0:
        return np.zeros((num_samples,) + data.shape[1:], dtype=data.dtype)
    axis_len = data.shape[0]
    if axis_len == 1:
        repeated = np.repeat(data, num_samples, axis=0)
        return repeated[:num_samples]

    orig_idx = np.linspace(0.0, axis_len - 1, axis_len)
    target_idx = np.linspace(0.0, axis_len - 1, num_samples)
    if data.ndim == 1:
        return np.interp(target_idx, orig_idx, data).astype(data.dtype)

    channels = [np.interp(target_idx, orig_idx, data[:, ch]) for ch in range(data.shape[1])]
    return np.stack(channels, axis=-1).astype(data.dtype)


def resample_signal(data: np.ndarray, num_samples: int) -> np.ndarray:
    if scipy_signal is not None:
        return scipy_signal.resample(data, num_samples)
    return _linear_resample(data, num_samples)


def fft_convolve_signal(data: np.ndarray, kernel: np.ndarray, mode: str = "full") -> np.ndarray:
    if scipy_signal is not None:
        return scipy_signal.fftconvolve(data, kernel, mode=mode)
    data = np.asarray(data)
    kernel = np.asarray(kernel)
    if data.ndim == 1:
        return np.convolve(data, kernel, mode=mode)
    outputs = [np.convolve(data[:, ch], kernel, mode=mode) for ch in range(data.shape[1])]
    return np.stack(outputs, axis=-1)


def configure_input_device():
    if not INPUT_DEVICE_PREF:
        return
    try:
        devices = sd.query_devices()
    except Exception as exc:
        print("Could not query audio devices:", exc)
        return
    target_index = None
    if INPUT_DEVICE_PREF.isdigit():
        idx = int(INPUT_DEVICE_PREF)
        if 0 <= idx < len(devices) and devices[idx]["max_input_channels"] > 0:
            target_index = idx
    if target_index is None:
        lower_pref = INPUT_DEVICE_PREF.lower()
        for idx, device in enumerate(devices):
            if device["max_input_channels"] <= 0:
                continue
            if lower_pref in device["name"].lower():
                target_index = idx
                break
    if target_index is None:
        print(f"Audio input preference {INPUT_DEVICE_PREF!r} not found; using default.")
        return
    default_output = None
    if sd.default.device is not None:
        try:
            _, default_output = sd.default.device
        except Exception:
            default_output = None
    sd.default.device = (target_index, default_output)
    print(f"Using audio input device {target_index}: {devices[target_index]['name']}")


def print_audio_devices():
    try:
        devices = sd.query_devices()
    except Exception as exc:
        print("Unable to list audio devices:", exc)
        return
    default = sd.default.device
    default_input = None
    default_output = None
    if default is not None:
        try:
            default_input, default_output = default
        except Exception:
            default_input = default_output = None
    print("=== Audio devices ===")
    for idx, device in enumerate(devices):
        roles = []
        if device["max_input_channels"] > 0:
            roles.append("IN")
        if device["max_output_channels"] > 0:
            roles.append("OUT")
        marker = ""
        if default_input is not None and idx == default_input:
            marker = "*IN* "
        elif default_output is not None and idx == default_output:
            marker = "*OUT* "
        print(
            f"{marker}{idx}: {device['name']} "
            f"[{'/'.join(roles) or 'none'}] "
            f"default_sr={device['default_samplerate']}"
        )
    print("=====================")


def classify_voice_mood(peak):
    if peak is None:
        return None
    if peak >= 0.45:
        return "alert"
    if peak <= 0.18:
        return "curious"
    return None

class Face:
    EXPRESSION_STYLES = {
        "neutral":   {"mouth": "flat", "mouth_width": 90, "eye_drop": 0,  "eye_scale": 1.0},
        "happy":     {"mouth": "smile", "mouth_width": 95, "eye_drop": -2, "eye_scale": 1.0},
        "excited":   {"mouth": "smile", "mouth_width": 125, "eye_drop": -4, "eye_scale": 1.15},
        "curious":   {"mouth": "smirk", "mouth_width": 85, "eye_drop": -4, "eye_scale": 0.95},
        "confused":  {"mouth": "wobble", "mouth_width": 90, "eye_drop": 2,  "eye_scale": 0.95},
        "sad":       {"mouth": "frown", "mouth_width": 100, "eye_drop": 6,  "eye_scale": 0.88},
        "concerned": {"mouth": "flat", "mouth_width": 95, "eye_drop": 3,  "eye_scale": 0.92},
        "surprised": {"mouth": "o",     "mouth_width": 80, "eye_drop": -6, "eye_scale": 1.15},
        "thinking":  {
            "mouth": "thinking",
            "mouth_width": 80,
            "eye_drop": -2,
            "eye_scale": 1.05,
        },
    }

    def __init__(self):
        self.expression = "neutral"
        self.idle_expression = "neutral"
        self.talking = False

        self.blink_timer = 0.0
        self.blink_interval = 3.0
        self.blink_length = 0.15
        self.blinking = False

        self.talk_timer = 0.0
        self.talk_phase = random.random() * math.tau

        self.talking_sequence = []
        self.talking_sequence_index = 0
        self.talking_sequence_timer = 0.0

        self.manual_expression = None
        self.manual_timer = 0.0

        self.last_talk_mood = None
        self.current_talk_mood = None
        self.head_offset = [0.0, 0.0]
        self.head_target = [0.0, 0.0]
        self.head_timer = 0.0
        self.idle_sad_elapsed = 0.0
        self.idle_sad_active = False
        self.idle_sad_timer = 0.0
        self.next_idle_sad = random.uniform(5.0, 7.0)
        self.signal_timer = random.uniform(8.0, 15.0)
        self.signal_active = False
        self.signal_duration = 0.0
        self.glitch_timer = random.uniform(3.0, 8.0)
        self.glitch_active = False
        self.glitch_type = None
        self.glitch_duration = 0.0
        self.glitch_intensity = 0.0
        self.scan_line_offset = 0.0
        self.noise_timer = 0.0
        self.voice_hint = None
        self.previous_frame = None

    def set_expression(self, expr):
        if expr not in EMOTION_SET:
            expr = "happy"
        self.idle_expression = expr
        if not self.talking:
            self.expression = expr

    def set_talking(self, talking):
        if talking and not self.talking:
            self.talking = True
            self._start_talking_sequence()
        elif not talking and self.talking:
            self.talking = False
            self.talk_timer = 0.0
            self.talking_sequence = []
            self.talking_sequence_index = 0
            self.talking_sequence_timer = 0.0
            self.current_talk_mood = None
            self.expression = self.idle_expression
        else:
            self.talking = talking

    def set_voice_hint(self, mood):
        self.voice_hint = mood
        if mood == "alert":
            self.preview_expression("surprised", duration=1.5)
        elif mood == "curious":
            self.preview_expression("curious", duration=1.5)

    def _pick_talk_mood(self):
        # pick a mood for this whole talking turn
        if self.voice_hint == "alert":
            mood = "excited"
        elif self.voice_hint == "curious":
            mood = "curious"
        else:
            candidates = list(EMOTIONS)
            if self.last_talk_mood in candidates and len(candidates) > 1:
                candidates.remove(self.last_talk_mood)
            mood = random.choice(candidates)
        self.voice_hint = None
        self.last_talk_mood = mood
        return mood

    def _start_talking_sequence(self):
        self.talk_timer = 0.0
        self.talk_phase = random.random() * math.tau

        mood = self._pick_talk_mood()
        self.current_talk_mood = mood

        self.talking_sequence = self._generate_talking_sequence(mood)
        self.talking_sequence_index = 0
        self.talking_sequence_timer = 0.0

        if self.talking_sequence:
            self.expression = self.talking_sequence[0][0]

    def _generate_talking_sequence(self, mood):
        # a small library of mood specific patterns
        pattern_map = {
            "neutral":   ["neutral", "thinking", "neutral"],
            "happy":     ["happy", "excited", "happy", "curious"],
            "excited":   ["excited", "happy", "excited", "surprised"],
            "curious":   ["curious", "thinking", "curious"],
            "confused":  ["confused", "thinking", "concerned", "confused"],
            "sad":       ["sad", "concerned", "sad"],
            "concerned": ["concerned", "thinking", "concerned"],
            "surprised": ["surprised", "excited", "surprised"],
            "thinking":  ["thinking", "curious", "thinking"],
        }

        base_pattern = pattern_map.get(mood, ["neutral", "thinking", "neutral"])

        loops = random.randint(2, 4)
        sequence = []

        for _ in range(loops):
            for expr in base_pattern:
                duration = random.uniform(0.85, 1.6)
                sequence.append((expr, duration))

        return sequence

    def preview_expression(self, expr, duration=1.5):
        if expr not in EMOTION_SET:
            return False
        if self.talking:
            return False
        self.manual_expression = expr
        self.manual_timer = duration
        self.expression = expr
        return True

    def update(self, dt):
        if self.talking:
            self.talk_timer += dt
            self._advance_talking_expression(dt)
        else:
            self.talk_timer = max(0.0, self.talk_timer - dt * 0.4)

        self.blink_timer += dt
        if not self.blinking and self.blink_timer > self.blink_interval:
            self.blinking = True
            self.blink_timer = 0.0
            self.blink_interval = random.uniform(2.5, 4.5)
        if self.blinking and self.blink_timer > self.blink_length:
            self.blinking = False
            self.blink_timer = 0.0

        if self.manual_expression:
            self.manual_timer -= dt
            if self.manual_timer <= 0:
                self.manual_expression = None
                if not self.talking:
                    self.expression = self.idle_expression

        if not self.talking and not self.manual_expression:
            if self.idle_sad_active:
                self.idle_sad_timer -= dt
                if self.idle_sad_timer <= 0:
                    self.idle_sad_active = False
                    self.idle_sad_elapsed = 0.0
                    self.next_idle_sad = random.uniform(5.0, 7.0)
                    self.expression = "happy" if self.idle_expression in {"neutral", "happy"} else self.idle_expression
            else:
                self.idle_sad_elapsed += dt
                if self.expression in {"neutral", "happy"} and self.idle_sad_elapsed >= self.next_idle_sad:
                    self.idle_sad_active = True
                    self.idle_sad_timer = random.uniform(0.6, 1.1)
                    self.expression = "sad"
        else:
            self.idle_sad_active = False
            self.idle_sad_elapsed = 0.0

        self.signal_timer -= dt
        if self.signal_active:
            self.signal_duration -= dt
            if self.signal_duration <= 0:
                self.signal_active = False
                self.signal_timer = random.uniform(8.0, 15.0)
        elif (
            not self.talking
            and not self.manual_expression
            and self.signal_timer <= 0
        ):
            self._trigger_signal_loss()
        
        self.glitch_timer -= dt
        if self.glitch_active:
            self.glitch_duration -= dt
            if self.glitch_duration <= 0:
                self.glitch_active = False
                self.glitch_type = None
                self.glitch_timer = random.uniform(3.0, 8.0)
        elif self.glitch_timer <= 0:
            self._trigger_glitch()
        
        self.noise_timer += dt

        self._update_head_motion(dt)

    def _advance_talking_expression(self, dt):
        if not self.talking_sequence:
            return
        if self.talking_sequence_index >= len(self.talking_sequence):
            return

        self.talking_sequence_timer += dt
        current, dur = self.talking_sequence[self.talking_sequence_index]

        if self.talking_sequence_timer >= dur:
            self.talking_sequence_timer -= dur
            self.talking_sequence_index += 1

            if self.talking_sequence_index >= len(self.talking_sequence):
                # done with this talk animation
                self.expression = self.idle_expression
            else:
                self.expression = self.talking_sequence[self.talking_sequence_index][0]

    def _update_head_motion(self, dt):
        jitter = 5 if self.talking else 2
        cooldown = 0.25 if self.talking else 0.7
        settle = 9.0 if self.talking else 4.5

        self.head_timer -= dt
        if self.head_timer <= 0:
            self.head_target = [
                random.uniform(-jitter, jitter),
                random.uniform(-jitter, jitter),
            ]
            self.head_timer = random.uniform(cooldown, cooldown + 0.25)

        for i in (0, 1):
            delta = self.head_target[i] - self.head_offset[i]
            self.head_offset[i] += delta * min(1.0, dt * settle)

    def _trigger_signal_loss(self):
        self.signal_active = True
        self.signal_duration = random.uniform(0.8, 1.2)
        self.expression = "surprised"
    
    def _trigger_glitch(self):
        self.glitch_active = True
        self.glitch_duration = random.uniform(0.15, 0.6)
        self.glitch_intensity = random.uniform(0.4, 1.0)
        glitch_types = [
            "static",
            "scanline",
            "chromatic",
            "distort",
            "flicker",
            "corrupt",
            "timeslice",
        ]
        self.glitch_type = random.choice(glitch_types)
        if self.glitch_type == "timeslice":
            self.glitch_duration = random.uniform(0.45, 0.6)

    def draw(self, surf):
        base_bg = (10, 10, 20)
        
        if self.signal_active:
            offset = random.randint(-10, 10)
            surf.fill((10 + offset, 10, 20 + offset))
        elif self.glitch_active and self.glitch_type in ["flicker", "corrupt"]:
            flicker = int(random.random() * 30 * self.glitch_intensity)
            surf.fill((base_bg[0] + flicker, base_bg[1] + flicker, base_bg[2] + flicker))
        else:
            surf.fill(base_bg)
        
        if self.glitch_active:
            self._apply_glitch_effects(surf)
        style = self.EXPRESSION_STYLES.get(self.expression, self.EXPRESSION_STYLES["neutral"])
        center = (
            WIDTH // 2 + int(self.head_offset[0]),
            HEIGHT // 2 + int(self.head_offset[1]),
        )
        radius = 140

        base = 28
        pulse_speed = 3.2 if self.talking else 1.6
        pulse = base + int(22 * (0.5 + 0.5 * math.sin(self.talk_timer * pulse_speed + self.talk_phase)))
        pulse = max(20, min(90, pulse))

        if self.signal_active:
            distort = random.randint(-6, 6)
            distorted_radius = radius + distort
            pygame.draw.circle(surf, (pulse, pulse, pulse + 20), center, max(120, distorted_radius))
            smear_width = random.randint(4, 10)
            smear = pygame.Rect(center[0] - radius, center[1] - smear_width // 2, radius * 2, smear_width)
            pygame.draw.rect(surf, (pulse + 20, pulse, pulse), smear, border_radius=smear_width // 2)
        else:
            pygame.draw.circle(surf, (pulse, pulse, pulse + 20), center, radius)

        if self.glitch_active and self.glitch_type == "distort":
            distorted_center = (
                center[0] + random.randint(-8, 8),
                center[1] + random.randint(-8, 8)
            )
            self._draw_eyes(surf, distorted_center, style)
            self._draw_mouth(surf, distorted_center, style)
        else:
            self._draw_eyes(surf, center, style)
            self._draw_mouth(surf, center, style)
        
        if self.glitch_active:
            self._apply_post_glitch_effects(surf)
        
        self._apply_ambient_noise(surf)
        self.previous_frame = surf.copy()

    def _apply_glitch_effects(self, surf):
        """Apply pre-render glitch effects to the background"""
        if self.glitch_type == "static":
            for _ in range(int(150 * self.glitch_intensity)):
                x = random.randint(0, WIDTH)
                y = random.randint(0, HEIGHT)
                size = random.randint(1, 4)
                brightness = random.randint(80, 200)
                color = (brightness, brightness, brightness)
                pygame.draw.rect(surf, color, pygame.Rect(x, y, size, size))
        
        elif self.glitch_type == "scanline":
            self.scan_line_offset += 3
            if self.scan_line_offset > HEIGHT:
                self.scan_line_offset = 0
            for y in range(0, HEIGHT, 3):
                offset_y = (y + int(self.scan_line_offset)) % HEIGHT
                alpha = int(100 * self.glitch_intensity)
                pygame.draw.line(surf, (alpha, alpha, alpha), (0, offset_y), (WIDTH, offset_y), 1)
    
    def _apply_post_glitch_effects(self, surf):
        """Apply post-render glitch effects over the face"""
        if self.glitch_type == "chromatic":
            shift = int(6 * self.glitch_intensity)
            temp_surf = surf.copy()
            red_channel = pygame.Surface((WIDTH, HEIGHT))
            red_channel.fill((0, 0, 0))
            red_channel.blit(temp_surf, (shift, 0))
            red_channel.set_colorkey((0, 0, 0))
            surf.blit(red_channel, (-shift, 0), special_flags=pygame.BLEND_RGB_ADD)
        
        elif self.glitch_type == "corrupt":
            num_bars = int(8 * self.glitch_intensity)
            for _ in range(num_bars):
                bar_y = random.randint(0, HEIGHT - 20)
                bar_height = random.randint(2, 12)
                shift_x = random.randint(-25, 25)
                
                if 0 <= bar_y < HEIGHT and bar_height > 0:
                    try:
                        bar_rect = pygame.Rect(0, bar_y, WIDTH, bar_height)
                        bar_section = surf.subsurface(bar_rect).copy()
                        new_x = max(-WIDTH//2, min(WIDTH//2, shift_x))
                        surf.blit(bar_section, (new_x, bar_y))
                    except ValueError:
                        pass
        
        elif self.glitch_type == "flicker":
            if random.random() < 0.3:
                dark_overlay = pygame.Surface((WIDTH, HEIGHT))
                dark_overlay.fill((0, 0, 0))
                dark_overlay.set_alpha(int(180 * self.glitch_intensity))
                surf.blit(dark_overlay, (0, 0))
        
        elif self.glitch_type == "timeslice":
            if self.previous_frame is not None:
                ghost = self.previous_frame.copy()
                offset = int(8 + 12 * self.glitch_intensity)
                alpha = int(150 * min(1.0, 0.5 + 0.5 * self.glitch_intensity))
                ghost.set_alpha(alpha)
                surf.blit(ghost, (offset, 0))
                ghost.set_alpha(int(alpha * 0.7))
                surf.blit(ghost, (-offset // 2, 0))

                num_slices = max(4, int(8 * self.glitch_intensity))
                for _ in range(num_slices):
                    slice_width = random.randint(15, 50)
                    x = random.randint(0, max(0, WIDTH - slice_width))
                    shift = random.randint(-25, 25)
                    dest_x = max(0, min(WIDTH - slice_width, x + shift))
                    slice_rect = pygame.Rect(x, 0, slice_width, HEIGHT)
                    surf.blit(self.previous_frame, (dest_x, 0), slice_rect)
        
        if self.glitch_active and random.random() < 0.4:
            num_zaps = random.randint(1, 4)
            for _ in range(num_zaps):
                x1, y1 = random.randint(0, WIDTH), random.randint(0, HEIGHT)
                x2, y2 = x1 + random.randint(-40, 40), y1 + random.randint(-40, 40)
                x2 = max(0, min(WIDTH, x2))
                y2 = max(0, min(HEIGHT, y2))
                
                brightness = random.randint(150, 255)
                zap_color = (brightness, brightness, min(255, brightness + 20))
                
                mid_x = (x1 + x2) // 2 + random.randint(-15, 15)
                mid_y = (y1 + y2) // 2 + random.randint(-15, 15)
                
                pygame.draw.line(surf, zap_color, (x1, y1), (mid_x, mid_y), 2)
                pygame.draw.line(surf, zap_color, (mid_x, mid_y), (x2, y2), 2)
    
    def _apply_ambient_noise(self, surf):
        """Subtle persistent visual artifacts to make it feel unstable"""
        noise_intensity = 0.3 + 0.15 * math.sin(self.noise_timer * 0.8)
        
        num_artifacts = int(25 * noise_intensity)
        for _ in range(num_artifacts):
            x = random.randint(0, WIDTH)
            y = random.randint(0, HEIGHT)
            brightness = random.randint(40, 100)
            pygame.draw.circle(surf, (brightness, brightness, brightness), (x, y), 1)
        
        if random.random() < 0.05:
            edge = random.choice(['top', 'bottom', 'left', 'right'])
            length = random.randint(20, 80)
            thickness = random.randint(1, 3)
            brightness = random.randint(100, 180)
            color = (brightness, brightness, brightness + 20)
            
            if edge == 'top':
                x = random.randint(0, WIDTH - length)
                pygame.draw.line(surf, color, (x, 0), (x + length, 0), thickness)
            elif edge == 'bottom':
                x = random.randint(0, WIDTH - length)
                pygame.draw.line(surf, color, (x, HEIGHT - 1), (x + length, HEIGHT - 1), thickness)
            elif edge == 'left':
                y = random.randint(0, HEIGHT - length)
                pygame.draw.line(surf, color, (0, y), (0, y + length), thickness)
            elif edge == 'right':
                y = random.randint(0, HEIGHT - length)
                pygame.draw.line(surf, color, (WIDTH - 1, y), (WIDTH - 1, y + length), thickness)
        
        if random.random() < 0.02:
            num_dead_pixels = random.randint(3, 10)
            for _ in range(num_dead_pixels):
                x = random.randint(0, WIDTH)
                y = random.randint(0, HEIGHT)
                color = random.choice([(0, 0, 0), (255, 255, 255), (200, 100, 100)])
                pygame.draw.rect(surf, color, pygame.Rect(x, y, 2, 2))
    
    def _draw_eyes(self, surf, center, style):
        eye_y = center[1] - 40 + style.get("eye_drop", 0)
        eye_offset_x = 45
        base = 18 * style.get("eye_scale", 1.0)

        if self.blinking:
            h = 4
            for sign in (-1, 1):
                x = center[0] + sign * eye_offset_x
                pygame.draw.rect(
                    surf,
                    (220, 220, 240),
                    pygame.Rect(x - base, eye_y - 2, base * 2, h),
                )
            return

        if self.expression == "thinking":
            color = (220, 220, 240)
            brow_y = eye_y - int(base * 2.2)
            brow_len = int(base * 2.0)
            tilt = int(base * 0.4)

            lx = center[0] - eye_offset_x
            pygame.draw.line(
                surf,
                color,
                (lx - brow_len // 2, brow_y + tilt),
                (lx + brow_len // 2, brow_y),
                3,
            )

            rx = center[0] + eye_offset_x
            pygame.draw.line(
                surf,
                color,
                (rx - brow_len // 2, brow_y),
                (rx + brow_len // 2, brow_y + tilt),
                3,
            )

        for sign in (-1, 1):
            x = center[0] + sign * eye_offset_x
            w = int(base * 0.9)
            h = int(base * 1.45)
            rect = pygame.Rect(x - w, eye_y - h, w * 2, h * 2)
            pygame.draw.ellipse(surf, (220, 220, 240), rect)

    def _talk_amount(self):
        if self.talking:
            base = 0.25 + 0.35 * (0.5 + 0.5 * math.sin(self.talk_timer * 4.2 + self.talk_phase))
            wobble = 0.08 * math.sin(self.talk_timer * 8.5 + self.talk_phase * 0.7)
            return max(0.08, base + wobble)
        return 0.08 + 0.03 * math.sin(self.talk_timer * 1.6 + self.talk_phase)

    def _draw_mouth(self, surf, center, style):
        m = style.get("mouth", "flat")
        width = style.get("mouth_width", 90)
        if self.talking:
            width *= 1.0 + 0.08 * math.sin(self.talk_timer * 5.3 + self.talk_phase)
        width = int(width)
        y = center[1] + 40
        talk = self._talk_amount()
        height = 35 + talk * 30
        color = (220, 220, 240)

        if m == "smile":
            rect = pygame.Rect(center[0] - width, y - height, width * 2, height * 2)
            pygame.draw.arc(surf, color, rect, 3.24, 6.1, 4)
        elif m == "frown":
            rect = pygame.Rect(center[0] - width, y - int(height * 0.8), width * 2, int(height * 1.8))
            pygame.draw.arc(surf, color, rect, 0.15, 2.99, 5)
        elif m == "line":
            pygame.draw.rect(
                surf,
                color,
                pygame.Rect(center[0] - width, y - 4, width * 2, 8),
                border_radius=4,
            )
        elif m == "o":
            r = max(14, int(18 + talk * 20))
            pygame.draw.circle(surf, color, (center[0], y + 10), r, 3)
        elif m == "smirk":
            # Simple asymmetric smile - one clean arc shifted to one side
            smirk_offset = int(width * 0.3)
            rect = pygame.Rect(
                center[0] - width + smirk_offset, 
                y - int(height * 0.6), 
                width * 2, 
                int(height * 1.4)
            )
            pygame.draw.arc(surf, color, rect, 3.3, 5.95, 5)
        elif m == "wobble":
            start = center[0] - width
            pts = []
            for i in range(12):
                t = i / 11.0
                x = start + t * width * 2
                yy = y + math.sin(self.talk_timer * 8.0 + t * math.pi * 2) * (8 + talk * 12)
                pts.append((x, yy))
            pygame.draw.lines(surf, color, False, pts, 4)
        elif m == "thinking":
            arc_h = int(height * 0.7)
            rect = pygame.Rect(center[0] - width, y - arc_h // 3, width * 2, arc_h)
            pygame.draw.arc(surf, color, rect, 3.4, 5.9, 4)
        else:
            arc_h = int(height * 0.6)
            rect = pygame.Rect(center[0] - width, y - arc_h, width * 2, arc_h * 2)
            pygame.draw.arc(surf, color, rect, 3.4, 5.8, 4)






class AudioRecorder:
    def __init__(self, samplerate=44100, channels=1, min_duration=0.3, release_tail=0.25):
        self.samplerate = samplerate
        self.channels = channels
        self.min_duration = min_duration
        self.release_tail = release_tail
        self.recording = False
        self._frames = []
        self._lock = threading.Lock()
        self._stream = None
        self._started_at = 0.0
        self._level = 0.0

    def _callback(self, indata, frames, time_info, status):
        if status:
            print("Audio status:", status)
        with self._lock:
            self._frames.append(indata.copy())
            current_peak = float(np.max(np.abs(indata))) if indata.size else 0.0
            self._level = max(current_peak, self._level * 0.8)

    def start(self):
        if self.recording:
            return
        self._frames = []
        self._level = 0.0
        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        self.recording = True
        self._started_at = time.time()
        print("Recording... (hold space)")

    def stop(self):
        if not self.recording:
            return None
        if self.release_tail > 0:
            time.sleep(self.release_tail)
        self._stream.stop()
        self._stream.close()
        self._stream = None
        self.recording = False
        print("Recording stopped")
        with self._lock:
            if not self._frames:
                return np.empty((0, self.channels), dtype="float32")
            data = np.concatenate(self._frames, axis=0)
            self._frames = []
        duration = data.shape[0] / float(self.samplerate)
        if duration < self.min_duration:
            print(f"Ignoring capture shorter than {self.min_duration}s")
            return np.empty((0, self.channels), dtype="float32")
        print(f"Captured audio duration: {duration:.2f}s")
        return data

    def level(self):
        with self._lock:
            return self._level


def _resample_audio(samples, orig_sr, target_sr):
    if orig_sr == target_sr:
        return samples
    duration = samples.shape[0] / float(orig_sr)
    target_length = max(1, int(duration * target_sr))
    old_indices = np.linspace(0.0, duration, num=samples.shape[0], endpoint=False)
    new_indices = np.linspace(0.0, duration, num=target_length, endpoint=False)
    resampled = np.interp(new_indices, old_indices, samples)
    return resampled.astype(samples.dtype)


def trim_silence(samples, samplerate, floor_ratio=0.05, padding=0.05):
    if samples.size == 0:
        return samples
    amplitudes = np.abs(samples)
    peak = float(np.max(amplitudes)) if amplitudes.size else 0.0
    if peak == 0.0:
        return samples
    threshold = max(0.01, peak * floor_ratio)
    indices = np.where(amplitudes >= threshold)[0]
    if indices.size == 0:
        return samples
    pad = int(padding * samplerate)
    start = max(0, indices[0] - pad)
    end = min(samples.shape[0], indices[-1] + pad)
    return samples[start:end]


def save_audio(
    data,
    filename=INPUT_WAV,
    samplerate=16000,
    target_samplerate=16000,
    keep_copy=KEEP_RECORDINGS,
):
    if data.size == 0:
        raise ValueError("No audio data captured")
    mono = data[:, 0] if data.ndim > 1 else data
    mono = mono.astype("float32")
    mono -= np.mean(mono)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    trimmed = trim_silence(mono, samplerate)
    if trimmed.size > samplerate * 0.1:
        mono = trimmed
    if peak > 0:
        mono = mono / peak * 0.98
    if samplerate != target_samplerate:
        mono = _resample_audio(mono, samplerate, target_samplerate)
        samplerate = target_samplerate
    padding = int(0.05 * samplerate)
    if padding > 0:
        mono = np.pad(mono, (0, padding), mode="constant")
    normalized = np.expand_dims(mono, axis=1)
    sf.write(str(filename), normalized, samplerate, subtype="PCM_16")
    info = {
        "path": Path(filename),
        "duration": normalized.shape[0] / float(samplerate),
        "peak": peak,
    }
    if keep_copy:
        RECORDINGS_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        copy_path = RECORDINGS_DIR / f"clip-{timestamp}.wav"
        shutil.copy(str(filename), copy_path)
        info["copy_path"] = copy_path
    return info

def transcribe_audio(filename=INPUT_WAV):
    print("Transcribing")
    with open(filename, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="en",
        )
    text = result.text
    print("You said:", text)
    return text

def ask_ai(text: str, voice_mood: Union[str, None] = None):
    """
    Ask the model to reply and give a simple emotion tag.
    """

    emotion_list = ", ".join(EMOTIONS)
    mood_sentence = ""
    if voice_mood == "alert":
        mood_sentence = "The user sounded loud and urgent, so be attentive. "
    elif voice_mood == "curious":
        mood_sentence = "The user sounded quiet and curious, respond gently. "

    system_prompt = (
        "You are a exhausted station assistant called Zeni. "
        "Reply succinctly in English, even if the user speaks another language. "
        f"{mood_sentence}"
        "Always return a JSON object with fields reply and emotion. "
        f"Emotion must be one of {emotion_list}. "
        "Choose emotions that match the tone of your reply."
    )

    request_args = {
        "model": "gpt-4.1-mini",
        "input": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": text,
            },
        ],
    }

    try:
        resp = client.responses.create(
            **request_args,
            response_format={
                "type": "json_object"
            },
        )
    except TypeError as exc:
        # Older versions of the OpenAI client do not support response_format.
        if "response_format" not in str(exc):
            raise
        resp = client.responses.create(**request_args)

    content = ""
    for message in getattr(resp, "output", []):
        for block in getattr(message, "content", []):
            if hasattr(block, "text"):
                content += block.text
    if not content:
        raise RuntimeError("No response content from model")

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(content[start : end + 1])
        else:
            raise RuntimeError(f"Model response not valid JSON: {content!r}")

    reply = data.get("reply", "")
    emotion = data.get("emotion", "neutral")

    print("AI:", reply, "| emotion:", emotion)

    return reply, emotion

def post_process_audio(audio_data, original_sample_rate):
    """
    Apply audio post-processing:
    - Resample to lower sample rate
    - Drop pitch by a small ratio
    - Add light reverb
    """
    # Target sample rate (lower for different sound quality)
    target_sample_rate = 16000
    
    # Pitch shift ratio (< 1.0 lowers pitch)
    pitch_ratio = 0.75  # 5% lower pitch
    
    # Resample to target rate
    num_samples = int(len(audio_data) * target_sample_rate / original_sample_rate)
    resampled = resample_signal(audio_data, num_samples)
    
    # Apply pitch shift by time-stretching then resampling
    # Stretch time by inverse of pitch ratio
    stretch_samples = int(len(resampled) / pitch_ratio)
    stretched = resample_signal(resampled, stretch_samples)
    
    # Resample back to target rate (this applies the pitch shift)
    pitched = resample_signal(stretched, num_samples)
    
    # Create simple reverb impulse response
    reverb_length = int(target_sample_rate * 0.05)  # 50ms reverb tail
    reverb_ir = np.exp(-np.linspace(0, 5, reverb_length)) * np.random.randn(reverb_length) * 0.15
    reverb_ir[0] = 1.0  # Direct sound
    
    # Apply reverb via convolution
    reverbed = fft_convolve_signal(pitched, reverb_ir, mode='full')[: len(pitched)]
    
    # Normalize to prevent clipping
    max_val = np.abs(reverbed).max()
    if max_val > 0:
        reverbed = reverbed / max_val * 0.95
    
    return reverbed, target_sample_rate

def speak_text(text: str, filename="reply.mp3"):
    print("Speaking")
    result = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
    )

    # Save original TTS output
    temp_filename = filename.replace(".mp3", "_temp.mp3")
    with open(temp_filename, "wb") as f:
        f.write(result.read())
    
    # Load audio for post-processing
    audio_data, sample_rate = sf.read(temp_filename)
    
    # Apply post-processing
    processed_audio, new_sample_rate = post_process_audio(audio_data, sample_rate)
    
    # Save processed audio as WAV (better for pygame)
    processed_filename = filename.replace(".mp3", ".wav")
    sf.write(processed_filename, processed_audio, new_sample_rate)
    
    # Clean up temp file
    os.remove(temp_filename)
    
    pygame.mixer.music.load(processed_filename)
    pygame.mixer.music.play()


def play_last_input(audio_path=INPUT_WAV):
    audio_path = Path(audio_path)
    if not audio_path.exists():
        print("No recorded input available for playback")
        return False
    try:
        sound = pygame.mixer.Sound(str(audio_path))
    except pygame.error as exc:
        print("Unable to play recorded input:", exc)
        return False
    sound.play()
    return True

def main():
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Set OPENAI_API_KEY in your environment")

    configure_input_device()

    face = Face()
    running = True
    status_text = DEFAULT_STATUS

    recorder = AudioRecorder()
    worker_thread = None
    last_audio_path = INPUT_WAV
    greeting_active = False
    last_user_interaction = time.time()
    signal_message_cooldown = 0.0

    def mark_activity():
        nonlocal last_user_interaction
        last_user_interaction = time.time()

    def draw_ui(show_meter=False, level=0.0):
        face.draw(screen)

        if show_meter:
            max_width = WIDTH - 40
            rect = pygame.Rect(20, HEIGHT - 70, max_width, 10)
            pygame.draw.rect(screen, (80, 80, 110), rect, 1)
            filled = int(max_width * min(level / 0.5, 1.0))
            if filled > 0:
                pygame.draw.rect(
                    screen,
                    (80, 200, 255),
                    pygame.Rect(rect.left, rect.top, filled, rect.height),
                )

        text_surface = FONT.render(status_text, True, (220, 220, 240))
        screen.blit(text_surface, (20, HEIGHT - 40))

        pygame.display.flip()
        return face.signal_active

    def handle_turn(audio_data):
        nonlocal status_text
        nonlocal last_audio_path
        try:
            face.set_expression("thinking")
            status_text = "Preparing audio"
            audio_info = save_audio(
                audio_data,
                filename=INPUT_WAV,
                samplerate=recorder.samplerate,
            )
            last_audio_path = audio_info["path"]
            peak = audio_info.get("peak", 0.0)
            if peak < 0.02:
                status_text = "Mic input very quiet - check source"
            voice_mood = classify_voice_mood(peak)
            if voice_mood:
                face.set_voice_hint(voice_mood)

            status_text = "Transcribing"
            transcript = transcribe_audio()

            if not transcript.strip():
                status_text = "Did not hear anything"
                face.set_expression("neutral")
                return

            status_text = "Thinking"
            face.set_expression("thinking")
            reply, emotion = ask_ai(transcript, voice_mood=voice_mood)

            face.set_expression(emotion)
            status_text = "Speaking"
            face.set_talking(True)
            speak_text(reply)

            while pygame.mixer.music.get_busy():
                time.sleep(0.05)

            face.set_talking(False)
            status_text = DEFAULT_STATUS
            face.set_expression("neutral")
            mark_activity()

        except Exception as e:
            print("Error in turn:", e)
            status_text = "Error see console"
            face.set_expression("sad")
            face.set_talking(False)

    def play_greeting(is_initial=False):
        nonlocal status_text
        nonlocal greeting_active
        nonlocal running
        nonlocal last_user_interaction

        if greeting_active:
            return
        if not is_initial:
            if recorder.recording or face.talking or pygame.mixer.music.get_busy():
                return
            if worker_thread is not None and worker_thread.is_alive():
                return

        greeting_active = True
        status_text = "Booting up..." if is_initial else "Still here..."
        message = GREETING_MESSAGES[0] if is_initial else random.choice(GREETING_MESSAGES)
        face.set_expression("happy")
        face.set_talking(True)
        draw_ui()
        speak_text(message)
        temp_clock = pygame.time.Clock()
        while pygame.mixer.music.get_busy() and running:
            dt = temp_clock.tick(60) / 1000.0
            face.update(dt)
            draw_ui()
            interrupted = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.mixer.music.stop()
                    interrupted = True
                    break
                elif event.type == pygame.KEYDOWN:
                    mark_activity()
                    pygame.mixer.music.stop()
                    interrupted = True
                    break
            if interrupted:
                break
        face.set_talking(False)
        face.set_expression("neutral")
        status_text = DEFAULT_STATUS
        greeting_active = False
        mark_activity()

    play_greeting(is_initial=True)

    while running:
        dt = clock.tick(60) / 1000.0
        face.update(dt)

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                mark_activity()
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.unicode in EMOTION_KEYS:
                    expr = EMOTION_KEYS[event.unicode]
                    if face.preview_expression(expr):
                        status_text = f"Previewing {expr}"
                elif event.key == pygame.K_SPACE:
                    if worker_thread is not None and worker_thread.is_alive():
                        status_text = "Busy finishing previous reply"
                    else:
                        if not recorder.recording:
                            recorder.start()
                            face.set_expression("thinking")
                            status_text = "Listening... release space"
                elif event.key == pygame.K_p:
                    if recorder.recording:
                        status_text = "Release space before preview"
                    else:
                        if play_last_input(last_audio_path):
                            status_text = "Playing last input sample"
                        else:
                            status_text = "No recording to play"
                elif event.key == pygame.K_d:
                    print_audio_devices()
                    status_text = "Device list printed to console"
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    mark_activity()
                    if recorder.recording:
                        status_text = "Finishing capture"
                        audio_capture = recorder.stop()
                        if audio_capture is None or audio_capture.size == 0:
                            status_text = "Did not hear anything"
                            face.set_expression("neutral")
                        else:
                            status_text = "Processing"
                            worker_thread = threading.Thread(
                                target=handle_turn,
                                args=(audio_capture,),
                                daemon=True,
                            )
                            worker_thread.start()
                            if face.talking_sequence:
                                pygame.mixer.music.set_endevent(pygame.USEREVENT + 1)

        if (
            not greeting_active
            and (time.time() - last_user_interaction) >= 30.0
            and not recorder.recording
            and not face.talking
            and not pygame.mixer.music.get_busy()
            and (worker_thread is None or not worker_thread.is_alive())
        ):
            play_greeting(is_initial=False)

        signal_now = draw_ui(recorder.recording, recorder.level() if recorder.recording else 0.0)
        if signal_now and signal_message_cooldown <= 0 and not greeting_active:
            status_text = "FATAL: containment.flag ≠ expected"
            signal_message_cooldown = 8.0
        if signal_message_cooldown > 0:
            signal_message_cooldown -= dt

        for event in events:
            if event.type == pygame.USEREVENT + 1:
                face.set_talking(False)

    pygame.quit()

if __name__ == "__main__":
    main()
