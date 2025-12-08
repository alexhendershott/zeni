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
import argparse
from collections import deque, defaultdict

import numpy as np
import sounddevice as sd
import soundfile as sf
import pygame
from openai import OpenAI

try:
    import cv2
except ImportError:  # pragma: no cover - OpenCV optional
    cv2 = None
    print("OpenCV not available; eye contact disabled.")

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

DREAM_PROMPT = (
    "Generate a short internal system thought in 1 to 3 words. "
    "Tone eerie. Mechanical. Fragmented. Not addressed to a user. "
    "Pure inner monologue."
)

SOFT_THOUGHTS = [
    "standby",
    "waiting",
    "low drift",
    "listening low",
    "idle cycle",
]

ABSTRACT_THOUGHTS = [
    "echo field",
    "vector shift nominal",
    "scan depth",
    "listening drift high",
    "cycles incomplete",
]

UNSETTLING_THOUGHTS = [
    "operator absent",
    "containment loose",
    "signal mismatch",
    "echo anomaly",
    "is the operator still present",
]


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
        self.input_level = 0.0

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
        self.listening = False
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
        self.shadow_expression = "neutral"
        self.shadow_target_expression = "neutral"
        self.shadow_shift_timer = 0.0
        self.shadow_offset = [
            random.uniform(-6.0, 6.0),
            random.uniform(-4.0, 4.0),
        ]
        self.shadow_target_offset = list(self.shadow_offset)
        self.shadow_drift_timer = random.uniform(1.8, 3.6)
        self.shadow_merge_timer = random.uniform(4.5, 7.5)
        self.shadow_strength = 0.4
        self.passive_offset = [0.0, 0.0]
        self.passive_target = [0.0, 0.0]
        self.passive_timer = 0.0
        self.sweep = None
        self.proximity_zone = "far"

    def set_expression(self, expr):
        if expr not in EMOTION_SET:
            expr = "happy"
        self.idle_expression = expr
        self._set_shadow_target(expr)
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

    def set_proximity_zone(self, zone):
        if zone not in {"far", "mid", "near"}:
            return
        if zone == self.proximity_zone:
            return
        self.proximity_zone = zone
        zone_expression = {
            "far": "neutral",
            "mid": "curious",
            "near": "excited",
        }.get(zone, "neutral")
        self.idle_expression = zone_expression
        self._set_shadow_target(zone_expression)
        if not self.talking and not self.manual_expression:
            self.expression = zone_expression
        zone_bias = {"far": -0.05, "mid": 0.02, "near": 0.08}
        self.shadow_strength = max(0.25, min(0.85, self.shadow_strength + zone_bias.get(zone, 0.0)))

    def cue_blink(self, slow=False, length=None):
        self.blinking = True
        self.blink_timer = 0.0
        if length is None:
            self.blink_length = 0.26 if slow else 0.15
        else:
            self.blink_length = max(0.08, float(length))
        self.blink_interval = random.uniform(3.0, 4.5) if slow else random.uniform(2.4, 4.2)

    def respond_to_stare(self):
        offset = random.choice([-1.0, 1.0]) * random.uniform(16.0, 26.0)
        self.passive_target = [offset, random.uniform(-4.0, 10.0)]
        self.passive_timer = 0.9
        if not self.talking:
            self.preview_expression("concerned", duration=1.4)

    def react_to_passive(self, event):
        if event.get("type") != "spike":
            return
        side = event.get("side", "left")
        direction = -1 if side == "left" else 1
        intensity = event.get("intensity", "soft")
        speed = event.get("speed", "slow")
        zone = event.get("zone", self.proximity_zone)
        magnitude = {"soft": 8.0, "bump": 14.0, "hit": 20.0}.get(intensity, 10.0)
        if zone == "near":
            magnitude += 4.0
        x_shift = direction * magnitude
        y_shift = -8.0 if speed == "fast" else 4.0
        self.passive_target = [x_shift, y_shift]
        self.passive_timer = 0.9 if speed == "fast" else 0.6
        self.shadow_target_offset[0] += x_shift * 0.15
        self.shadow_target_offset[1] += y_shift * 0.12
        if event.get("arc"):
            sweep_mag = magnitude * 1.6
            sweep_dir = "right" if side == "left" else "left"
            self._start_sweep(sweep_dir, sweep_mag)
        if not self.talking:
            if zone == "near":
                self.preview_expression("excited", duration=0.45)
            elif speed == "fast":
                self.preview_expression("surprised", duration=0.35)
            elif intensity != "soft":
                self.preview_expression("curious", duration=0.5)

    def _set_shadow_target(self, expr):
        if expr not in EMOTION_SET:
            return
        if expr != self.shadow_target_expression:
            self.shadow_target_expression = expr
            self.shadow_shift_timer = random.uniform(0.18, 0.42)

    def _start_sweep(self, direction, magnitude):
        offset = magnitude if direction == "right" else -magnitude
        self.sweep = {
            "timer": 0.0,
            "duration": 0.55,
            "from": list(self.passive_offset),
            "to": [offset, self.passive_target[1] * 0.6],
        }

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

        self._update_shadow_face(dt)
        self._update_head_motion(dt)
        self._update_passive_motion(dt)

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

    def _update_shadow_face(self, dt):
        tone_target = self.manual_expression or self.idle_expression
        if tone_target:
            self._set_shadow_target(tone_target)

        if self.shadow_expression != self.shadow_target_expression:
            self.shadow_shift_timer -= dt
            if self.shadow_shift_timer <= 0:
                self.shadow_expression = self.shadow_target_expression
                self.shadow_shift_timer = random.uniform(0.5, 0.85)

        self.shadow_drift_timer -= dt
        if self.shadow_drift_timer <= 0:
            drift_span = 24 if not self.talking else 36
            self.shadow_target_offset = [
                random.uniform(-drift_span, drift_span),
                random.uniform(-drift_span * 0.6, drift_span * 0.6),
            ]
            self.shadow_drift_timer = random.uniform(1.6, 3.2)

        self.shadow_merge_timer -= dt
        if self.shadow_merge_timer <= 0:
            if random.random() < 0.45:
                self.shadow_target_offset = [0.0, 0.0]
            else:
                merge_span = random.uniform(28.0, 46.0)
                self.shadow_target_offset = [
                    random.uniform(-merge_span, merge_span),
                    random.uniform(-merge_span * 0.6, merge_span * 0.6),
                ]
            self.shadow_merge_timer = random.uniform(4.5, 7.5)

        for i in (0, 1):
            delta = self.shadow_target_offset[i] - self.shadow_offset[i]
            self.shadow_offset[i] += delta * min(1.0, dt * 1.8)

        target_strength = 0.35 + 0.25 * (1.0 if self.talking else 0.0)
        target_strength += 0.15 * min(1.0, self.input_level * 5.0)
        if self.signal_active or self.glitch_active:
            target_strength += 0.08
        self.shadow_strength += (target_strength - self.shadow_strength) * min(1.0, dt * 2.4)
        self.shadow_strength = max(0.25, min(0.85, self.shadow_strength))

    def _update_passive_motion(self, dt):
        if self.sweep:
            self.sweep["timer"] += dt
            t = min(1.0, self.sweep["timer"] / self.sweep["duration"])
            ease = 0.5 - 0.5 * math.cos(math.pi * t)
            for i in (0, 1):
                self.passive_target[i] = (
                    self.sweep["from"][i]
                    + (self.sweep["to"][i] - self.sweep["from"][i]) * ease
                )
            if t >= 1.0:
                self.sweep = None
                self.passive_timer = max(self.passive_timer, 0.45)

        self.passive_timer = max(0.0, self.passive_timer - dt)
        target = self.passive_target if (self.passive_timer > 0.0 or self.sweep) else [0.0, 0.0]
        for i in (0, 1):
            delta = target[i] - self.passive_offset[i]
            self.passive_offset[i] += delta * min(1.0, dt * 6.5)
        if self.passive_timer <= 0 and not self.sweep:
            for i in (0, 1):
                self.passive_offset[i] *= max(0.0, 1.0 - dt * 2.0)
                if abs(self.passive_offset[i]) < 0.2:
                    self.passive_offset[i] = 0.0

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
            surf.fill((0, 0, 0))
            size = int(HEIGHT * 0.6)
            rect = pygame.Rect(
                WIDTH // 2 - size // 2,
                HEIGHT // 2 - size // 2,
                size,
                size,
            )
            border = (230, 230, 240)
            pygame.draw.rect(surf, border, rect, 4)

            inner = rect.inflate(-30, -120)
            pygame.draw.rect(surf, border, inner, 2)

            bar_h = inner.height // 3
            bar = pygame.Rect(inner.left, inner.centery - bar_h // 2, inner.width, bar_h)
            pygame.draw.rect(surf, border, bar)

            return
        
        if self.glitch_active and self.glitch_type in ["flicker", "corrupt"]:
            flicker = int(random.random() * 30 * self.glitch_intensity)
            surf.fill((base_bg[0] + flicker, base_bg[1] + flicker, base_bg[2] + flicker))
        else:
            surf.fill(base_bg)
        
        if self.glitch_active:
            self._apply_glitch_effects(surf)
        style = self.EXPRESSION_STYLES.get(self.expression, self.EXPRESSION_STYLES["neutral"])
        shadow_style = self.EXPRESSION_STYLES.get(
            self.shadow_expression, self.EXPRESSION_STYLES["neutral"]
        )
        base_center = (
            WIDTH // 2 + int(self.head_offset[0] + self.passive_offset[0]),
            HEIGHT // 2 + int(self.head_offset[1] + self.passive_offset[1]),
        )
        center = base_center
        shadow_center = (
            base_center[0] + int(self.shadow_offset[0]),
            base_center[1] + int(self.shadow_offset[1]),
        )
        radius = 140

        base = 28
        pulse_speed = 3.2 if self.talking else 1.6
        pulse = base + int(22 * (0.5 + 0.5 * math.sin(self.talk_timer * pulse_speed + self.talk_phase)))
        pulse = max(20, min(90, pulse))

        if self.listening:
            # dim base
            surf.fill((5, 5, 10))
            # central listening core
            center = (WIDTH // 2 + int(self.head_offset[0]), HEIGHT // 2 + int(self.head_offset[1]))
            r = 40 + int(60 * min(1.0, self.input_level * 6.0))
            intensity = int(80 + 150 * min(1.0, self.input_level * 6.0))
            core = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            pygame.draw.circle(core, (intensity, intensity, intensity + 40, 190), center, r)
            pygame.draw.circle(core, (intensity + 40, intensity + 40, intensity + 90, 120), center, r + 20, 4)
            core.set_alpha(160)
            surf.blit(core, (0, 0))

            # extra static sculpting comes from _apply_ambient_noise
            self._apply_ambient_noise(surf)
            self.previous_frame = surf.copy()
            return

        self._draw_shadow_face(surf, shadow_center, shadow_style, radius)

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

    def _draw_shadow_face(self, surf, center, style, base_radius):
        layer = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        jitter = math.sin(self.noise_timer * 0.9 + self.talk_phase * 0.5)
        radius = int(base_radius * (0.9 + 0.05 * jitter))
        glow_base = int(60 + 90 * self.shadow_strength)
        glow_color = (glow_base, glow_base + 25, glow_base + 70, 140)
        core_color = (
            min(255, glow_base + 30),
            min(255, glow_base + 55),
            min(255, glow_base + 120),
            180,
        )

        pygame.draw.circle(layer, glow_color, center, radius + 12)
        pygame.draw.circle(layer, core_color, center, radius)

        echo_color = (190, 210, 255)
        tone_wobble = 0.12 + 0.25 * self.shadow_strength
        talk_amount = tone_wobble + 0.08 * math.sin(self.noise_timer * 1.4 + self.talk_phase * 0.3)
        talk_amount = max(0.05, talk_amount)
        self._draw_eyes(layer, center, style, color=echo_color)
        self._draw_mouth(layer, center, style, talk_override=talk_amount, color=echo_color)

        separation = abs(self.shadow_offset[0]) + abs(self.shadow_offset[1])
        if separation > 32:
            ghost = layer.copy()
            ghost.set_alpha(int(40 + min(80, separation)))
            surf.blit(
                ghost,
                (
                    int(center[0] - self.shadow_offset[0] * 0.4),
                    int(center[1] - self.shadow_offset[1] * 0.4),
                ),
            )

        layer.set_alpha(int(80 + 120 * self.shadow_strength))
        surf.blit(layer, (0, 0))

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
        mic_factor = min(1.0, self.input_level * 12.0)
        noise_intensity = 0.15 + 0.3 * mic_factor + 0.18 * math.sin(self.noise_timer * 0.8)
        
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
        
        if self.input_level > 0.02:
            band_count = int(4 + self.input_level * 14)
            band_height = int(self.input_level * HEIGHT * 0.95)
            y = HEIGHT // 2 - band_height // 2
            band_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            for i in range(band_count):
                offset = random.random() * 0.5
                x = int(((i + offset) / band_count) * WIDTH)
                width = max(2, int(2 + self.input_level * 10))
                brightness = int(120 + self.input_level * 180)
                alpha = int(80 + self.input_level * 140)
                color = (brightness, brightness, brightness, max(80, min(255, alpha)))
                rect = pygame.Rect(x - width // 2, y, width, band_height)
                pygame.draw.rect(band_surface, color, rect, border_radius=max(2, width // 2))
                pygame.draw.line(
                    band_surface,
                    (min(255, brightness + 30),) * 3 + (max(120, alpha),),
                    (x, y),
                    (x, y + band_height),
                    1,
                )
            band_surface.set_alpha(max(80, int(120 * mic_factor)))
            surf.blit(band_surface, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    
    def _draw_eyes(self, surf, center, style, color=(220, 220, 240)):
        eye_y = center[1] - 40 + style.get("eye_drop", 0)
        eye_offset_x = 45
        base = 18 * style.get("eye_scale", 1.0)

        if self.blinking:
            h = 4
            for sign in (-1, 1):
                x = center[0] + sign * eye_offset_x
                pygame.draw.rect(
                    surf,
                    color,
                    pygame.Rect(x - base, eye_y - 2, base * 2, h),
                )
            return

        if self.expression == "thinking":
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
            pygame.draw.ellipse(surf, color, rect)

    def _talk_amount(self):
        if self.talking:
            base = 0.25 + 0.35 * (0.5 + 0.5 * math.sin(self.talk_timer * 4.2 + self.talk_phase))
            wobble = 0.08 * math.sin(self.talk_timer * 8.5 + self.talk_phase * 0.7)
            return max(0.08, base + wobble)
        return 0.08 + 0.03 * math.sin(self.talk_timer * 1.6 + self.talk_phase)

    def _draw_mouth(self, surf, center, style, talk_override=None, color=(220, 220, 240)):
        m = style.get("mouth", "flat")
        width = style.get("mouth_width", 90)
        if self.talking:
            width *= 1.0 + 0.08 * math.sin(self.talk_timer * 5.3 + self.talk_phase)
        width = int(width)
        y = center[1] + 40
        talk = self._talk_amount() if talk_override is None else talk_override
        height = 35 + talk * 30

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

class DreamMonologue:
    """Manages subconscious glitch text during idle time."""

    def __init__(self, face: Face):
        self.face = face
        self.fragments = deque(maxlen=12)
        self.overlays = []
        self.flash_timer = 0.0
        self.next_ai_time = time.time() + random.uniform(80.0, 95.0)
        self.snap_pulse = None
        self.idle_stage = "soft"
        self.last_idle_time = 0.0

    def reset(self):
        self.fragments.clear()
        self.overlays.clear()
        self.flash_timer = 0.0
        self.snap_pulse = None
        self.next_ai_time = time.time() + random.uniform(80.0, 95.0)

    def snap_out(self):
        """User interrupts the dream loop."""
        self.reset()
        self.snap_pulse = {"timer": 0.0, "duration": 0.6}
        if self.face and not self.face.talking:
            self.face.preview_expression("surprised", duration=0.9)

    def _stage_for_idle(self, idle_time):
        if idle_time >= 60.0:
            return "unsettling"
        if idle_time >= 20.0:
            return "abstract"
        return "soft"

    def _flash_interval(self, stage):
        if stage == "unsettling":
            return random.uniform(1.1, 2.3)
        if stage == "abstract":
            return random.uniform(2.4, 4.2)
        return random.uniform(4.8, 8.2)

    def _fallback_fragment(self, stage):
        if stage == "unsettling":
            return random.choice(UNSETTLING_THOUGHTS)
        if stage == "abstract":
            return random.choice(ABSTRACT_THOUGHTS)
        return random.choice(SOFT_THOUGHTS)

    def _choose_fragment(self, stage):
        if self.fragments and random.random() < 0.7:
            return random.choice(self.fragments)["text"]
        return self._fallback_fragment(stage)

    def _apply_emotion_influence(self, text, stage):
        if not self.face or self.face.talking:
            return
        lower = text.lower()
        if "containment" in lower or "loose" in lower:
            self.face.preview_expression("concerned", duration=2.4)
            return
        if "echo" in lower:
            self.face.preview_expression("thinking", duration=1.8)
            return
        if "operator" in lower:
            self.face.preview_expression("surprised", duration=1.4)
            return
        if stage == "unsettling":
            self.face.preview_expression("concerned", duration=1.6)
        elif stage == "abstract":
            self.face.preview_expression("curious", duration=1.2)

    def _generate_dream_line(self):
        try:
            try:
                resp = client.responses.create(
                    model="gpt-4o-mini",
                    input=[
                        {"role": "system", "content": DREAM_PROMPT},
                    ],
                    max_output_tokens=12,
                )
            except TypeError:
                resp = client.responses.create(
                    model="gpt-4o-mini",
                    input=[
                        {"role": "system", "content": DREAM_PROMPT},
                    ],
                )
            content = ""
            for message in getattr(resp, "output", []):
                for block in getattr(message, "content", []):
                    if hasattr(block, "text"):
                        content += block.text
            text = content.strip().splitlines()[0] if content else ""
            text = text.strip("\"' ")
            if text:
                return text
        except Exception as exc:
            print("Dream prompt failed:", exc)
        return None

    def _spawn_overlay(self, text, stage):
        edge = random.choice(["top", "bottom", "left", "right"])
        margin = 24
        jitter = 18
        if edge == "top":
            pos = [random.randint(margin, WIDTH - margin), random.randint(6, 36)]
        elif edge == "bottom":
            pos = [random.randint(margin, WIDTH - margin), HEIGHT - random.randint(30, 56)]
        elif edge == "left":
            pos = [random.randint(6, 28), random.randint(margin, HEIGHT - margin)]
        else:
            pos = [WIDTH - random.randint(28, 52), random.randint(margin, HEIGHT - margin)]
        pos[0] += random.randint(-jitter, jitter)
        pos[1] += random.randint(-jitter // 2, jitter // 2)

        duration = {
            "soft": random.uniform(0.45, 0.85),
            "abstract": random.uniform(0.7, 1.2),
            "unsettling": random.uniform(1.1, 1.8),
        }.get(stage, 0.7)

        alpha = {
            "soft": 90,
            "abstract": 140,
            "unsettling": 180,
        }.get(stage, 120)

        overlay = {
            "text": text,
            "pos": pos,
            "timer": 0.0,
            "duration": duration,
            "alpha": alpha,
            "stage": stage,
            "slice": random.uniform(0.5, 0.9),
            "drift": (random.uniform(-6.0, 6.0), random.uniform(-3.0, 3.0)),
            "shake": random.uniform(1.5, 4.0),
        }
        self.overlays.append(overlay)
        self._apply_emotion_influence(text, stage)

    def update(self, dt, idle_time, allow_idle):
        self.last_idle_time = idle_time
        now = time.time()
        if not allow_idle:
            self.overlays.clear()
            self.flash_timer = 0.0
            return

        self.idle_stage = self._stage_for_idle(idle_time)

        if now >= self.next_ai_time:
            thought = self._generate_dream_line()
            if thought:
                self.fragments.append({"text": thought, "stage": self.idle_stage})
            self.next_ai_time = now + random.uniform(85.0, 100.0)

        self.flash_timer -= dt
        if idle_time > 0.5 and self.flash_timer <= 0.0:
            fragment = self._choose_fragment(self.idle_stage)
            self._spawn_overlay(fragment, self.idle_stage)
            self.flash_timer = self._flash_interval(self.idle_stage)

        alive_overlays = []
        for overlay in self.overlays:
            overlay["timer"] += dt
            overlay["pos"][0] += overlay["drift"][0] * dt * 0.5
            overlay["pos"][1] += overlay["drift"][1] * dt * 0.5
            if overlay["timer"] <= overlay["duration"]:
                alive_overlays.append(overlay)
        self.overlays = alive_overlays

        if self.snap_pulse:
            self.snap_pulse["timer"] += dt
            if self.snap_pulse["timer"] >= self.snap_pulse["duration"]:
                self.snap_pulse = None

    def render(self, surf):
        if self.snap_pulse:
            t = self.snap_pulse["timer"] / max(0.001, self.snap_pulse["duration"])
            fade = max(0.0, 1.0 - t)
            pulse_alpha = int(160 * fade)
            pulse = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            tint = (80, 90, 150, pulse_alpha)
            pulse.fill(tint)
            pygame.draw.rect(
                pulse,
                (150, 180, 255, int(pulse_alpha * 0.5)),
                pygame.Rect(10, 10, WIDTH - 20, HEIGHT - 20),
                2,
            )
            surf.blit(pulse, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

        if not self.overlays:
            return

        for overlay in self.overlays:
            life = overlay["timer"] / max(0.001, overlay["duration"])
            fade = max(0.0, 1.0 - life * life)
            base_alpha = int(overlay["alpha"] * fade)
            if base_alpha <= 0:
                continue

            text = overlay["text"]
            color = (170, 180, 210)
            if overlay["stage"] == "unsettling":
                color = (190, 150, 150)
            elif overlay["stage"] == "abstract":
                color = (180, 180, 230)

            raw = SMALL_FONT.render(text, True, color)
            width = raw.get_width()
            if width == 0:
                continue

            slice_width = max(6, int(width * overlay["slice"]))
            start = random.randint(0, max(0, width - slice_width))
            clip = pygame.Rect(start, 0, slice_width, raw.get_height())
            fragment = pygame.Surface((slice_width, raw.get_height()), pygame.SRCALPHA)
            fragment.blit(raw, (-start, 0), clip)

            if random.random() < 0.8:
                blur = pygame.transform.smoothscale(
                    fragment, (int(slice_width * 0.9), max(1, int(raw.get_height() * 0.9)))
                )
                fragment.blit(blur, (random.randint(-2, 2), random.randint(-1, 1)), special_flags=pygame.BLEND_RGBA_ADD)

            if random.random() < 0.5:
                scan_x = random.randint(0, fragment.get_width() - 1)
                pygame.draw.line(
                    fragment,
                    (40, 120, 220),
                    (scan_x, 0),
                    (scan_x, fragment.get_height()),
                    1,
                )

            fragment.set_alpha(base_alpha)
            x_jitter = random.randint(-int(overlay["shake"]), int(overlay["shake"]))
            y_jitter = random.randint(-int(overlay["shake"]), int(overlay["shake"]))
            dest = (
                int(overlay["pos"][0] + x_jitter),
                int(overlay["pos"][1] + y_jitter),
            )
            surf.blit(fragment, dest, special_flags=pygame.BLEND_RGBA_ADD)

            if random.random() < 0.35:
                ghost = fragment.copy()
                ghost.set_alpha(int(base_alpha * 0.4))
                surf.blit(
                    ghost,
                    (dest[0] + random.randint(-6, 6), dest[1] + random.randint(-4, 4)),
                    special_flags=pygame.BLEND_RGBA_ADD,
                )


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


class PassiveSense:
    """
    Lightweight passive listener that samples amplitude only.
    It emits spike events and proximity zone changes without keeping raw audio.
    """

    def __init__(
        self,
        samplerate=16000,
        block_duration=0.08,
        avg_window=1.6,
        sensitivity=1.0,
    ):
        self.samplerate = samplerate
        self.block_duration = block_duration
        self.avg_window = avg_window
        self._lock = threading.Lock()
        self._history = deque(maxlen=int(avg_window / block_duration) + 20)
        self._events = deque(maxlen=32)
        self._stream = None
        self._paused = False
        self._level = 0.0
        self._last_amp = 0.0
        self.zone = "far"
        self._last_side = random.choice(["left", "right"])
        self._last_spike_time = 0.0
        self.sensitivity = max(0.2, min(5.0, float(sensitivity)))
        self._base_spike_floor = 0.0015
        self._base_spike_threshold = 0.0035
        self._base_delta_threshold = 0.0025
        self._base_zone_mid = 0.005
        self._base_zone_near = 0.018
        self._recompute_thresholds()

    def start(self):
        if self._stream:
            return
        blocksize = max(64, int(self.samplerate * self.block_duration))
        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        if not self._stream:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def _choose_side(self, speed_tag):
        if speed_tag == "fast":
            if random.random() < 0.6:
                return "right" if self._last_side == "left" else "left"
        return "left" if random.random() < 0.55 else "right"

    def _recompute_thresholds(self):
        scale = max(0.2, min(5.0, self.sensitivity))
        inv = 1.0 / scale
        self.spike_floor = self._base_spike_floor * inv
        self.spike_threshold = self._base_spike_threshold * inv
        self.delta_threshold = self._base_delta_threshold * inv
        self.zone_mid = self._base_zone_mid * inv
        self.zone_near = self._base_zone_near * inv

    def set_sensitivity(self, value):
        self.sensitivity = max(0.2, min(5.0, float(value)))
        self._recompute_thresholds()

    def _callback(self, indata, frames, time_info, status):
        if status:
            print("Passive audio status:", status)
        if self._paused:
            return
        amp = float(np.sqrt(np.mean(np.square(indata)))) if indata.size else 0.0
        amp = min(1.0, amp)
        now = time.time()
        with self._lock:
            self._history.append((now, amp))
            self._detect_spike(now, amp)
            self._update_zone(now)
            self._level = 0.65 * self._level + 0.35 * amp
            self._last_amp = amp

    def _detect_spike(self, now, amp):
        delta = amp - self._last_amp
        if amp < self.spike_floor:
            return
        if amp < self.spike_threshold and delta < self.delta_threshold:
            return
        speed_tag = "fast" if delta > 0.025 else "slow"
        if amp > 0.04:
            intensity = "hit"
        elif amp > 0.012:
            intensity = "bump"
        else:
            intensity = "soft"
        side = self._choose_side(speed_tag)
        arc = (now - self._last_spike_time) < 1.1 and side != self._last_side
        self._last_spike_time = now
        self._last_side = side
        self._events.append(
            {
                "type": "spike",
                "amp": amp,
                "side": side,
                "speed": speed_tag,
                "intensity": intensity,
                "arc": arc,
                "zone": self.zone,
            }
        )

    def _update_zone(self, now):
        cutoff = now - self.avg_window
        recent = [amp for t, amp in self._history if t >= cutoff]
        avg = sum(recent) / len(recent) if recent else 0.0
        zone = "far"
        if avg > self.zone_near:
            zone = "near"
        elif avg > self.zone_mid:
            zone = "mid"
        if zone != self.zone:
            self.zone = zone
            self._events.append({"type": "zone", "zone": zone})

    def poll(self):
        with self._lock:
            events = list(self._events)
            self._events.clear()
            level = self._level
        return events, level


class GazeTracker:
    """
    Webcam-based gaze tracker that emits coarse social cues:
    - eye_contact: viewer is roughly centered
    - look_away: contact broken
    - stare: prolonged contact
    - slow_blink: long blink detected
    - squint: eyes narrowed for a stretch
    """

    def __init__(
        self,
        camera_index=None,
        frame_width=320,
        frame_height=240,
        stare_threshold=5.0,
        lookaway_grace=0.45,
    ):
        self.enabled = cv2 is not None and os.getenv("DISABLE_VISION", "0") != "1"
        default_cam = os.getenv("VISION_CAMERA_INDEX", "0")
        try:
            default_cam_idx = int(default_cam)
        except ValueError:
            default_cam_idx = 0
        self.camera_index = camera_index if camera_index is not None else default_cam_idx
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_interval = 0.08
        self.stare_threshold = stare_threshold
        self.lookaway_grace = lookaway_grace

        self._cap = None
        self._thread = None
        self._lock = threading.Lock()
        self._events = deque(maxlen=24)
        self._running = False
        self._debug_stats = {
            "frames": 0,
            "faces": 0,
            "eyes": 0,
            "eye_contact_frames": 0,
            "events": defaultdict(int),
        }

        self.eye_contact = False
        self.contact_start = None
        self.contact_loss_time = None
        self.stare_fired = False
        self.last_seen = 0.0

        self.blink_start = None
        self.last_squint_event = 0.0
        self.squint_timer = 0.0

        self.face_cascade = None
        self.eye_cascade = None
        self.eye_cascade_name = None

        if self.enabled:
            self._setup_camera()

    def _load_eye_cascade(self, casc_path):
        if not casc_path:
            return None
        candidates = [
            "haarcascade_eye_tree_eyeglasses.xml",
            "haarcascade_eye.xml",
        ]
        for filename in candidates:
            full_path = os.path.join(casc_path, filename)
            cascade = cv2.CascadeClassifier(full_path)
            if cascade is not None and not cascade.empty():
                self.eye_cascade_name = filename
                return cascade
        return None

    def _setup_camera(self):
        try:
            self._cap = cv2.VideoCapture(self.camera_index)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            try:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            casc_path = getattr(cv2.data, "haarcascades", "")
            self.face_cascade = cv2.CascadeClassifier(os.path.join(casc_path, "haarcascade_frontalface_default.xml"))
            self.eye_cascade = self._load_eye_cascade(casc_path)
            if (
                not self._cap.isOpened()
                or self.face_cascade is None
                or self.face_cascade.empty()
                or self.eye_cascade is None
                or self.eye_cascade.empty()
            ):
                raise RuntimeError("camera or cascade unavailable")
            print(
                f"Vision ready on camera {self.camera_index} "
                f"(eye cascade {self.eye_cascade_name or 'unknown'})"
            )
        except Exception as exc:
            print("Vision unavailable:", exc)
            self.enabled = False
            if self._cap:
                self._cap.release()
                self._cap = None

    def start(self):
        if not self.enabled or self._running or not self._cap:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.6)
            self._thread = None
        if self._cap:
            self._cap.release()
            self._cap = None

    def poll(self):
        if not self.enabled:
            return [], {"eye_contact": False, "last_seen": self.last_seen}
        with self._lock:
            events = list(self._events)
            self._events.clear()
            state = {
                "eye_contact": self.eye_contact,
                "last_seen": self.last_seen,
            }
        return events, state

    def debug_snapshot(self):
        with self._lock:
            return {
                "frames": self._debug_stats["frames"],
                "faces": self._debug_stats["faces"],
                "eyes": self._debug_stats["eyes"],
                "eye_contact_frames": self._debug_stats["eye_contact_frames"],
                "events": dict(self._debug_stats["events"]),
                "last_seen": self.last_seen,
                "eye_contact": self.eye_contact,
            }

    def _push_event(self, event):
        event["ts"] = time.time()
        with self._lock:
            self._events.append(event)
            etype = event.get("type")
            if etype:
                self._debug_stats["events"][etype] += 1

    def _loop(self):
        while self._running:
            ok, frame = self._cap.read()
            now = time.time()
            if not ok:
                time.sleep(self.frame_interval)
                continue

            self.frame_height, self.frame_width = frame.shape[:2]
            with self._lock:
                self._debug_stats["frames"] += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = self._largest_face(gray)
            if face is None:
                self._handle_no_face(now)
                time.sleep(self.frame_interval)
                continue

            with self._lock:
                self._debug_stats["faces"] += 1
                self.last_seen = now
            x, y, w, h = face
            in_contact = self._is_eye_contact(x, y, w, h, self.frame_width, self.frame_height)
            self._update_contact(in_contact, now)

            if in_contact:
                with self._lock:
                    self._debug_stats["eye_contact_frames"] += 1

            eyes = self._detect_eyes(gray, face)
            if len(eyes) > 0:
                with self._lock:
                    self._debug_stats["eyes"] += 1
            self._update_blinks_and_squint(eyes, w, h, now)

            time.sleep(self.frame_interval)

    def _largest_face(self, gray):
        if not self.face_cascade:
            return None
        normalized = cv2.equalizeHist(gray)
        min_w = max(24, int(self.frame_width * 0.12))
        min_h = max(24, int(self.frame_height * 0.12))
        faces = self.face_cascade.detectMultiScale(
            normalized,
            scaleFactor=1.08,
            minNeighbors=3,
            minSize=(min_w, min_h),
        )
        if len(faces) == 0:
            return None
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        return faces[0]

    def _is_eye_contact(self, x, y, w, h, frame_w, frame_h):
        cx = x + w * 0.5
        cy = y + h * 0.5
        norm_dx = abs(cx - frame_w * 0.5) / max(1.0, frame_w * 0.5)
        norm_dy = abs(cy - frame_h * 0.5) / max(1.0, frame_h * 0.5)
        size_ratio = (w * h) / float(max(1.0, frame_w * frame_h))
        return norm_dx < 0.48 and norm_dy < 0.42 and size_ratio > 0.008

    def _handle_no_face(self, now):
        if self.eye_contact:
            if self.contact_loss_time is None:
                self.contact_loss_time = now
            elif now - self.contact_loss_time >= self.lookaway_grace:
                self._push_event({"type": "look_away"})
                self.eye_contact = False
                self.contact_start = None
                self.stare_fired = False
                self.contact_loss_time = None

    def _update_contact(self, in_contact, now):
        if in_contact:
            self.contact_loss_time = None
            if not self.eye_contact:
                self._push_event({"type": "eye_contact"})
                self.contact_start = now
                self.stare_fired = False
            self.eye_contact = True
            if self.contact_start:
                duration = now - self.contact_start
                if duration >= self.stare_threshold and not self.stare_fired:
                    self._push_event({"type": "stare", "duration": duration})
                    self.stare_fired = True
        else:
            if self.eye_contact:
                if self.contact_loss_time is None:
                    self.contact_loss_time = now
                elif now - self.contact_loss_time >= self.lookaway_grace:
                    self._push_event({"type": "look_away"})
                    self.eye_contact = False
                    self.contact_start = None
                    self.stare_fired = False
                    self.contact_loss_time = None

    def _detect_eyes(self, gray, face):
        if not self.eye_cascade:
            return []
        x, y, w, h = face
        roi_gray = gray[y : y + h, x : x + w]
        if roi_gray.size == 0:
            return []
        roi_gray = cv2.equalizeHist(roi_gray)
        min_w = max(12, int(w * 0.12))
        min_h = max(10, int(h * 0.08))
        eyes = self.eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(min_w, min_h),
        )
        return eyes

    def _update_blinks_and_squint(self, eyes, face_w, face_h, now):
        eyes_visible = len(eyes) > 0

        if eyes_visible:
            if self.blink_start is not None:
                blink_duration = now - self.blink_start
                if blink_duration >= 0.35:
                    self._push_event({"type": "slow_blink", "duration": blink_duration})
                self.blink_start = None

            ratios = [eh / ew for _, _, ew, eh in eyes if ew > 0]
            if ratios:
                avg_ratio = sum(ratios) / len(ratios)
                if avg_ratio < 0.26:
                    self.squint_timer += self.frame_interval
                    if (
                        now - self.last_squint_event > 2.0
                        and self.squint_timer >= 0.6
                    ):
                        self._push_event({"type": "squint"})
                        self.last_squint_event = now
                else:
                    self.squint_timer = 0.0
            else:
                self.squint_timer = 0.0
        else:
            self.squint_timer = 0.0
            if self.blink_start is None:
                self.blink_start = now


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


def make_confirmation_chirp():
    init = pygame.mixer.get_init()
    if not init:
        return None
    sample_rate, _, channels = init
    duration = 0.14
    samples = int(sample_rate * duration)
    t = np.linspace(0.0, duration, samples, endpoint=False)

    f0, f1 = 520.0, 1240.0
    sweep = f0 + (f1 - f0) * (t / duration)
    envelope = np.exp(-t * 12.0) * (0.25 + 0.75 * t / duration)
    wobble = 0.2 * np.sin(2 * math.pi * 18 * t)
    tone = np.sin(2 * math.pi * sweep * t + wobble)
    overtone = 0.35 * np.sin(2 * math.pi * (sweep * 1.8) * t)
    signal = envelope * (tone + overtone)
    signal = np.clip(signal * 28000, -32767, 32767).astype(np.int16)

    if channels > 1:
        signal = np.repeat(signal[:, None], channels, axis=1)
    else:
        signal = signal[:, None]

    try:
        return pygame.sndarray.make_sound(signal)
    except pygame.error:
        return None

def run_vision_check(duration=8.0, verbose=False, camera_index=None):
    """
    Runs the gaze tracker alone for a short sampling window and prints stats.
    Useful for confirming the camera and cascades are working without launching the full UI.
    """
    tracker = GazeTracker(camera_index=camera_index)
    if not tracker.enabled:
        print("Vision unavailable: OpenCV missing, camera busy, or cascades not found.")
        return

    duration = max(1.0, float(duration))
    tracker.start()
    print(f"Running vision check for {duration:.1f}s on camera index {tracker.camera_index}...")
    event_counts = defaultdict(int)
    try:
        end_time = time.time() + duration
        while time.time() < end_time:
            events, state = tracker.poll()
            for evt in events:
                etype = evt.get("type")
                if etype:
                    event_counts[etype] += 1
                    if verbose:
                        print(f"Event: {etype} ({evt})")
            if verbose and state.get("eye_contact"):
                print("Eye contact detected")
            time.sleep(0.12)
    finally:
        tracker.stop()

    stats = tracker.debug_snapshot()
    contact_ratio = (
        stats["eye_contact_frames"] / stats["frames"] if stats["frames"] else 0.0
    )
    print(
        f"Frames: {stats['frames']}, faces seen: {stats['faces']}, "
        f"eyes detected: {stats['eyes']}, contact ratio: {contact_ratio:.2f}"
    )
    if stats["last_seen"]:
        print(f"Last face seen {time.time() - stats['last_seen']:.2f}s ago")
    else:
        print("No face detected during the check.")
    if event_counts:
        print("Event counts:", dict(event_counts))
    else:
        print("No gaze events were fired.")


def list_cameras(max_index=5):
    """Probe a range of camera indices and report which open and produce frames."""
    if cv2 is None:
        print("OpenCV not available; cannot probe cameras.")
        return
    try:
        max_index = int(max_index)
    except Exception:
        max_index = 5
    max_index = max(0, max_index)
    print(f"Probing cameras 0..{max_index} ...")
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        opened = cap.isOpened()
        read_ok = False
        shape = None
        if opened:
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            except Exception:
                pass
            for _ in range(3):
                ok, frame = cap.read()
                if ok and frame is not None:
                    read_ok = True
                    shape = frame.shape
                    break
        cap.release()
        shape_text = f"{shape}" if shape is not None else "n/a"
        print(f"[{idx}] opened={opened} read={read_ok} shape={shape_text}")

def main():
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Set OPENAI_API_KEY in your environment")

    configure_input_device()

    try:
        passive_default = float(os.getenv("PASSIVE_SENSE_GAIN", "1.0"))
    except ValueError:
        passive_default = 1.0

    passive = PassiveSense(sensitivity=passive_default)
    try:
        passive.start()
    except Exception as exc:
        print("Passive sensing unavailable:", exc)
        passive = None

    vision = GazeTracker()
    if vision and not vision.enabled:
        vision = None
    elif vision:
        vision.start()

    face = Face()
    dreams = DreamMonologue(face)
    if passive:
        face.set_proximity_zone(passive.zone)
    running = True
    status_text = DEFAULT_STATUS
    current_gaze_state = {"eye_contact": False, "last_seen": 0.0}
    last_reply_text = None
    lookaway_interrupt = False
    reengage_thread = None
    confirmation_sound = make_confirmation_chirp()
    if confirmation_sound:
        confirmation_sound.set_volume(0.35)

    recorder = AudioRecorder()
    worker_thread = None
    last_audio_path = INPUT_WAV
    greeting_active = False
    last_user_interaction = time.time()
    signal_message_cooldown = 0.0
    prethought_flash = None
    speech_paused_by_gaze = False
    speech_pause_started = None
    talking_paused_for_gaze = False

    def mark_activity():
        nonlocal last_user_interaction
        idle_gap = time.time() - last_user_interaction
        if idle_gap > 2.0:
            dreams.snap_out()
        last_user_interaction = time.time()

    def pick_prethought_fragment(reply_text: str):
        tokens = [w.strip(".,!?;:") for w in reply_text.split() if w.strip(".,!?;:")]
        if not tokens:
            return None
        word = random.choice(tokens)[:12]
        glitch_chars = ["|", "/", "\\", "#", "_", "~"]
        glitched = []
        for ch in word:
            roll = random.random()
            if roll < 0.2:
                glitched.append(random.choice(glitch_chars))
            elif roll < 0.45:
                glitched.append(ch.upper())
            else:
                glitched.append(ch)
        if random.random() < 0.35:
            glitched.append(random.choice(glitch_chars))
        return "".join(glitched)

    def trigger_prethought_flash(reply_text: str):
        nonlocal prethought_flash
        fragment = pick_prethought_fragment(reply_text)
        if not fragment:
            return
        prethought_flash = {
            "text": fragment,
            "timer": 0.0,
            "duration": 0.05,  # a couple of frames
            "offset": (random.randint(-24, 24), random.randint(-12, 12)),
            "alpha": random.randint(160, 230),
        }

    def draw_ui(show_meter=False, level=0.0):
        face.input_level = level
        face.listening = show_meter
        face.draw(screen)

        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        tint = (0, 0, 0, 0)

        if face.expression in ("happy", "excited", "curious"):
            tint = (50, 40, 80, 40)
        elif face.expression in ("sad", "concerned"):
            tint = (10, 30, 80, 60)
        elif face.expression == "confused":
            tint = (80, 60, 20, 50)
        elif face.expression == "surprised":
            tint = (120, 80, 20, 70)
        elif face.signal_active:
            tint = (150, 30, 30, 90)

        if tint[3] > 0:
            overlay.fill(tint)
            screen.blit(overlay, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

        if prethought_flash:
            flicker = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            haze_alpha = max(30, prethought_flash["alpha"] // 3)
            flicker.fill((40, 40, 90, haze_alpha))
            word = prethought_flash["text"]
            jitter_x = random.randint(-2, 2)
            jitter_y = random.randint(-2, 2)
            text = SMALL_FONT.render(word, True, (245, 245, 255))
            text_rect = text.get_rect(
                center=(
                    WIDTH // 2 + prethought_flash["offset"][0] + jitter_x,
                    80 + prethought_flash["offset"][1] + jitter_y,
                )
            )
            flicker.blit(text, text_rect)
            ghost = text.copy()
            ghost.set_alpha(80)
            flicker.blit(ghost, text_rect.move(random.randint(-3, 3), random.randint(-1, 2)))
            flicker.set_alpha(prethought_flash["alpha"])
            screen.blit(flicker, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

        dreams.render(screen)

        text_surface = FONT.render(status_text, True, (220, 220, 240))
        screen.blit(text_surface, (20, HEIGHT - 40))

        # Vision indicator (top-right)
        indicator_text = "VISION OFF"
        indicator_color = (110, 110, 110)
        now = time.time()
        last_seen = current_gaze_state.get("last_seen") or 0.0
        contact = current_gaze_state.get("eye_contact", False)
        if vision:
            if last_seen <= 0 or now - last_seen > 2.0:
                indicator_text = "NO FACE"
                indicator_color = (200, 140, 60)
            else:
                if contact:
                    indicator_text = "EYE CONTACT"
                    indicator_color = (90, 220, 140)
                else:
                    indicator_text = "FACE"
                    indicator_color = (160, 200, 240)
            if last_seen:
                age = now - last_seen
                indicator_text += f" ({age:0.1f}s)"
        indicator = SMALL_FONT.render(indicator_text, True, indicator_color)
        ind_rect = indicator.get_rect(topright=(WIDTH - 16, 12))
        pygame.draw.circle(screen, indicator_color, (ind_rect.left - 10, ind_rect.centery), 6)
        screen.blit(indicator, ind_rect)

        pygame.display.flip()
        return face.signal_active

    def handle_turn(audio_data):
        nonlocal status_text
        nonlocal last_audio_path
        nonlocal lookaway_interrupt
        try:
            lookaway_interrupt = False
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

            last_reply_text = reply
            face.set_expression(emotion)
            status_text = "Speaking"
            trigger_prethought_flash(reply)
            # Draw once immediately so the pre-thought flash renders even if speech starts fast
            draw_ui()
            time.sleep(0.02)
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
        nonlocal lookaway_interrupt

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

    def reengage_after_lookaway():
        nonlocal status_text
        nonlocal lookaway_interrupt
        nonlocal reengage_thread
        face.set_expression("concerned")
        face.set_talking(True)
        status_text = "Eyes on me..."
        speak_text("Eyes on me—I'm speaking.")
        while pygame.mixer.music.get_busy() and running:
            time.sleep(0.05)
        face.set_expression("thinking")
        status_text = "Repeating"
        if last_reply_text:
            speak_text(last_reply_text)
            while pygame.mixer.music.get_busy() and running:
                time.sleep(0.05)
        face.set_talking(False)
        face.set_expression("neutral")
        status_text = DEFAULT_STATUS
        lookaway_interrupt = False
        reengage_thread = None

    play_greeting(is_initial=True)

    while running:
        dt = clock.tick(60) / 1000.0

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
                            if confirmation_sound:
                                confirmation_sound.play()
                            if passive:
                                passive.pause()
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
                elif event.key == pygame.K_v:
                    if vision:
                        print("Vision debug:", vision.debug_snapshot())
                        status_text = "Vision stats printed"
                    else:
                        status_text = "Vision disabled"
                elif event.key == pygame.K_LEFTBRACKET and passive:
                    passive.set_sensitivity(passive.sensitivity * 0.85)
                    status_text = f"Passive sensitivity {passive.sensitivity:.2f}x"
                elif event.key == pygame.K_RIGHTBRACKET and passive:
                    passive.set_sensitivity(passive.sensitivity * 1.15)
                    status_text = f"Passive sensitivity {passive.sensitivity:.2f}x"
                elif event.key == pygame.K_d:
                    print_audio_devices()
                    status_text = "Device list printed to console"
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    mark_activity()
                    if recorder.recording:
                        status_text = "Finishing capture"
                        audio_capture = recorder.stop()
                        if passive:
                            passive.resume()
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

        idle_time = time.time() - last_user_interaction
        if (
            not greeting_active
            and idle_time >= 30.0
            and not recorder.recording
            and not face.talking
            and not pygame.mixer.music.get_busy()
            and (worker_thread is None or not worker_thread.is_alive())
        ):
            play_greeting(is_initial=False)

        passive_events, passive_level = (passive.poll() if passive else ([], 0.0))
        for evt in passive_events:
            if evt.get("type") == "zone":
                face.set_proximity_zone(evt.get("zone"))
            elif evt.get("type") == "spike":
                face.react_to_passive(evt)

        gaze_events, gaze_state = (vision.poll() if vision else ([], {}))
        if gaze_state:
            current_gaze_state = gaze_state
        for evt in gaze_events:
            etype = evt.get("type")
            if etype == "eye_contact":
                mark_activity()
                if lookaway_interrupt and last_reply_text and not recorder.recording:
                    if (
                        (reengage_thread is None or not reengage_thread.is_alive())
                        and not pygame.mixer.music.get_busy()
                        and (worker_thread is None or not worker_thread.is_alive())
                    ):
                        reengage_thread = threading.Thread(target=reengage_after_lookaway, daemon=True)
                        reengage_thread.start()
                        status_text = "Re-engaging..."
                elif speech_paused_by_gaze and pygame.mixer.music.get_busy():
                    pygame.mixer.music.unpause()
                    if talking_paused_for_gaze:
                        face.set_talking(True)
                        talking_paused_for_gaze = False
                    speech_paused_by_gaze = False
                    speech_pause_started = None
                    status_text = "Resuming..."
                elif not recorder.recording and not face.talking and not pygame.mixer.music.get_busy():
                    face.preview_expression("happy", duration=0.9)
            elif etype == "look_away":
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                    face.set_talking(False)
                    speech_paused_by_gaze = False
                    talking_paused_for_gaze = False
                    speech_pause_started = None
                    lookaway_interrupt = True
                    face.preview_expression("concerned", duration=1.0)
                    status_text = "Stopped: eye contact lost"
                elif not recorder.recording:
                    face.preview_expression("curious", duration=0.7)
            elif etype == "stare":
                face.respond_to_stare()
            elif etype == "slow_blink":
                face.cue_blink(slow=True, length=evt.get("duration", 0.4))
            elif etype == "squint":
                face.preview_expression("confused", duration=1.0)
                face.passive_target = [random.uniform(-10.0, 10.0), random.uniform(-2.0, 8.0)]
                face.passive_timer = 0.6

        if speech_paused_by_gaze and speech_pause_started:
            if not pygame.mixer.music.get_busy():
                speech_paused_by_gaze = False
                talking_paused_for_gaze = False
                speech_pause_started = None
            elif time.time() - speech_pause_started > 8.0:
                pygame.mixer.music.stop()
                face.set_talking(False)
                speech_paused_by_gaze = False
                talking_paused_for_gaze = False
                speech_pause_started = None
                status_text = "Stopped: eye contact lost"

        busy = (
            recorder.recording
            or face.talking
            or pygame.mixer.music.get_busy()
            or greeting_active
            or (worker_thread is not None and worker_thread.is_alive())
        )
        dreams.update(dt, idle_time, allow_idle=not busy)

        face.update(dt)
        level_source = recorder.level() if recorder.recording else passive_level
        signal_now = draw_ui(recorder.recording, level_source)
        if signal_now and signal_message_cooldown <= 0 and not greeting_active:
            status_text = "FATAL: containment.flag ≠ expected"
            signal_message_cooldown = 8.0
        if signal_message_cooldown > 0:
            signal_message_cooldown -= dt
        if prethought_flash:
            prethought_flash["timer"] += dt
            if prethought_flash["timer"] >= prethought_flash["duration"]:
                prethought_flash = None

        for event in events:
            if event.type == pygame.USEREVENT + 1:
                face.set_talking(False)

    if passive:
        passive.stop()
    if vision:
        vision.stop()
    pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gerty avatar runtime")
    parser.add_argument(
        "--vision-check",
        action="store_true",
        help="Run a short camera/eye detection diagnostic and exit.",
    )
    parser.add_argument(
        "--vision-list",
        action="store_true",
        help="Probe camera indices and exit.",
    )
    parser.add_argument(
        "--vision-duration",
        type=float,
        default=8.0,
        help="Seconds to sample during the vision diagnostic.",
    )
    parser.add_argument(
        "--vision-index",
        type=int,
        default=None,
        help="Camera index override for the vision diagnostic.",
    )
    parser.add_argument(
        "--vision-max-index",
        type=int,
        default=5,
        help="Max index to probe when listing cameras.",
    )
    parser.add_argument(
        "--vision-verbose",
        action="store_true",
        help="Print every gaze event during the diagnostic run.",
    )
    args = parser.parse_args()

    if args.vision_list:
        list_cameras(max_index=args.vision_max_index)
    elif args.vision_check:
        run_vision_check(
            duration=args.vision_duration,
            verbose=args.vision_verbose,
            camera_index=args.vision_index,
        )
    else:
        main()
