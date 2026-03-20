# src/onesec/analyzer/__init__.py
from onesec.analyzer.base import Analyzer
from onesec.analyzer.scene import SceneAnalyzer
from onesec.analyzer.audio import AudioAnalyzer
from onesec.analyzer.motion import MotionAnalyzer

# Level-2 analyzers (require optional extras)
try:
    from onesec.analyzer.clip_scorer import ClipScorer
except ImportError:
    pass

try:
    from onesec.analyzer.whisper import WhisperAnalyzer
except ImportError:
    pass

__all__ = [
    "Analyzer",
    "SceneAnalyzer",
    "AudioAnalyzer",
    "MotionAnalyzer",
    *( ["ClipScorer"] if "ClipScorer" in dir() else []),
    *( ["WhisperAnalyzer"] if "WhisperAnalyzer" in dir() else []),
]
