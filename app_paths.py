"""
TNG PKT 경로 — 사용자 쓰기 폴더와 읽기 전용 자원 폴더를 한 곳에서 정한다.

  · user_dir()      : %APPDATA%\\TNG PKT  (설정 · 레이아웃 · 로그). 프로그램 폴더에는 쓰지 않는다.
  · resource(rel)   : 프로그램 폴더 기준 읽기 전용 자원 (models/*.pt, style.qss …).
                      EXE 로 묶였을 때(PyInstaller)는 sys._MEIPASS 기준.
"""
import os
import sys

APP_NAME = "TNG PKT"


def user_dir():
    base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Roaming")
    d = os.path.join(base, APP_NAME)
    os.makedirs(d, exist_ok=True)
    return d


def resource_dir():
    return getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(__file__))


def resource(rel):
    return os.path.join(resource_dir(), rel)


def settings_path():
    return os.path.join(user_dir(), "settings.json")


def layout_path():
    return os.path.join(user_dir(), "layout.ini")


def default_log_dir():
    return os.path.join(user_dir(), "logs")
