"""
화면 문구 — 한곳에 모음 (22부 시범, 23부 영어 추가). 코드는 키로만 부른다:

    from strings_ko import tr
    tr("vfo.center")                     # → "중심" / "Offset"
    tr("main.max_tx_time_exceeded").format(300)   # 문구 안 {} 는 그대로, 호출하는 쪽이 .format

언어: 설정 '일반 · 언어' (settings.json "language", ko / en). 앱 시작 때 한 번 정함 (바꾸면 재시작 후 적용).
환경 변수 TNG_LANG 이 있으면 그것이 먼저 (시험 · 스크린샷용).
영어 문구는 strings_en.py 에 같은 키. 영어에 없는 키는 한국어로, 둘 다 없으면 키 그대로 (빠진 문구가 화면에서 바로 보이게).
"""
import json
import os

LANGS = (("ko", "한국어"), ("en", "English"))


def os_lang():
    """Windows 표시 언어가 한국어면 ko, 그 밖은 en (첫 실행 기본값, 25부). TNG_OS_LANG = 시험용 흉내"""
    if os.environ.get("TNG_OS_LANG") in ("ko", "en"):
        return os.environ["TNG_OS_LANG"]
    try:
        import ctypes
        return "ko" if (ctypes.windll.kernel32.GetUserDefaultUILanguage() & 0x3FF) == 0x12 else "en"
    except Exception:
        import locale
        return "ko" if (locale.getlocale()[0] or "").lower().startswith(("ko", "korean")) else "en"


def _read_lang():
    """TNG_LANG → 설정 파일 language → (설정 파일 없음 = 첫 실행) OS 표시 언어. 옛 설정 파일에 language 가 없으면 ko"""
    v = os.environ.get("TNG_LANG")
    if not v:
        try:
            from tngpkt import app_paths
            path = app_paths.settings_path()
            if not os.path.exists(path):
                return os_lang()
            with open(path, encoding="utf-8") as fp:
                v = json.load(fp).get("language")
        except Exception:
            v = None
    return v if v in ("ko", "en") else "ko"


LANG = _read_lang()

KO = {}                     # 아래 _strings_ko_data 에서 채움

from tngpkt._strings_ko_data import KO as _KO   # noqa: E402
KO.update(_KO)
_EN = None


def set_lang(lang):
    """시험용: 언어 바꾸기 (앱은 시작 때 한 번만)"""
    global LANG
    LANG = lang if lang in ("ko", "en") else "ko"


def tr(key, **kw):
    """키 → 문구 (지금 언어). 없는 키는 한국어 → 키 그대로"""
    global _EN
    s = None
    if LANG == "en":
        if _EN is None:
            try:
                from tngpkt.strings_en import EN as _e
                _EN = _e
            except Exception:
                _EN = {}
        s = _EN.get(key)
    if s is None:
        s = KO.get(key, key)
    return s.format(**kw) if kw else s


_FR = None


def trf(s):
    """조각 번역 (23부): 수신기 · 결과 분류처럼 로직 안에서 한국어로 조립되는 문구를 화면에 넣기 직전에 바꾼다.
    한국어 모드는 그대로 돌려줌 (동작 변화 없음). 조각 = strings 의 frag.* 키, 긴 것부터"""
    global _FR
    if LANG != "en" or not s or not isinstance(s, str):
        return s
    if _FR is None:
        tr("frag.x")                                   # _EN 읽기
        pairs = [(v, _EN.get(k, "")) for k, v in KO.items() if k.startswith("frag.")]
        pairs.sort(key=lambda p: -len(p[0]))
        import re
        _FR = (re.compile("|".join(re.escape(p[0]) for p in pairs)), dict(pairs))
    rx, d = _FR

    def rep(m):
        e = d.get(m.group(0), m.group(0))
        a, b = m.start(), m.end()
        if e and a > 0 and m.string[a - 1].isalnum() and not e.startswith("x "):
            e = " " + e
        if e and (a == 0 or m.string[:a].strip() == ""):
            e = e[0].upper() + e[1:]
        if e and b < len(m.string) and m.string[b].isalnum():
            e = e + " "
        return e
    return rx.sub(rep, s)
