"""Windows 콘솔에서 한글이 깨지지 않도록 표준출력을 UTF-8 로 맞춘다."""
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
