"""
NeuroMod 테마 — 모든 색·폰트·간격·그래프 스타일은 여기 한 곳에서만 정의한다.

방향: 오래된 계측기 · SDR++ · WSJT-X · DAW 믹서. 중립 짙은 회색 바탕, 무채색 위주, 촘촘한 배치.
    · 네온 · 형광 · 빛 번짐 · 그라데이션 · 반투명 없음. 순수 검정 없음
    · 강조색(ACCENT)은 화면 전체에서 송신 버튼 하나에만 쓴다
    · 의미 색은 유지하되 채도를 낮춘 탁한 톤
        RX 수신 = 회청   TX 송신 = 황토   OK 동기·성공 = 올리브 초록   ERR 오류 = 벽돌   FEC 복구 = 겨자
    · 모서리 RADIUS 2px, 간격은 4px 배수(SPACE), 영역은 1px 선으로만 나눈다
    · 워터폴은 바꾸지 않는다: 워터폴 안 표시(대역선 · 코스타스 사각형)는 wf_* 로 따로 고정
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))

PALETTE = {
    # 바탕 (낮을수록 어둡다) — 아주 약간 따뜻한 중립 회색
    "bg0": "#1b1b1c",        # 창 · 패널 사이 틈
    "bg1": "#222223",        # 패널
    "bg2": "#2a2a2c",        # 입력칸 · 버튼
    "bg3": "#333335",        # 마우스 올림
    "bg4": "#3c3c3f",        # 누름 · 선택
    "plot": "#1c1c1d",       # 그래프 바탕
    # 선
    "line": "#303032",       # 1px 구분선
    "line2": "#434346",      # 테두리 · 강조 구분선
    "grid": "#2c2c2e",       # 그래프 격자
    # 글자
    # 글자 (명도비: text · text2 는 모든 바탕에서 4.5:1 이상, text3 은 3:1 이상 — contrast_report() 로 확인)
    "text": "#dedcd6",
    "text2": "#b0aea7",
    "text3": "#8f8d86",
    # 의미 색 (탁하게, 글자로 써도 바탕 bg1~bg3 에서 4:1 이상)
    "rx": "#86aab2",
    "tx": "#c09a60",
    "ok": "#8fab74",
    "err": "#dc887c",
    "fec": "#c2a55c",
    "qso_sep": "#8aa3c2",    # 수신 패널 메시지 구분선 (탁한 청회색 — 본문 · 의미 색과 구별)
    # 의미 색의 어두운 판 (막대 바탕 · 옅은 칠)
    "rx_dim": "#2b3335",
    "tx_dim": "#352f26",
    "ok_dim": "#2c3328",
    "err_dim": "#3a2b28",
    "fec_dim": "#36322a",
    # LED 꺼진 칸 (거의 안 보이게)
    "seg_off": "#29292b",
    # 강조색: 송신 버튼 하나만
    "accent": "#c9964f",
    "accent_hover": "#d6a661",
    "accent_text": "#1e1a14",
    # 범주 색 (비트쌍 00/01/10/11 구분 전용, 채도 낮게)
    "cat0": "#86a0bd",
    "cat1": "#bd92a2",
    "cat2": "#a0b47e",
    "cat3": "#b0afa9",
    # 워터폴 안 표시 (6단계 그대로 고정 — 워터폴은 손대지 않는다)
    "wf_band": "#22d3ee",
    "wf_a": "#22c55e",
    "wf_b": "#22d3ee",
    "wf_tent": "#e879f9",    # 확정 전 (CRC 전) 표지 · 구간 점선, 세 모드 공통 (09-27). 워터폴 · 구간 표시의 다른 색 (청록 · 초록 · 주황 · 연두 · 빨강 · 회색) 과 겹치지 않는 자홍
}
RADIUS = 2                       # px, 모든 모서리
SPACE = 4                        # px, 간격 기준 단위 (여백 · 간격은 이 배수만)
CTRL_H = 26                      # px, 버튼 · 입력칸 · 드롭다운 높이 (모두 같음)
TITLE_H = 26                     # px, 패널 제목줄 높이
GRID_ALPHA = 0.22                # 그래프 격자 불투명도 (모든 그래프 공통)
LINE_W = 1.0                     # 그래프 선 두께 (공통)

# 글꼴: 윈도우에서 힌팅이 잘 된, 진짜 Bold 파일이 있는 것만 (가변 폰트에 가짜 굵기를 씌우면 뭉개진다)
#   맑은 고딕 = malgun.ttf / malgunbd.ttf, Consolas = consola.ttf / consolab.ttf
UI_FONTS = ["Malgun Gothic", "Pretendard", "Segoe UI"]
MONO_FONTS = ["Consolas", "D2Coding", "Courier New"]
# 글자 크기 (pt). 본문 10 미만, 보조 9 미만은 쓰지 않는다
PT_BODY = 10.0                   # 본문 · 라벨 · 버튼 · 표
PT_SMALL = 9.0                   # 보조 문구 · 축 숫자 · 축 제목 · 배지
PT_VALUE = 11.0                  # 미터 값
PT_BIG = 14.0                    # 복원 여유 큰 숫자
AXIS_W = 60                      # px, 위아래로 쌓아 가로축을 맞추는 그래프들의 왼쪽 축 폭 (스펙트럼 · 워터폴 · 확대 워터폴)
SPEC_RANGE = (-130, -30)         # dBFS, 스펙트럼 기본 세로 범위 (고정). 입력 잡음 바닥 ~ 큰 신호
UI_PT = PT_BODY
MONO_PT = PT_BODY

WATERFALL_CMAPS = ["inferno", "magma", "viridis", "plasma", "cividis", "turbo"]
# 데이터 열지도(워터폴 · 코스타스 동기 맵)는 '차분한 색' 규칙의 예외: 인지적으로 균일하고 대비가 강한 컬러맵
SYNC_CMAP_DEFAULT = "magma"


def sp(n):
    """간격 n 단위 (px)"""
    return SPACE * n


def c(name):
    return PALETTE[name]


def pick_fonts():
    """설치된 폰트 중 첫 번째를 고른다. 없으면 대체 폰트."""
    from PySide6.QtGui import QFontDatabase
    fams = set(QFontDatabase.families())
    ui = next((f for f in UI_FONTS if f in fams), UI_FONTS[-1])
    mono = next((f for f in MONO_FONTS if f in fams), MONO_FONTS[-1])
    return ui, mono


_FONTS = {}


def ui_font(pt=None, bold=False):
    from PySide6.QtGui import QFont
    if not _FONTS:
        _FONTS["ui"], _FONTS["mono"] = pick_fonts()
    f = QFont(_FONTS["ui"])
    f.setPointSizeF(pt or UI_PT)
    f.setBold(bold)
    return f


def mono_font(pt=None, bold=False):
    """숫자용 고정폭 폰트 (자릿수가 흔들리지 않는다)"""
    from PySide6.QtGui import QFont
    if not _FONTS:
        _FONTS["ui"], _FONTS["mono"] = pick_fonts()
    f = QFont(_FONTS["mono"])
    f.setPointSizeF(pt or MONO_PT)
    f.setBold(bold)
    f.setStyleHint(QFont.Monospace)
    return f


def font_names():
    if not _FONTS:
        _FONTS["ui"], _FONTS["mono"] = pick_fonts()
    return _FONTS["ui"], _FONTS["mono"]


def load_qss():
    """style.qss 의 {{이름}} 자리에 팔레트 값을 넣어 돌려준다."""
    with open(os.path.join(HERE, "style.qss"), encoding="utf-8") as fp:
        s = fp.read()
    ui, mono = font_names()
    icons = _icons()
    vals = dict(PALETTE, radius="{}px".format(RADIUS), ui_font=ui, mono_font=mono, ui_pt="{}pt".format(UI_PT),
                small_pt="{}pt".format(PT_SMALL), ctrl_h="{}px".format(CTRL_H - 2), title_h="{}px".format(TITLE_H),
                **icons)
    for k in range(1, 7):
        vals["s{}".format(k)] = "{}px".format(sp(k))
    for k, v in vals.items():
        s = s.replace("{{" + k + "}}", str(v))
    return s


def _icons():
    """체크 표시 · 드롭다운 화살표를 팔레트 색으로 그린 작은 SVG 파일 (QSS 는 그림 파일만 받는다)"""
    d = os.path.join(HERE, "assets")
    os.makedirs(d, exist_ok=True)
    svgs = {
        "icon_check": '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 12 12">'
                      '<path d="M2.5 6.2 L5 8.6 L9.5 3.4" fill="none" stroke="{}" stroke-width="1.6"/></svg>'
                      .format(c("bg0")),
        "icon_arrow": '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="6" viewBox="0 0 10 6">'
                      '<path d="M1 1 L5 5 L9 1" fill="none" stroke="{}" stroke-width="1.3"/></svg>'.format(c("text2")),
        "icon_arrow_dis": '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="6" viewBox="0 0 10 6">'
                          '<path d="M1 1 L5 5 L9 1" fill="none" stroke="{}" stroke-width="1.3"/></svg>'.format(c("text3")),
    }
    out = {}
    for k, v in svgs.items():
        p = os.path.join(d, k + ".svg")
        try:
            if not os.path.exists(p) or open(p, encoding="utf-8").read() != v:
                with open(p, "w", encoding="utf-8") as fp:
                    fp.write(v)
        except OSError:
            pass
        out[k] = p.replace("\\", "/")
    return out


def setup_pyqtgraph():
    import pyqtgraph as pg
    try:
        from PySide6.QtWidgets import QApplication
        if QApplication.instance() is not None:
            QApplication.instance().setFont(ui_font())
    except Exception:
        pass
    pg.setConfigOptions(background=c("plot"), foreground=c("text2"), antialias=True,
                        imageAxisOrder="row-major")


def style_plot(p, title=None, xl=None, yl=None, grid=True):
    """pyqtgraph PlotWidget/PlotItem 을 테마에 맞춘다 (축·격자·글자 공통)."""
    import pyqtgraph as pg
    from PySide6.QtGui import QColor
    item = p.getPlotItem() if hasattr(p, "getPlotItem") else p
    for ax in ("left", "bottom", "right", "top"):
        a = item.getAxis(ax)
        a.setPen(pg.mkPen(c("line"), width=1))
        a.setTextPen(pg.mkPen(c("text2")))
        a.setStyle(tickFont=mono_font(PT_SMALL), tickLength=-4, tickTextOffset=3)
        a.enableAutoSIPrefix(False)      # 축 이름 옆 (x0.001) 같은 배율 표시를 끈다
    if grid:
        item.showGrid(x=True, y=True, alpha=GRID_ALPHA)
    if title:
        item.setTitle('<span style="color:{}; font-size:{}pt">{}</span>'.format(c("text2"), PT_SMALL, title))
    if xl is not None:
        item.setLabel("bottom", axis_label(xl))
    if yl is not None:
        item.setLabel("left", axis_label(yl))
    if hasattr(p, "setMenuEnabled"):
        p.setMenuEnabled(False)
    if hasattr(item, "hideButtons"):
        item.hideButtons()
    return p


def axis_label(text):
    """그래프 축 제목: 보조 글자 기준 (9pt, text2)"""
    return '<span style="color:{}; font-size:{}pt; font-family:\'{}\'">{}</span>'.format(
        c("text2"), PT_SMALL, font_names()[0], text)


def _lum(h):
    v = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    v = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in v]
    return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]


def contrast(fg, bg):
    """WCAG 명도비 (팔레트 이름)"""
    a, b = sorted((_lum(c(fg)), _lum(c(bg))), reverse=True)
    return (a + 0.05) / (b + 0.05)


# 글자색별 최소 명도비 (바탕 bg1 · bg2 · bg3 · plot 모두에서)
CONTRAST_RULES = {"qso_sep": 4.5, "text": 4.5, "text2": 4.5, "text3": 3.0, "rx": 4.5, "tx": 4.5, "ok": 4.5,
                  "err": 4.5, "fec": 4.5, "cat0": 4.5, "cat1": 4.5, "cat2": 4.5, "cat3": 4.5}


def contrast_report():
    """[(글자색, 바탕, 명도비, 기준)] 중 기준에 못 미치는 것 (없으면 빈 목록)"""
    bad = []
    for fg, need in CONTRAST_RULES.items():
        for bg in ("bg1", "bg2", "bg3", "plot"):
            r = contrast(fg, bg)
            if r < need:
                bad.append((fg, bg, round(r, 2), need))
    return bad


def cmap_lut(name, n=256):
    """워터폴 컬러맵 (pyqtgraph 에 없으면 matplotlib 에서 가져온다)"""
    import pyqtgraph as pg
    for src in (None, "matplotlib"):
        try:
            cm = pg.colormap.get(name, source=src) if src else pg.colormap.get(name)
            return cm.getLookupTable(nPts=n)
        except Exception:
            continue
    return pg.colormap.get("inferno").getLookupTable(nPts=n)


def pen(name, width=LINE_W, alpha=255, style=None):
    import pyqtgraph as pg
    from PySide6.QtGui import QColor
    col = QColor(c(name))
    col.setAlpha(alpha)
    kw = {"width": width}
    if style is not None:
        kw["style"] = style
    return pg.mkPen(col, **kw)


def brush(name, alpha=255):
    import pyqtgraph as pg
    from PySide6.QtGui import QColor
    col = QColor(c(name))
    col.setAlpha(alpha)
    return pg.mkBrush(col)


def qcolor(name, alpha=255):
    from PySide6.QtGui import QColor
    col = QColor(c(name))
    col.setAlpha(alpha)
    return col
