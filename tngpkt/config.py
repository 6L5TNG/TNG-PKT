"""
공통 설정 (모든 단계가 이 파일의 값을 사용한다).

설정값은 반드시 여기 한 곳에만 둔다.
"""

from dataclasses import dataclass, asdict, field
import torch


@dataclass
class Config:
    # ---- 변조 구조 ----
    # 한 번에 전송하는 비트 수 (k) 와, 그 비트를 실어 보낼 복소 채널 사용 횟수 (n)
    # 전송률 = k / n [bit / 복소심볼]
    # 기본값 k=2, n=1 → QPSK 와 같은 전송률(2 bit/심볼).
    #   * QPSK 의 Eb/N0 대비 BER 이론값은 BPSK 와 동일하므로
    #     1단계 검증에서 BPSK 이론 곡선을 그대로 기준선으로 쓸 수 있다.
    bits_per_block: int = 2      # k
    channel_uses: int = 1        # n

    # ---- 신경망 크기 ----
    tx_hidden: tuple = (128, 128)
    rx_hidden: tuple = (128, 128)

    # ---- 학습 ----
    batch_size: int = 4096
    steps: int = 20000           # 학습 반복 횟수
    lr: float = 1e-3
    lr_min: float = 1e-5         # 코사인 스케줄 최저 학습률
    # 학습 중 매 배치마다 이 범위에서 Eb/N0 를 무작위로 뽑는다 [dB]
    train_ebn0_min_db: float = 0.0
    train_ebn0_max_db: float = 12.0
    log_every: int = 500
    seed: int = 1234

    # ---- 평가 ----
    eval_ebn0_db: tuple = (0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.)
    eval_min_bit_errors: int = 200      # 각 지점에서 이만큼 비트오류가 쌓일 때까지 반복
    eval_max_blocks: int = 4_000_000    # 안전 상한 (블록 수)
    eval_batch: int = 100_000

    # ---- 경로 ----
    model_path: str = "models/stage1.pt"
    result_dir: str = "results"

    # ---- 장치 ----
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    @property
    def rate(self) -> float:
        """전송률 [bit / 복소심볼]"""
        return self.bits_per_block / self.channel_uses

    def to_dict(self):
        return asdict(self)


CFG = Config()


@dataclass
class Stage2Config:
    """
    2단계: 시간축 파형 + 현실적인 채널.

    1단계는 추상적인 복소 심볼 하나만 다뤘지만, 주파수/타이밍 오차와 다중경로를
    다루려면 '샘플레이트를 가진 실제 파형'이 필요하다. 그래서 2단계부터는
    복소 기저대역 파형(샘플레이트 fs_base)을 직접 만들고 채널에 통과시킨다.
    (3단계에서 이 기저대역을 1500Hz 로 올려 8kHz 오디오 WAV 로 만든다.)
    """

    # ---- 파형 / 대역폭 ----
    fs_base: float = 2000.0      # 복소 기저대역 샘플레이트 [Hz]
    sps: int = 8                 # 심볼당 샘플 수 → 심볼률 = fs_base/sps = 250 baud
    bits_per_symbol: int = 2     # k
    symbols_per_frame: int = 32  # S  → 프레임 = 256 샘플 = 128ms, 64비트, 500bps

    # 점유대역폭을 구조적으로 보장하는 고정 저역통과 필터 (학습되지 않음)
    lpf_cutoff_hz: float = 200.0   # 편측 -6dB 차단주파수. 저지대역(-80dB)이 250Hz 안에 들어와
    #                               어떤 정의로 재도 점유대역폭 ≤ 500Hz 가 보장된다
    lpf_taps: int = 129
    lpf_trans_hz: float = 40.0     # 천이대역 폭 (카이저 창 설계에 사용)

    audio_fs: float = 8000.0       # 3단계 오디오 샘플레이트
    audio_center_hz: float = 1500.0  # SSB 오디오 중심 주파수

    # ---- 신경망 크기 ----
    pos_embed_dim: int = 16
    tx_hidden: tuple = (256, 256)
    rx_channels: int = 96
    rx_dilations: tuple = (1, 2, 4, 8, 16)

    # ---- 채널 손상 최대치 (커리큘럼 난이도 1.0 일 때의 값) ----
    max_phase_rad: float = 3.14159265358979      # ±180° → 사실상 0~360°
    max_freq_off_hz: float = 10.0                # ±10 Hz
    max_timing_samp: float = 4.0                 # ±4 샘플 = ±0.5 심볼 (분수 지연 포함)
    multipath_delay_ms: tuple = (1.0, 3.0)       # 반사파 지연 범위
    max_paths: int = 2                           # 반사파 개수 (0~2개)
    max_reflect_amp: float = 0.7                 # 반사파 진폭 (주경로 대비)
    max_level_db: float = 20.0                   # 수신 레벨 변화 ±20 dB
    frac_delay_half: int = 8                     # 분수 지연 sinc 커널 반폭
    guard_samples: int = 40                      # 프레임 앞뒤 여유 구간

    # ---- 학습 ----
    batch_size: int = 256          # 프레임 수
    steps: int = 30000
    lr: float = 2e-3
    lr_min: float = 1e-5
    warmup_frac: float = 0.10      # 난이도 0 으로 시작하는 구간
    curriculum_frac: float = 0.60  # 이 시점까지 난이도를 0→1 로 올린다
    train_ebn0_min_db: float = 2.0
    train_ebn0_max_db: float = 16.0
    log_every: int = 500
    seed: int = 20250923

    # ---- 평가 ----
    eval_ebn0_db: tuple = (2., 4., 6., 8., 10., 12., 14., 16., 18.)
    eval_min_bit_errors: int = 300
    eval_max_frames: int = 300_000
    eval_batch: int = 2048

    # ---- 경로 ----
    model_path: str = "models/stage2.pt"
    result_dir: str = "results"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    @property
    def frame_len(self) -> int:
        """프레임 한 개의 기저대역 샘플 수"""
        return self.symbols_per_frame * self.sps

    @property
    def bits_per_frame(self) -> int:
        return self.symbols_per_frame * self.bits_per_symbol

    @property
    def symbol_rate(self) -> float:
        return self.fs_base / self.sps

    @property
    def bit_rate(self) -> float:
        return self.symbol_rate * self.bits_per_symbol

    @property
    def frame_ms(self) -> float:
        return 1000.0 * self.frame_len / self.fs_base

    def to_dict(self):
        return asdict(self)


CFG2 = Stage2Config()


@dataclass
class Stage3Config:
    """
    3단계: 연속 스트림 학습 + 텍스트 모뎀.

    2단계와 달라진 핵심:
      * 송신 NN 이 프레임을 하나씩 따로 만들지 않는다. 메시지 전체를 하나의 연속
        스트림으로 만든 뒤 고정 LPF 를 **딱 한 번** 통과시킨다. 프레임 경계가
        학습 안으로 들어오므로 경계 불연속(스플래터)이 구조적으로 사라진다.
      * 수신 NN 은 프레임 앞뒤로 여유 구간(ctx_symbols)을 함께 본다. LPF 가
        경계 너머로 번지는 만큼을 보고 판단하게 하기 위함이다.
      * 프레임 = 헤더 2바이트 + 페이로드 8바이트 + CRC16 2바이트 = 96비트.
    """

    # ---- 파형 ----
    fs_base: float = 2000.0
    sps: int = 8                      # → 250 baud
    bits_per_symbol: int = 2
    symbols_per_frame: int = 48       # → 프레임 384샘플 = 192ms = 96비트
    ctx_symbols: int = 4              # 수신이 프레임 앞뒤로 더 보는 심볼 수

    lpf_cutoff_hz: float = 200.0
    lpf_taps: int = 129
    lpf_trans_hz: float = 40.0

    # ---- 오디오 ----
    audio_fs: float = 8000.0
    audio_center_hz: float = 1500.0
    ramp_ms: float = 20.0             # 송신 시작/끝 램프 길이
    noise_band_hz: tuple = (250.0, 2750.0)   # SNR 기준 대역 (폭 2500Hz)

    # ---- 대역폭 합격 기준 (3단계부터 모든 단계 공통) ----
    bw_limit_hz: float = 500.0
    bw_criterion_db: float = -60.0    # 연속 송신 스트림 기준 -60dB 대역폭

    # ---- 신경망 ----
    pos_embed_dim: int = 16
    tx_hidden: tuple = (256, 256)
    rx_channels: int = 96
    rx_dilations: tuple = (1, 2, 4, 8, 16)

    # ---- 채널 손상 최대치 ----
    max_phase_rad: float = 3.14159265358979
    max_freq_off_hz: float = 10.0
    max_timing_samp: float = 4.0
    multipath_delay_ms: tuple = (1.0, 3.0)
    max_paths: int = 2
    max_reflect_amp: float = 0.7
    max_level_db: float = 20.0
    frac_delay_half: int = 8
    guard_samples: int = 48

    # ---- 학습 ----
    frames_per_stream: int = 3        # 스트림 하나에 프레임 3개, 가운데 것만 복호
    batch_size: int = 192
    steps: int = 60000
    lr: float = 2e-3
    lr_min: float = 1e-5
    warmup_frac: float = 0.10
    curriculum_frac: float = 0.60
    train_ebn0_min_db: float = 2.0
    train_ebn0_max_db: float = 16.0
    log_every: int = 500
    seed: int = 20250924

    # ---- 평가 ----
    eval_ebn0_db: tuple = (2., 4., 6., 8., 10., 12., 14., 16., 18.)
    eval_min_bit_errors: int = 300
    eval_max_frames: int = 200_000
    eval_batch: int = 1024

    # ---- 경로 ----
    model_path: str = "models/stage3.pt"
    result_dir: str = "results"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    @property
    def frame_len(self) -> int:
        return self.symbols_per_frame * self.sps

    @property
    def ctx_len(self) -> int:
        return self.ctx_symbols * self.sps

    @property
    def win_len(self) -> int:
        """수신 NN 이 한 번에 보는 길이 = 프레임 + 앞뒤 여유"""
        return self.frame_len + 2 * self.ctx_len

    @property
    def bits_per_frame(self) -> int:
        return self.symbols_per_frame * self.bits_per_symbol

    @property
    def symbol_rate(self) -> float:
        return self.fs_base / self.sps

    @property
    def bit_rate(self) -> float:
        return self.symbol_rate * self.bits_per_symbol

    @property
    def frame_ms(self) -> float:
        return 1000.0 * self.frame_len / self.fs_base

    def to_dict(self):
        return asdict(self)


CFG3 = Stage3Config()


@dataclass
class PreambleConfig:
    """
    5단계 추가: 프리앰블/포스트앰블 (코스타스 배열).

    송신 = [1500Hz 톤 300ms] [코스타스 A 224ms] [AI 데이터 (4단계 그대로)] [코스타스 B 224ms]
    AI 변조/복조와 학습된 모델은 건드리지 않는다.
    """
    tone_ms: float = 300.0             # 무전기 송신 전환·AGC 안정용 희생 구간
    costas_a: tuple = (3, 1, 4, 0, 6, 5, 2)          # FT8 과 같은 배열
    costas_b: tuple = (0, 6, 4, 5, 1, 3, 2)          # 199개 중 파형 교차상관 최소권 + 조합 최대일치 2 (근거는 README)
    n_tones: int = 7
    tone_spacing_hz: float = 62.5      # 1500Hz 중심 → 1312.5 ~ 1687.5Hz
    symbol_ms: float = 32.0            # 톤당 길이 (변조지수 h = 62.5*0.032 = 2)
    freq_smooth_ms: float = 4.0        # 톤 전환부 주파수를 부드럽게 (스플래터 방지)
    ramp_start_ms: float = 20.0        # 송신 시작 램프 (희생 톤 안)
    ramp_end_ms: float = 10.0          # 송신 끝 램프 (코스타스 B 마지막 톤 안)
    ramp_join_ms: float = 5.0          # 구간 경계 램프
    # 프리앰블/포스트앰블 생성에 AI 송신과 **같은 고정 LPF** 를 넣는다.
    # 없으면 CPFSK 톤 전환 부엽·경계 램프 때문에 짧은 메시지에서 -60dB 대역폭이
    # 831Hz 까지 벌어졌다 (측정). AI 데이터 구간은 다시 필터링하지 않는다.
    lpf_cutoff_hz: float = 200.0
    lpf_taps: int = 129

    # 수신: 시간-주파수 2차원 상관
    max_freq_off_hz: float = 100.0     # 주파수 오차 탐색 범위 ±
    search_lpf_hz: float = 330.0       # 탐색 전 잡음 제거 필터 (톤 187.5 + 오차 100 + 여유)
    coarse_step: int = 4               # 시간 탐색 간격 [샘플 @2000Hz] = 2ms
    coarse_nfft: int = 1024            # 주파수 격자 약 2Hz
    fine_nfft: int = 8192              # 정밀 주파수 격자 약 0.24Hz
    detect_threshold: float = 0.18     # rho^2 문턱. 잡음 전용 10분 최대 0.116 → 꼬리 외삽으로 여유를 둠
    max_message_s: float = 120.0       # A 이후 B 를 찾을 최대 거리
