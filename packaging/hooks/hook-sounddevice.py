# Include only the default non-ASIO Windows PortAudio runtime.
from pathlib import Path
from PyInstaller.utils.hooks import get_module_file_attribute
module_dir = Path(get_module_file_attribute('sounddevice')).parent
data_dir = module_dir / '_sounddevice_data' / 'portaudio-binaries'
runtime = data_dir / 'libportaudio64bit.dll'
assert runtime.is_file(), 'Expected Windows x64 non-ASIO PortAudio runtime'
binaries = [(str(runtime), str(data_dir.relative_to(module_dir)))]
datas = [(str(data_dir / 'README.md'), str(data_dir.relative_to(module_dir)))]
