# Filter unused plugins before shared-library dependency collection.
from pathlib import Path
from PyInstaller.utils.hooks.qt import add_qt6_dependencies
hiddenimports, binaries, datas = add_qt6_dependencies(__file__)
binaries = [(src, dst) for src, dst in binaries
            if Path(src).name.lower() not in ('qtvirtualkeyboardplugin.dll', 'qpdf.dll')]
