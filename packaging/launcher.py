"""Windows packaging entry point; application UI stays in neuromod_app."""
import multiprocessing
import os
import runpy
import sys

if __name__ == '__main__':
    multiprocessing.freeze_support()
    if '--diag-worker' in sys.argv:
        sys.argv.remove('--diag-worker')
        # PyInstaller windowed mode clears Python streams. QProcess supplies pipes.
        sys.stdout = open(1, 'w', encoding='utf-8', closefd=False)
        sys.stderr = open(2, 'w', encoding='utf-8', closefd=False)
        import diag
        diag.main()
    else:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('6L5TNG.TNGPKT')
        from PySide6.QtGui import QIcon
        from PySide6.QtWidgets import QApplication
        import app_paths
        app = QApplication(sys.argv[:1])
        app.setApplicationName('TNG PKT')
        app.setWindowIcon(QIcon(app_paths.resource('assets/tng-pkt.ico')))
        runpy.run_module('neuromod_app', run_name='__main__')
