from pathlib import Path
import ast
import os
import torch
from PyInstaller.utils.hooks import copy_metadata
root=Path(SPECPATH)
source=Path(os.environ.get('TNG_PKT_SOURCE',root.parent)).resolve()
assert torch.version.cuda is None and '+cpu' in torch.__version__, 'CPU torch required'
version={}
for node in ast.parse((source/'registry.py').read_text(encoding='utf-8')).body:
    if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant):
        version.update({t.id:node.value.value for t in node.targets if isinstance(t,ast.Name)})
v=tuple(map(int,version['APP_VERSION'].split('.')))+(version['APP_BUILD'],)
ver=Path(os.environ.get('TNG_PKT_BUILD_ROOT',str(root)))/'version-info.txt'
ver.write_text("VSVersionInfo(ffi=FixedFileInfo(filevers=%r,prodvers=%r,mask=0x3f,flags=0x0,OS=0x40004,fileType=0x1,subtype=0x0,date=(0,0)),kids=[StringFileInfo([StringTable('040904B0',[StringStruct('CompanyName','6L5TNG'),StringStruct('FileDescription','TNG PKT'),StringStruct('FileVersion','%s'),StringStruct('ProductName','TNG PKT'),StringStruct('ProductVersion','%s'),StringStruct('OriginalFilename','TNG PKT.exe')])]),VarFileInfo([VarStruct('Translation',[1033,1200])])])"%(v,v,version['APP_VERSION'],version['APP_VERSION']),encoding='utf-8')
datas=[(str(source/'models'),'models'),(str(source/'assets'),'assets'),(str(source/'style.qss'),'.'),(str(source/'LICENSE'),'.'),(str(source/'THIRD_PARTY_NOTICES.md'),'.'),(str(source/'THIRD_PARTY_LICENSES.txt'),'.')]
# Only distributions used by the runtime About library list.
for distribution in ('PySide6', 'torch', 'scipy', 'pyqtgraph', 'sounddevice'):
    datas += copy_metadata(distribution)
hidden=[p.stem for p in source.glob('*.py')]
a=Analysis([str(root/'launcher.py')],pathex=[str(source)],binaries=[],datas=datas,hiddenimports=hidden,hookspath=[str(root/'hooks')],runtime_hooks=[],excludes=['tkinter','PyQt5','PyQt6','PySide2','onnx','onnxruntime','torchvision','torchaudio','pytest','IPython','notebook','tensorboard','PySide6.QtVirtualKeyboard','PySide6.QtPdf','PySide6.QtPdfWidgets'],noarchive=False)
# Unused automatic compiler/software-renderer collection is not distributed.
omit = {'opengl32sw.dll', 'protoc.exe'}
a.binaries = [item for item in a.binaries if Path(item[0]).name.lower() not in omit]
# Exact audited LGPL-capable Qt module set. Unknown modules require review.
qt_allowed = {'qt6core.dll','qt6gui.dll','qt6widgets.dll','qt6network.dll',
              'qt6opengl.dll','qt6openglwidgets.dll','qt6svg.dll','qt6test.dll',
              'qt6printsupport.dll'}
assert not any('virtualkeyboard' in str(item).lower() or '-asio.dll' in str(item).lower()
               for item in a.binaries + a.datas), 'Unexpected GPL Virtual Keyboard or ASIO runtime'
assert all(Path(item[0]).name.lower() in qt_allowed for item in a.binaries
           if Path(item[0]).name.lower().startswith('qt6') and Path(item[0]).suffix.lower() == '.dll'), 'Unexpected Qt module: license review required'
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,[],exclude_binaries=True,name='TNG PKT',debug=False,bootloader_ignore_signals=False,strip=False,upx=False,console=False,icon=str(source/'assets/tng-pkt.ico'),version=str(ver))
coll=COLLECT(exe,a.binaries,a.datas,strip=False,upx=False,name='TNG PKT')
