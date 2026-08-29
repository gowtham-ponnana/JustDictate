# -*- mode: python ; coding: utf-8 -*-
import os
import site

from PyInstaller.utils.hooks import collect_all

site_packages = os.path.join(
    os.path.dirname(site.getsitepackages()[0]),
    "lib", "python3.11", "site-packages"
) if not site.getsitepackages()[0].endswith("site-packages") else site.getsitepackages()[0]

for sp in site.getsitepackages():
    if sp.endswith("site-packages"):
        site_packages = sp
        break

datas = [
    (os.path.join(site_packages, '_sounddevice_data'), '_sounddevice_data'),
]
binaries = [
    (os.path.join(site_packages, '_sounddevice_data/portaudio-binaries/libportaudio.dylib'), '_sounddevice_data/portaudio-binaries'),
]
hiddenimports = [
    'rumps',
    'pynput',
    'pynput.keyboard',
    'pynput.keyboard._darwin',
    'sounddevice',
    'numpy',
    'mlx',
    'mlx.core',
    'mlx.nn',
    'mlx.utils',
    'parakeet_mlx',
    'parakeet_mlx.audio',
    'parakeet_mlx.alignment',
    'librosa',
    'huggingface_hub',
    'dacite',
    'AppKit',
    'Foundation',
    'CoreFoundation',
    'Quartz',
    'CoreText',
    'HIServices',
    'objc',
    'PyObjCTools',
    'PyObjCTools.AppHelper',
    'click',
    'tqdm',
    'yaml',
    'certifi',
    'cffi',
    'hf_xet',
]

for pkg in (
    'mlx',
    'parakeet_mlx',
    'librosa',
    'huggingface_hub',
    'numba',
    'scipy',
    'sklearn',
    'dacite',
    'soxr',
):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

a = Analysis(
    ['just_dictate.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='JustDictate',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='JustDictate',
)

app = BUNDLE(
    coll,
    name='JustDictate.app',
    icon='icon.icns',
    bundle_identifier='com.gowtham.just-dictate',
    info_plist={
        'CFBundleDisplayName': 'JustDictate',
        'CFBundleName': 'JustDictate',
        'CFBundleShortVersionString': '0.1.0',
        'CFBundleVersion': '0.1.0',
        'LSUIElement': True,
        'LSMinimumSystemVersion': '13.0',
        'NSHighResolutionCapable': True,
        'NSMicrophoneUsageDescription': 'JustDictate needs microphone access to record speech for transcription.',
        'NSAppleEventsUsageDescription': 'JustDictate needs to send keystrokes to type transcribed text.',
    },
)
