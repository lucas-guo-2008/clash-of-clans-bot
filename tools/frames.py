"""The saved frames the tests run on, published as a GitHub Release asset so a fresh clone (and CI) can run
every test without the frames in git (68 frames, ~205 MB; they are lossless on purpose).

    python -m tools.frames pack    # zip the local frames the tests use, and write tests/frames.txt
    python -m tools.frames fetch   # download the zip for tests/frames.txt and unpack what is missing

`pack` is for when a test starts using a new frame: zip, bump TAG, upload the zip to a release named TAG,
commit tests/frames.txt. `fetch` checks every file against the sha256 in tests/frames.txt.
"""

import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "templates"
MANIFEST = ROOT / "tests" / "frames.txt"
TAG = "frames-v1"
URL = f"https://github.com/lucas-guo-2008/clash-of-clans-bot/releases/download/{TAG}/frames.zip"
ZIP = ROOT / "frames.zip"  # written by pack for the upload (gitignored)

# Every frame the suite loads (logged from a full run, 2026-11-20): all of initial_collection, and these
# unrecognized ones.
UNKNOWN = (
    "unknown_20260919_005437_204263.png", "unknown_20260919_005539_938030.png",
    "unknown_20260919_005548_874261.png", "unknown_20260919_005553_346372.png",
    "unknown_20260919_005725_732450.png", "unknown_20260930_222752_292758.png",
    "unknown_20261116_211314_194623.png", "unknown_20261116_211329_173888.png",
    "unknown_20261116_214340_253710.png", "unknown_20261116_214355_239508.png",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pack() -> None:
    files = sorted((TEMPLATES / "initial_collection").glob("adb_frame_*.png"),
                   key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    files += [TEMPLATES / "unknown" / name for name in UNKNOWN]
    lines = []
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_STORED) as z:  # PNGs do not compress further
        for path in files:
            rel = path.relative_to(TEMPLATES).as_posix()
            data = path.read_bytes()
            z.writestr(rel, data)
            lines.append(f"{sha256(data)}  {rel}")
    MANIFEST.write_text(f"# {TAG}: python -m tools.frames fetch\n" + "\n".join(lines) + "\n")
    print(f"{len(files)} frames, {ZIP.stat().st_size / 1e6:.0f} MB -> {ZIP.name}; upload it to a release tagged {TAG}")


def fetch() -> int:
    want = {}
    for line in MANIFEST.read_text().splitlines():
        if line and not line.startswith("#"):
            digest, rel = line.split(maxsplit=1)
            want[rel] = digest
    missing = [rel for rel in want if not (TEMPLATES / rel).exists()]
    if not missing:
        print(f"all {len(want)} frames present")
        return 0
    print(f"downloading {URL} for {len(missing)} missing frames")
    with urllib.request.urlopen(URL) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    for rel in missing:
        data = archive.read(rel)
        if sha256(data) != want[rel]:
            print(f"{rel}: checksum does not match tests/frames.txt", file=sys.stderr)
            return 1
        (TEMPLATES / rel).parent.mkdir(parents=True, exist_ok=True)
        (TEMPLATES / rel).write_bytes(data)
    print(f"unpacked {len(missing)} frames")
    return 0


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "pack":
        pack()
    elif command == "fetch":
        sys.exit(fetch())
    else:
        sys.exit(__doc__)
