#!/usr/bin/env python3
"""从明确指定的官方归档安装固定Sharpa SDK；运行时不依赖归档所在目录。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from zipfile import ZipFile

VERSION = "5.0.11"
ARCHIVE_SHA256 = "35b7667975a91fba700e42f2e73a8dead871057575de66cd0da59c4bc5f6bbb6"
RESOURCE_ROOT = "sharpa_teleop_wave_resources/"
DEFAULT_DESTINATION = (
    Path(__file__).resolve().parents[1]
    / "Motorhand_FFB/.cache/hardware-sdk/sharpa"
    / VERSION
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install(archive: Path, destination: Path) -> int:
    """完整保留资源与许可，只接受已固定的官方分发。"""
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("Sharpa SDK归档SHA256不匹配，未安装任何文件")
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sharpa-install-", dir=destination.parent) as temporary:
        staged = Path(temporary) / VERSION
        staged.mkdir()
        with ZipFile(archive) as wheel:
            for member in wheel.infolist():
                if not member.filename.startswith(RESOURCE_ROOT):
                    continue
                relative = PurePosixPath(member.filename[len(RESOURCE_ROOT):])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Sharpa SDK归档路径无效")
                target = staged.joinpath(*relative.parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with wheel.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                mode = (member.external_attr >> 16) & 0o777
                if mode:
                    target.chmod(mode)
        required = (
            "LICENSE", "NOTICE", "README.md", "VERSION",
            "include/SharpaWaveSDK.h", "lib/libsharpa-wave-sdk.so",
        )
        if not all((staged / name).is_file() for name in required):
            raise ValueError("Sharpa SDK分发缺少完整许可、头文件或真实SDK库")
        values = dict(line.split("=", 1) for line in (staged / "VERSION").read_text().splitlines() if "=" in line)
        if values.get("VERSION") != VERSION:
            raise ValueError("Sharpa SDK版本不匹配")
        hashes = {
            path.relative_to(staged).as_posix(): sha256(path)
            for path in sorted(staged.rglob("*")) if path.is_file()
        }
        manifest = {"version": VERSION, "archive_sha256": ARCHIVE_SHA256, "files_sha256": hashes}
        (staged / "installation.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        if destination.exists():
            if all((destination / name).is_file() and sha256(destination / name) == digest for name, digest in hashes.items()):
                return len(hashes)
            raise FileExistsError("目标SDK目录已有不同或不完整文件；请指定新目录或先人工移走原目录")
        os.replace(staged, destination)
        return len(hashes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="明确指定Sharpa 5.0.11官方资源wheel")
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    arguments = parser.parse_args()
    try:
        count = install(arguments.archive.expanduser().resolve(), arguments.destination.expanduser())
    except (OSError, ValueError) as error:
        parser.exit(1, f"安装失败：{error}\n")
    print(f"Sharpa SDK {VERSION}已验证：{count}个资源文件，目录：{arguments.destination.resolve()}")


if __name__ == "__main__":
    main()
