"""Copy a selected ResNet-18 checkpoint beside the corresponding PID settings."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    profile = args.directory / "pid.json"
    settings = json.loads(profile.read_text())
    target = args.directory / "cnn.pth"
    digest = checksum(args.checkpoint)
    if target.exists():
        if checksum(target) != digest:
            parser.error(f"{target} exists with different weights")
        if settings.get("cnn_sha256") == digest:
            print(f"Existing checkpoint verified: {target}")
            return
    else:
        temporary = target.with_suffix(".pth.tmp")
        shutil.copyfile(args.checkpoint, temporary)
        if checksum(temporary) != digest:
            raise OSError(f"Copied checkpoint hash differs: {temporary}")
        temporary.replace(target)
    settings.update(status="trained", cnn_sha256=digest, cnn_source=str(args.checkpoint))
    temporary_profile = profile.with_suffix(".json.tmp")
    temporary_profile.write_text(json.dumps(settings, indent=2) + "\n")
    temporary_profile.replace(profile)
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
