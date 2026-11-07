"""Fetch the pinned public TabICL classifier checkpoint; never uploads data."""

import hashlib
from pathlib import Path
import tempfile
from urllib.request import urlopen

NAME = "tabicl-classifier-v2-20260212.ckpt"
REVISION = "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
SHA256 = "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"
SIZE = 110368038
URL = f"https://huggingface.co/jingang/TabICL/resolve/{REVISION}/{NAME}"


def verify(path):
    if path.stat().st_size != SIZE or hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError(f"Checkpoint checksum mismatch: {path}")


def main():
    path = Path("data/raw/models") / NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        verify(path)
    else:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as target:
            temporary = Path(target.name)
            try:
                with urlopen(URL, timeout=60) as response:
                    count = 0
                    while chunk := response.read(1024 * 1024):
                        count += len(chunk)
                        if count > SIZE:
                            raise ValueError("Checkpoint exceeds pinned size")
                        target.write(chunk)
                target.flush()
                verify(temporary)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    print(f"Verified {path}: {SHA256}")


if __name__ == "__main__":
    main()
