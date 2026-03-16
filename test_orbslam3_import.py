from __future__ import annotations

import pyorbslam3


def main() -> None:
    print("pyorbslam3 import ok")
    print("sensors:", [name for name in dir(pyorbslam3.Sensor) if name.isupper()])


if __name__ == "__main__":
    main()
