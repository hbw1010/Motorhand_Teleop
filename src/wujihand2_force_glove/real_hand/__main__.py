"""python -m wujihand2_force_glove.real_hand 的显式安全入口。"""

from .runtime import main


if __name__ == "__main__":
    raise SystemExit(main())
