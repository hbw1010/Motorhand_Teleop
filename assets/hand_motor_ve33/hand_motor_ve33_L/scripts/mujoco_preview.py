"""本侧 vE.3.3 指节输入 MuJoCo 预览入口。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[2]))

from hand_controller.ve33_preview import main

if __name__ == "__main__":
    main(model_dir=ROOT)
