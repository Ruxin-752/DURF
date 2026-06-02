"""快速测试 trainer 能否启动并运行 100 steps。"""
import sys
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)  # 确保相对路径（config/）正确

from src.train.trainer import train

if __name__ == "__main__":
    train([
        '--mode', 'no-feedback',
        '--total-timesteps', '100',
        '--participant', 'P00',
        '--seed', '42',
        '--config', 'config/default.yaml',
    ])
