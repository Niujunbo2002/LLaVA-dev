import sys
sys.path.append("/mnt/petrelfs/niujunbo/niujunbo_dev/NativeResLLaVA")
import warnings
warnings.filterwarnings('ignore')
from llava.train.train import train

if __name__ == "__main__":
    train()