import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    """画像をデフォルトのビューアで開く代わりに、
    リポジトリの data/ch03 ディレクトリへ PNG ファイルとして保存します。
    """
    pil_img = Image.fromarray(np.uint8(img))
    # スクリプト位置から見た deep_learning/data/ch03 へ出力
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'ch03'))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'mnist_image.png')
    pil_img.save(out_path)
    print(f"saved image -> {out_path}")

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 形状を元の画像サイズに変形
print(img.shape)  # (28, 28)

img_show(img)