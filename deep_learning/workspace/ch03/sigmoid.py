import numpy as np
import matplotlib.pylab as plt


def sigmoid(x):
    """シグモイド関数を計算して返す。

    Args:
        x: スカラーまたは numpy 配列。任意の形状を受け取り、要素ごとに計算されます。

    Returns:
        numpy.ndarray or float: 入力と同じ形状の出力。各要素は 0 より大きく 1 未満の値になります。

    Notes:
        - 定義: sigmoid(x) = 1 / (1 + exp(-x))
        - 出力は (0, 1) の範囲に収まり、確率的解釈に用いられることがあります。
        - 微分: d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x)) 。ニューラルネットワークの逆伝播でよく使われます。
        - 数値的注意: 非常に大きな負の値では exp(-x) が非常に大きくなり得ますが、numpy の実装で通常扱えます。必要なら入力をクリップするなどの対処を検討してください。
    """
    return 1 / (1 + np.exp(-x))

X = np.arange(-5.0, 5.0, 0.1)
Y = sigmoid(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1)
plt.show()
plt.title("sigmoid")
plt.savefig("/app/data/ch03/sigmoid.png") 