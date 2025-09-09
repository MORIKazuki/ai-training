import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    """MNIST データセットを読み込み、テストデータを返す。

    戻り値:
        x_test (numpy.ndarray): 画像データ、正規化済み、flatten=True のため形状は (N, 784)
        t_test (numpy.ndarray): ラベルの一次元配列、要素は 0-9 の整数

    補足:
        - normalize=True により画素値は 0.0-1.0 の範囲に正規化されます。
        - one_hot_label=False のためラベルはワンホットではなく整数配列です。
    """
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    """事前学習済みのネットワークパラメータをファイルから読み込んで返す。

    ファイル名はカレントワーキングディレクトリの "sample_weight.pkl" を想定。
    戻されるオブジェクト `network` は辞書で、キーは 'W1','W2','W3','b1','b2','b3' を含むことを期待します。

    例外:
        ファイルが存在しない場合は FileNotFoundError が発生します。
    """
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    """単一入力 `x` に対して順伝播を行い、出力確率ベクトルを返す。

    Args:
        network (dict): 'W1','W2','W3','b1','b2','b3' を含むパラメータ辞書
        x (numpy.ndarray): 1次元の入力ベクトル（flatten=True の MNIST なら形状 (784,)）

    Returns:
        y (numpy.ndarray): softmax で正規化された出力確率ベクトル（形状例: (10,)）

    実装の流れ:
        1) 入力と重みの内積を取り、活性化関数 sigmoid を中間層で使用
        2) 最終層のスコアに対して softmax を適用して確率に変換

    注意:
        - `x` が 1 次元であることを想定しています。バッチ処理を行う場合は行列演算に対応するよう修正が必要です。
    """
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()
print(network)
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y) # 最も確率の高い要素のインデックスを取得
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))