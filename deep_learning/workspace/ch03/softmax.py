import numpy as np

# オーバーフローするソフトマックス関数
def softmax_overflow(a):
    exp = np.exp(a)   # 指数関数
    sum = np.sum(exp) # 指数関数の和
    y = exp / sum # ここでオーバーフロー
    return y

def softmax(a):
    c = np.max(a)
    exp = np.exp(a - c) # オーバーフロー対策
    sum = np.sum(exp)   # 指数関数の和
    y = exp / sum       # ここでオーバーフローは発生しない
    return y


# print(softmax_overflow(np.array([1010, 1000, 990])))
# 上記実行でエラーとなる
# /app/workspace/ch03/softmax.py:5: RuntimeWarning: overflow encountered in exp
#   exp = np.exp(x) # 指数関数
# /app/workspace/ch03/softmax.py:7: RuntimeWarning: invalid value encountered in divide
#  y = exp / sum
# [nan nan nan]

result = softmax(np.array([1010, 1000, 990]))
print(result)
print(np.sum(result))

# root@aa12d91d4e5c:/app/workspace/ch03# python softmax.py 
# [9.99954600e-01 4.53978686e-05 2.06106005e-09]
# 1.0

"""
- オーバーフローが起きなくなる
softmax_overflow を softmax として「入力の最大値 c を引く」実装にしたことで、np.exp によるオーバーフローが回避されます。
例えば [1010,1000,990] のような大きな値では、ナイーブ実装は exp(1010) で overflow → nan になりましたが、c を引くことで安定に確率ベクトルが得られます
（添付の実行結果では [9.999546e-01, 4.5398e-05, 2.061e-09], 合計 1.0）。

- 出力は依然として「確率のベクトル」になる（正規化）
両関数とも exp を合計で割るため、返り値は正規化された確率（和が 1）になります。ただしナイーブ実装は数値的に壊れる可能性があります。

副次的な影響・注意点
- 変数名で組み込み関数を隠している
変数名 sum を使うと組み込みの sum() を隠します。実害は少ないですが、可読性と潜在バグ回避のため sum_exp や exp_sum を推奨します。

- 多次元配列への対応がない（axis 指定）
現在の実装は 1 次元向けです。ミニバッチ（2D）などで軸指定が必要なら、axis 引数と keepdims=True を使って実装するのを推奨します。

- 型/精度・極端値の扱い
非常に小さい値はゼロに丸められ（アンダーフロー）ても確率の比率は保たれるため通常問題になりません。NaN や Inf が入力に含まれる場合はそのまま伝播します（必要なら前処理で対処）。

- 深層学習での自動微分への影響
数値安定化（最大値を引く）は勾配計算に影響せず、安定した学習に寄与します。対数を取る用途では log-sum-exp を使うとさらに安全です（cross-entropy 実装等で推奨）。
"""