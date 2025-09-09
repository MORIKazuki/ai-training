# ai-training

## 目的
[ゼロから作るDeep Learning ―Pythonで学ぶディープラーニングの理論と実装](https://www.oreilly.co.jp/books/9784873117584/)の学習

## 開発環境
- OS:Windows11
- DockerDesktop
- VisualStudioCode
    - [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
    - [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

## 開発環境構築手順
1. dockerディレクトリ配下に必要であれば/certificates/certificate.crtを配置する
2. 以下のコマンドを実行する
```
> docker-compose up -d --build
```

3. 以下のコマンドでコンテナ内にログインする
```
> docker exec -it deep_learning_training bash
```

4. サンプルコードを実行する
```
# python workspace/sample_matplotlib.py
```

5. `deep_learning/data`配下に`sin.png`が生成されることを確認する(VisualStudioCode拡張機能をあらかじめ入れておくと便利)

## 参考資料
[ゼロから作るDeep Learning ―Pythonで学ぶディープラーニングの理論と実装](https://www.oreilly.co.jp/books/9784873117584/)