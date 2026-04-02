[2026/04/02 H.Asakura]

# LangChainを用いたRAGシステム

# このプロジェクト環境について
このプロジェクトでは、近年利用が広がっているパッケージツールである uv を用いています。
uv を用いることでライブラリバージョンを厳密に管理しつつ仮想環境を構築できます。
そのため uv コマンドを用いて実行する手順を記載しておきます。

# 概要
このRAGシステムは、次の図のようにまず最初にナレッジドキュメントとなるテキストファイル(.txt)を
./local_docsディレクトリ内へ格納します。その後このドキュメントをベクトルデータに変換後、ベクトル
データベースへ格納します。このベクトル化処理は後述の main.py のオプション指定で行います。
ベクトルデータが作成後は、RAGを使ったチャットを行うことができます。

### 操作手順
　　　　　　　　　1. ナレッジドキュメント格納(./local_docs)
　　　　　　　　　　　　　　　↓
　　　　　　　　　2. ドキュメントベクトル化(./chroma_db)
3. 質問(入力)　　　　　　　　 ↓ 
　　　↓　　　　　　　　　　　 ↓ 
4. RagChat(main.py)　⇔　VectorDB(ChromaDB)
　　　↓
5. RagChat(main.py)　⇔　Ollama(LLM Model)
　　　↓
6. 回答(出力)

# 必要環境
このRagChatプロジェクトの他に推論環境(Ollama)が必要です。
こちらは、GPUがある環境にOllamaをサービスとして起動し、LLMモデルと、LLMモデルに
対応した埋め込みモデルを事前にインストールしておく必要があります。
そうすることで、Ollamaサーバが待ち受ける http://<hostname>:11434 に対してLLM処理
をオフロード実行することができます。

RagChat　⇔　Ollama(http://<hostname>:11434)

# 実行方法
実行するには、次のコマンドで行います。

## RAGシステムの実行(空打ち)
```bash
#> uv run main.py
``` 
このコマンドのヘルプが表示されます。

```text
usage: main.py [-h] [-a] [-d] [-c] [query]

LangChainを用いたRAGシステム

positional arguments:
  query         直接質問

options:
  -h, --help    show this help message and exit
  -a, --add     ドキュメントをDBへ登録
  -d, --delete  DB削除
  -c, --chat    チャット
```

## ベクトルデータベース作成
```bash
#> uv run main.py -a
``` 

## ベクトルデータベース削除
```bash
#> uv run main.py -d
``` 

## 単一質問投入
```bash
#> uv run main.py <質問文字列>
``` 

## チャット形式の連続質問投入
```bash
#> uv run main.py -c
``` 
この -c オプションを実行することで、チャットモードに入り、
q を入力することでチャットから抜けます。

