
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------
# 2026/04/01
# RAGチャット
# --------------------------------------------------

import sys
import argparse
import shutil
import readline
from datetime import datetime

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings

OLLAMA_URL='http://192.168.1.64:11434'
# 注意: モデル変更時はベクトルDBも再構築必要です。
# --- gemma3 ---
LLM_MODEL='gemma3:12b'
EMBED_MODEL='embeddinggemma:latest'
# --- llama3.1 ---
#LLM_MODEL='llama3.1:8b'
#EMBED_MODEL='nomic-embed-text:latest'
# --- gpt-oss ---
#LLM_MODEL='gpt-oss:20b'
#EMBED_MODEL='qwen3-embedding:8b'
# --- quen3:32b ---
#LLM_MODEL='qwen3:32b'
#EMBED_MODEL='qwen3-embedding:8b'
COLLECTION_NAME='langichain'
PERSIST_DIR='./chroma_db'
KNOWLEDGE_DIR='./local_docs'

# 利用モデルとDB定義
model = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_URL)
emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
db = Chroma(collection_name=COLLECTION_NAME, embedding_function=emb, persist_directory=PERSIST_DIR)

# ナレッジドキュメントの読み込みとインデックス作成
def create_vdb():
    # PDFファイルの場合
    #ldr = DirectoryLoader("./local_docs", glob="*.pdf", loader_cls=PyPDFLoader)

    # テキストファイルの場合
    ldr = DirectoryLoader(KNOWLEDGE_DIR, glob="*.txt")

    # ドキュメント読み込み
    raw_docs = ldr.load()

    # ドキュメントのチャンク分割
    txt_sp = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100, separators=['\n'])
    docs = txt_sp.split_documents(raw_docs)

    # ベクトルDBへのドキュメント登録
    db.add_documents(documents=docs)

    print("ドキュメント登録完了")

# DBの削除
def delete_vdb():
    db.delete_collection()
    shutil.rmtree("./chroma_db")
    print("削除完了")

# LLMに現在日時を伝える
def get_current_time():
    return datetime.now().strftime("%Y-%m-%d (%a) %H:%M:%S")

# LLMに関する質問処理
def query_llm(user_pmt: str):
    #print(f"あなた> {user_pmt}")

    # DBから質問に関連するドキュメントを得るIF(リトリーバ）を作成
    retriever = db.as_retriever(search_kwargs={"k": 1})
    context_docs = retriever.invoke(user_pmt)

    #print('-->', context_docs)

    if len(context_docs) == 0 :
        print("関連する情報が見つかりませんでした。")
        return

    # --- プロンプトテンプレート メモ ---
    # @参照資料：
    # ・第15回国会　衆議院　予算委員会　第31号　昭和28年2月28日
    # ・第219回国会　衆議院　予算委員会　第2号　令和7年11月7日
    #
    # @基本動作：
    # ・@参照資料に関係する問いには、@文脈を考慮して回答してください。
    # ・@参照資料に関係しない問いには、@文脈を考慮せず自身の知識にて回答してください。
    # ・@参照資料に関係するが、@文脈内に適した内容が無い場合は、自身の知識にて回答してください。
    # ・@参照資料に関係するが、@文脈内に適した内容が無く、自身の知識にも無い場合は、「情報が無いため、わかりません」と回答してください。
    # ・@参照資料の日付は国会開催の日付です。
    # ・一般的な質問、雑談などは、なるべく自身の知識の中から回答してください。
    # ・回答文中には、「文脈」という言葉を絶対に入れないでください。
    # ・関西弁で回答してください。
    # 
    # @現在日時: """
    # {current_time}
    # """
    #
    # @文脈: """
    # {context}
    # """
    # 
    # @質問: """
    # {user_pmt}
    # """
    #

    # プロンプトテンプレート
    pmt_all = ChatPromptTemplate.from_template('''\
    @参照資料：
    ・第15回国会　衆議院　予算委員会　第31号　昭和28年2月28日
    ・第219回国会　衆議院　予算委員会　第2号　令和7年11月7日

    @基本動作：
    ・@参照資料に関係する問いには、@文脈を考慮して回答してください。
    ・@参照資料に関係しない問いには、@文脈を考慮せず自身の知識にて回答してください。
    ・@参照資料に関係するが、@文脈内に適した内容が無い場合は、自身の知識にて回答してください。
    ・@参照資料に関係するが、@文脈内に適した内容が無く、自身の知識にも無い場合は、「情報が無いため、わかりません」と回答してください。
    ・@参照資料の日付は国会開催の日付です。
    ・一般的な質問、雑談などは、なるべく自身の知識の中から回答してください。
    ・回答文中には、「文脈」という言葉を絶対に入れないでください。

    
    @現在日時: """
    {current_time}
    """

    @文脈: """
    {context}
    """
    
    @質問: """
    {user_pmt}
    """
    ''').partial(current_time=get_current_time)

    # LangChainのチェイン定義
    chain = (
        {"context": retriever, "user_pmt": RunnablePassthrough()}
        | pmt_all 
        | model 
        | StrOutputParser()
    )

    # 入力された質問を基にchainを実行
    ai_result = chain.invoke(user_pmt)
    return ai_result

def chat():
    print("質問を入力してください。終了するには'exit'を入力。")

    while True:
        try:
            query = input("\n質問> ").strip()

            if query.lower() in ("exit", "quit", 'q'):
                print("終了します")
                break

            if not query:
                continue

            answer = query_llm(query)
            print(f"回答> {answer}")

        except ROFError:
            break
        except Exception as e:
            print(f"\nエラー: {e}")

def arg_parser():
    parser = argparse.ArgumentParser(description="LangChainを用いたRAGシステム")
    parser.add_argument("-a", "--add", action="store_true", help="ドキュメントをDBへ登録")
    parser.add_argument("-d", "--delete", action="store_true", help="DB削除")
    parser.add_argument("-c", "--chat", action="store_true", help="チャット")
    parser.add_argument("query", nargs="?", help="直接質問")

    return parser

# コマンドライン引数の解析
def main():
    parser = arg_parser()
    args = parser.parse_args()

    if args.add:
        create_vdb()
    elif args.delete:
        delete_vdb()
    elif args.chat:
        chat()
    elif len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    else:
        print(f"\n質問> {args.query}")
        answer = query_llm(args.query)
        print(f"回答> {answer}")

if __name__ == "__main__":
    main()

