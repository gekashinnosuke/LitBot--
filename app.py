# ============================================
# 論文チャットボット（FAISS 永続化高速版）
# ============================================

import streamlit as st
import openai
import numpy as np
import pickle
from PyPDF2 import PdfReader
import faiss
import os

# ===============================
# 設定
# ===============================
openai.api_key = os.environ.get("OPENAI_API_KEY")  # ←ここにAPIキー
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-5.1"

# ===============================
# ユーティリティ関数
# ===============================
def split_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def get_embedding(text):
    res = openai.embeddings.create(model=EMBED_MODEL, input=text)
    return np.array(res.data[0].embedding, dtype=np.float32)

# ===============================
# PDFリスト
# ===============================
PDF_FILES = [
    "1-s2.0-S0016236125029199-main.pdf",
    "20251118　計測診断　話題提供　秋濱.pdf",
    "230011初校校正原稿_委員長修正.pdf",
    "48_20174835.pdf",
    "49_20184655.pdf",
    "50_20194330.pdf",
    "50_20194671.pdf",
    "50_20194902.pdf",
    "51_20204534.pdf",
    "52_20214125.pdf",
    "53_20224165.pdf",
    "54_20234272.pdf",
    "55_20244339.pdf",
    "56_20254681.pdf",
    "Ansys_Chemkin_Theory_Manual.pdf",
    "ICES2018P-OitaU_完全版.pdf",
    "ICFD2025P.pdf",
    "JSAE2025SM.pdf",
    "PROCI-D-15-00194P.pdf",
    "Soot_oxidation_and_the_mechanisms_of_oxidation_induced_fragmentation_in_a_two_stage_burner_for_ethylene_and_surrogate_fuels.pdf",
    "すす計算_連載+秋濱・橋本　提出版　20170528.pdf"
]

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="LitBot2-FAISS高速版", layout="wide")
st.title("LitBot2-反応性ガス力学研究室")
st.markdown("登録済み論文から質問できます")

# ===============================
# Embedding & FAISSロード/作成
# ===============================
if "index" not in st.session_state:

    if os.path.exists("faiss.index") and os.path.exists("chunks.pkl"):
        st.info("保存済みFAISSインデックスを読み込み中...")
        # FAISSインデックス読み込み
        st.session_state["index"] = faiss.read_index("faiss.index")
        # チャンク情報読み込み
        with open("chunks.pkl", "rb") as f:
            st.session_state["chunks"] = pickle.load(f)
        st.success("FAISSインデックス読み込み完了（高速動作可能）")
    else:
        st.info("初回Embedding作成中（時間がかかります）...")
        all_chunks = []
        all_embeddings = []

        for fname in PDF_FILES:
            reader = PdfReader(fname)
            text = ""
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"

            chunks = split_text(text)
            for chunk in chunks:
                all_chunks.append({"filename": fname, "text": chunk})
                all_embeddings.append(get_embedding(chunk))

        # FAISSインデックス作成
        dim = len(all_embeddings[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(all_embeddings, dtype=np.float32))

        # 保存
        faiss.write_index(index, "faiss.index")
        with open("chunks.pkl", "wb") as f:
            pickle.dump(all_chunks, f)

        st.session_state["index"] = index
        st.session_state["chunks"] = all_chunks
        st.success("初回Embedding作成完了（次回以降高速）")

# ===============================
# チャット処理
# ===============================
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 過去メッセージ表示
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# ユーザー質問
question = st.chat_input("質問を入力してください")

if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    with st.spinner("回答生成中..."):
        q_emb = get_embedding(question).reshape(1, -1)
        # FAISS検索
        D, I = st.session_state["index"].search(q_emb, 5)
        context = ""
        for idx in I[0]:
            chunk = st.session_state["chunks"][idx]
            context += f"[論文: {chunk['filename']}]\n{chunk['text']}\n\n"

        prompt = f"""
以下は複数論文から抽出した関連部分です。
これを参考に質問に答えてください。

{context}

質問:
{question}
"""
        response = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        answer = response.choices[0].message.content.strip()

    st.session_state["messages"].append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)
