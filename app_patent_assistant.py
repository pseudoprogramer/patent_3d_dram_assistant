import streamlit as st
import os
import requests
import json
import re

# LangChain 및 Gemini 관련 라이브러리
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 1. 애플리케이션 기본 설정 및 프롬프트 ---
st.set_page_config(page_title="AI 특허 분석 에이전트", layout="wide")

# [프롬프트 1] 사용자의 질문에서 검색 키워드를 추출하기 위한 프롬프트
KEYWORD_EXTRACTION_PROMPT = PromptTemplate.from_template(
    """
    You are an expert in semiconductor and patent search. Your task is to extract the most relevant and effective search keywords from the user's question.
    The keywords should be concise technical terms. Output them as a comma-separated list.
    
    User's Question: {question}
    
    Keywords:
    """
)

# [프롬프트 2] 검색된 문서를 바탕으로 최종 답변을 생성하기 위한 프롬프트
FINAL_ANSWER_PROMPT = PromptTemplate.from_template(
    """
    You are a helpful AI assistant specializing in patent analysis.
    Based ONLY on the following retrieved patent documents, provide a comprehensive answer to the user's original question.
    If the documents don't provide enough information, clearly state that the answer cannot be found in the provided documents.
    Provide a clear and concise answer, and always cite the source patent documents you used by their filenames (e.g., `[us20230012345a1p.txt]`).

    **Retrieved Documents:**
    {context}

    **User's Original Question:**
    {question}

    **Your Comprehensive Answer:**
    """
)

# --- 2. 사이드바 - 설정 ---
with st.sidebar:
    st.header("✨ AI & DB 서버 설정")
    gemini_api_key = st.text_input("Gemini API Key", type="password")
    db_server_url = st.text_input("DB 검색 서버 주소", help="예: http://localhost:8000")
    
    st.markdown("---")
    st.header("🤖 모델 선택")
    # [수정] 사용자가 Gemini 2.5 모델을 선택할 수 있도록 옵션 추가
    selected_model = st.radio(
        "답변 생성 모델 선택:",
        ("gemini-2.5-pro", "gemini-2.5-flash"),
        captions=["최고 품질 (2.5 Pro)", "최신/균형 (2.5 Flash)"],
        horizontal=True,
        index=0 # 기본값으로 2.5 Pro 선택
    )

    st.markdown("---")
    st.header("📚 분석 대상 선택")
    db_options = {"3D DRAM 특허": "3d_dram"}
    selected_db_name = st.selectbox("분석할 DB를 선택하세요.", options=db_options.keys())
    selected_db_id = db_options[selected_db_name]
    
    if st.button("대화 기록 초기화"):
        st.session_state.messages = []
        st.rerun()

# --- 3. 메인 Q&A 로직 (AI 에이전트 루틴) ---
st.title(f"🤖 AI 특허 분석 에이전트 ({selected_db_name})")

if not gemini_api_key or not db_server_url:
    st.info("사이드바에 Gemini API Key와 DB 검색 서버 주소를 모두 입력해주세요.")
else:
    if "messages" not in st.session_state or st.session_state.get("current_model") != selected_model:
        st.session_state.messages = []
        st.session_state.current_model = selected_model

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_question := st.chat_input("분석하고 싶은 내용을 자유롭게 질문해보세요..."):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            try:
                # [수정] 선택된 모델로 LLM 객체 생성
                llm = ChatGoogleGenerativeAI(model=selected_model, google_api_key=gemini_api_key, temperature=0.0)

                # --- 루틴 2 & 3: Gemini에게 검색 키워드 추출 요청 ---
                with st.spinner("1/3: 질문을 분석하여 검색 키워드를 추출하는 중..."):
                    keyword_chain = KEYWORD_EXTRACTION_PROMPT | llm | StrOutputParser()
                    extracted_keywords = keyword_chain.invoke({"question": user_question})
                    keyword_list = [k.strip() for k in extracted_keywords.split(',') if k.strip()]
                    st.info(f"🔍 추출된 검색 키워드: `{', '.join(keyword_list)}`")

                # --- 루틴 4: 키워드로 DB 서버에 문서 검색 요청 ---
                with st.spinner(f"2/3: DB 서버에서 '{len(keyword_list)}'개 키워드로 관련 특허를 검색하는 중..."):
                    search_url = f"{db_server_url.rstrip('/')}/search_by_keywords"
                    search_payload = {"db_id": selected_db_id, "keywords": keyword_list}
                    response = requests.post(search_url, json=search_payload, timeout=60)
                    response.raise_for_status()
                    retrieved_data = response.json().get('documents', [])
                
                if not retrieved_data:
                    st.warning("관련된 특허 문서를 찾지 못했습니다. 다른 질문으로 시도해보세요.")
                else:
                    st.success(f"📄 총 {len(retrieved_data)}개의 관련 특허를 찾았습니다. 이제 각 문서를 종합하여 답변을 생성합니다.")

                    # --- 루틴 5: 검색된 문서로 Gemini에게 최종 답변 생성 요청 ---
                    with st.spinner("3/3: Gemini가 검색된 문헌을 종합하여 최종 답변을 작성하는 중..."):
                        
                        def format_docs(docs):
                            return "\n\n".join([f"--- Source: {os.path.basename(doc['metadata'].get('source', 'N/A'))} ---\n{doc['page_content']}" for doc in docs])
                        
                        final_rag_chain = (
                            {"context": lambda x: format_docs(retrieved_data), "question": RunnablePassthrough()}
                            | FINAL_ANSWER_PROMPT
                            | llm
                            | StrOutputParser()
                        )
                        
                        final_answer = final_rag_chain.invoke(user_question)
                        st.markdown(final_answer)
                        st.session_state.messages.append({"role": "assistant", "content": final_answer})

            except Exception as e:
                st.error(f"에이전트 실행 중 오류가 발생했습니다: {e}")
