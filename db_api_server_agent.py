import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os

# LangChain 및 DB 관련 라이브러리
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- 1. 초기 설정 및 모델/DB 로딩 ---

print("AI 에이전트용 DB 서버 초기화 중...")
print("임베딩 모델을 로드합니다. (최초 실행 시 시간이 걸릴 수 있습니다)")
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={'device': 'cuda'}
)
print("임베딩 모델 로드 완료.")

available_dbs = {}
db_folders = {
    "core_patents": "faiss_index_core_patents_gpu",
}


for db_id, folder_name in db_folders.items():
    db_path = os.path.join('.', folder_name)
    if os.path.exists(db_path):
        print(f"'{db_id}' DB 로딩 중...")
        available_dbs[db_id] = FAISS.load_local(
            db_path, embeddings, allow_dangerous_deserialization=True
        )
        print(f"'{db_id}' DB 로드 완료.")

app = FastAPI()

class SearchRequest(BaseModel):
    db_id: str
    keywords: List[str]
    k_per_keyword: int = 5 # 각 키워드당 몇 개의 문서를 찾을지

# --- 2. 업그레이드된 API 엔드포인트 ---

@app.post("/search_by_keywords")
def search_by_keywords(request: SearchRequest):
    """
    키워드 리스트를 받아, 각 키워드에 대해 문서를 검색하고 중복을 제거한 뒤 결과를 반환합니다.
    """
    print(f"\n'{request.db_id}' DB에 대한 키워드 검색 요청 수신: {request.keywords}")
    if request.db_id not in available_dbs:
        raise HTTPException(status_code=404, detail=f"'{request.db_id}' DB를 찾을 수 없습니다.")
    
    vector_db = available_dbs[request.db_id]
    
    try:
        all_retrieved_docs = []
        unique_doc_sources = set()

        for keyword in request.keywords:
            retriever = vector_db.as_retriever(search_kwargs={'k': request.k_per_keyword})
            retrieved_docs = retriever.invoke(keyword)
            
            for doc in retrieved_docs:
                source = doc.metadata.get('source')
                if source not in unique_doc_sources:
                    all_retrieved_docs.append(doc)
                    unique_doc_sources.add(source)
        
        results = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in retrieved_docs]
        
        print(f"-> 총 {len(results)}개의 고유 문서를 찾았습니다.")
        return {"documents": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"키워드 검색 중 서버 오류 발생: {e}")

if __name__ == "__main__":
    print("DB 검색 API 서버를 시작하려면 Anaconda Prompt에서 아래 명령어를 입력하세요:")
    print("uvicorn db_api_server_agent:app --host 0.0.0.0 --port 8000")
