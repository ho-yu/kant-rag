from __future__ import annotations

import argparse
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from langchain_core.retrievers import BaseRetriever
from qdrant_client import models

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config" / "retrieval_a" / "naive.yaml"
MODELS_PATH = REPO_ROOT / "config" / "retrieval_a" / "models.yaml"
LOCAL_QDRANT_PATH = REPO_ROOT / "artifacts" / "retrieval_a" / "qdrant_local"

"""
1. 함수 선언

- load_config(): 기본값이 있는 매개변수이다. 따로 넘기지 않으면 config/retrieval_a/naive.yaml 을 쓴다. 
- -> dict[str, Any]: "키는 문자열, 값은 아무거나"라는 타입 힌트이다. 

2. 파일 열기 

with open(path, encoding="utf-8") as f:

- encoding="utg-8" : 파일을 어떤 문자 인코딩으로 읽을지 정한다. 
- with ... as f: with 문은 블록이 끝나면 파일을 자동으로 닫아준다. 이런 걸 컨텍스트 메니저라고 한다. with 없이 쓰면 직접 닫아야 한다.

f = open(path, encoding="utf-8")
data = yaml.safe_load(f)
f.close()          # 깜빡하거나, 위에서 에러가 나면 실행이 안 됨

3. yaml 해석해서 돌려주기 

return yaml.safe_load(f)
- 파일을 내용을 읽어서 yaml 문법대로 해석해 파이썬 객체로 바꾼다. yaml은 맨 위의 import yaml (PyYAML 패키지)이다.

왜 load가 아니라 safe_load 인가?
- yaml.load는 yaml 안에 적힌 파이썬 객체를 만들 수 있어서, 악의적인 yaml을 읽으면 임의의 코드가 실행될 위험이 있습니다.
- safe_load는 문자열, 숫자, 리스트, 딕셔너리 같은 기본 타입만 만들어서 안전합니다. 설정 파일을 읽을 때는 거의 항상 safe_load를 씁니다.

전체 흐름:
경로 받기 → 파일 열기(utf-8) → yaml 해석 → 딕셔너리 반환
                                └→ with 블록 종료 → 파일 자동으로 닫힘

                                       
"""
def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


DIFFICULTIES = ("basic", "intermediate", "advanced")
DOMAINS = ("shipping", "finance", "manufacturing")
SPARSE_BACKENDS = ("bge-m3", "bm25")


"""

1. model_slug(), 이 함수가 푸는 문제 

- Hugging Face 모델 ID는 조직/모델명 형식이다 (BAAI /bge-m3). 이 값을 Qdrant 컬랙션 이름에 넣으려면 두가지가 걸린다.
    1. /는 이름이나 경로에 넣기 곤란한 문자이다. 
    2. 대소문자가 섞여 있으면 BGE-M3와 bge-m3가 다른 값이 된다. 
  그래서 "조직명은 버리고, 소문자롤 정규화한다"가 이 함수가 하는 일이다. 이렇게 줄인 식별자를 흔히 slug 라고 부른다. 

2. 메서드 체이닝(method chaining): 왜 한 줄로 쓸 수 있나

- split()은 리스트를, [-1]은 문자열을, lower()는 문자열을 돌려준다. 각 단계가 돌려줄 객체에 바로 다음 연산을 붙일 수 있어서 한줄로 이어진다. 
- 풀어 쓰면 이렇게 된다:
    - parts = model.split("/") # 문자열을 "/" 가 나올 때마다 잘라서 리스트로 돌려주는 문자열 메서드이다. "BAAI/bge-m3".split("/") # ["BAAI", "bge-m3"]
    - name = parts[-1] # -1 인 이유는 조직명이 아닌 매번 모델명을 얻기 위해서다. 
    - return name.lower()
    => chaining 하면 model.split("/")[-1].lower() # split을 하면 리스트 값이 출력되어 [-1]가 chaining 가능하다. 
- 읽을 때는 각 단계 뒤에 타입이 무엇인지 따라가는 습관을 들이면 좋다. 체이닝이 길어지면 타입을 놓쳐서 실수하기 쉽기 때문이다. 


Ex. 
model = "BAAI/bge-m3"
identify = model_slug(model)

def model_slug("BAAI/bge-m3") -> str:
    return model.split("/")[-1].lower()
    # model.split = ["BAAI", "bge-m3"]
    # [-1] = "bge-m3"
    # .lower() = "bge-m3"

=> identify = "bge-m3"
"""
def model_slug(model: str) -> str:
    return model.split("/")[-1].lower()

"""

이 함수가 하는 일: 어떤 주제(domain)와 난이도(difficulty)로 검색할지 최종 결정하는 함수이다. 

1. 함수 선언
def resolve_scope(cfg: dict[str, Any], domain: str | None, difficulty: str | None) -> tuple[str, str]:

- cfg: laod_config()로 읽은 naive.yaml 딕셔너리이다. 
- str | None: 문자열 또는 None이라는 뜻이다. 
- -> tuple[str, str]: 문자열 두 개짜리 튜플을 돌려준다는 뜻이다. -> ("shipping", "basic")

2. 설정에서 collection 부분만 꺼내기 
c = cfg["collection"] # {"prefix": "kant", "domain": "shipping", "difficulty": "basic", "vector_name": ""}

- naive.yaml에서 collection 값만 받아서 변수 c에 저장한다. 

3. 값 정하고 정리하기 
domain = (domain or c["domain"]).strip().lower()
difficulty = (difficulty or c["difficulty"]).strip().lower()

- domain or c["domain"]: 넘겨받은 값이 있으면 그 값, None이면 설정 기본값
- 괄호 ( ): 괄호가 없으면 .strip()이 c["domain"]에만 붙습니다. or로 고른 결과 전체에 적용하려고 괄호로 묶은 겁니다.
- .strip(): 앞뒤 공백을 없앤다.
- .lower(): 소문자로 바꾼다. 

4. 허용된 값인지 확인 
if domain not in DOMAINS:
    raise ValueError(f"domain은 {DOMAINS} 중 하나여야 합니다: {domain!r}")
if difficulty not in DIFFICULTIES:
    raise ValueError(f"difficulty는 {DIFFICULTIES} 중 하나여야 합니다: {difficulty!r}")

- 파일 위쪽에 정의된 허용 목록과 비교한다. 
DIFFICULTIES = ("basic", " intermediate", "advanced")
DOMAINS = ("shipping", "finance", "manufacturing")
SPARSE_BACKENDS = ("bge-m3", "bm25")

- not in 튜플: 튜플 안에 그 값이 없으면 참이다.
- 오타를 넣으면 에러가 발생한다. 

5. 두 값 돌려주기 
return domain, difficulty

"""

def resolve_scope(cfg: dict[str, Any], domain: str | None, difficulty: str | None) -> tuple[str, str]:
    c = cfg["collection"]
    domain = (domain or c["domain"]).strip().lower()
    difficulty = (difficulty or c["difficulty"]).strip().lower()
    if domain not in DOMAINS:
        raise ValueError(f"domain은 {DOMAINS} 중 하나여야 합니다: {domain!r}")
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty는 {DIFFICULTIES} 중 하나여야 합니다: {difficulty!r}")
    return domain, difficulty


"""

model_profile(): 이 함수가 하는 일: 설정 파일 models.yaml에서 이름 하나에 해당하는 모델 설정을 꺼내고, 빠진 값은 기본값으로 채운 뒤, 잘못된 조합이면 에러를 내는 함수이다. 

1. 함수 선언: def model_profile(difficulty: str, path: Path = MODELS_PATH) -> dict[str, str]:

- path: Path = MODELS_PATH: 기본값이 있는 매개변수입니다. 따로 넘기지 않으면 config/retrieval_a/models.yaml을 씁니다. 테스트할 때는 다른 파일을 넘길 수 있습니다.
- -> dict[str, str]: 키도 값도 문자열인 딕셔너리를 돌려준다는 뜻이다. 

2. yaml 읽기
profiles = load_config(models.yaml): load_config가 models.yaml을 파이썬 딕셔너리로 바꿔준다. 

3. 이름이 있는지 먼저 확인
if difficulty not in profiles:
        raise ValueError(f"{path.name}에 {difficulty!r} 설정이 없습니다. 있는 이름: {list(profiles)}")

- in으로 딕셔너리를 검사하면 키가 있는지를 본다. difficulty키가 profiles 딕셔너리안에 없으면 바로 에러 발생.
- raise ValueError(...): 에러를 일부러 발생시켜 함수를 바로 멈춘다. 

4. 설정 꺼내서 정리하기 

p = profiles[difficulty] # if difficulty in profiles / true 하면 difficulty 값을 p 변수에 저장
profile = {
    "embedding": p["embedding"],
    "query_prefix": p.get("query_prefix") or "",
    "sparse": p.get("sparse", "bm25"),
    "reranker": p["reranker"],
}

Ex. p = profiles["basic"]

p = {
        "embedding": "BAAI/bge-m3",
        "query_prefix": "",
        "sparse": "bge-m3",
        "reranker": "BAAI/bge-reranker-v2-m3"
    }

profile = {
    "embedding": p["embedding"], # "BAAI/bge-m3"
    "query_prefix": p.get("query_prefix") or "", # "" / 
    "sparse": p.get("sparse", "bm25"), # "bge-m3"
    "reranker": p["reranker"], # "BAAI/bge-reranker-v2-m3"
}

=> .get()을 쓰는 이유: 키가 없을 때 에러를 내지 않고 다른 값을 돌려받으려고 쓴다.

p = {"embedding": "BAAI/bge-m3"}

p["sparse"]              # KeyError: 'sparse'    → 프로그램이 멈춤
p.get("sparse")          # None                  → 멈추지 않고 None을 줌
p.get("sparse", "bm25")  # "bm25"                → 멈추지 않고 내가 정한 기본값을 줌


5. 검증 1: sparse 값이 허용된 값인가

if profile["sparse"] not in SPARSE_BACKENDS: # sparse 값이 SPARSE_BACKENDS에 없으면 바로 에러 처리한다. 
    raise ValueError(...)

6. 검증 2: 조합이 맞는가 

if profile["sparse"] == "bge-m3" and model_slug(profile["embedding"]) != "bge-m3":
    raise ValueError(...)

    - 값 하나하나는 맞아도 조합이 말이 안 되는 경우를 거른다. 
    - bge-m3 sparse 벡터는 bge-m3 모델이 dense 벡터와 함께 만들어 내는 것이라, 임베딩 모델이 qwen3인데 sparse: bge-m3를 쓸 수는 없다.
    - 여기서 앞에서 배운 model_slug가 쓰입니다. "BAAI/bge-m3" → "bge-m3"로 바꿔서 비교합니다
    - and는 단락 평가를 합니다. 앞 조건이 거짓이면(sparse가 bm25이면) 뒤의 model_slug는 아예 실행하지 않습니다.

7. 반환

return profile => 여기까지 왔다면 검증을 모두 통과한 딕셔너리이다. 
"""

def model_profile(difficulty: str, path: Path = MODELS_PATH) -> dict[str, str]:
    profiles = load_config(path)
    if difficulty not in profiles:
        raise ValueError(f"{path.name}에 {difficulty!r} 설정이 없습니다. 있는 이름: {list(profiles)}")
    p = profiles[difficulty]
    profile = {
        "embedding": p["embedding"],
        "query_prefix": p.get("query_prefix") or "",
        "sparse": p.get("sparse", "bm25"),
        "reranker": p["reranker"],
    }
    if profile["sparse"] not in SPARSE_BACKENDS:
        raise ValueError(f"sparse는 {SPARSE_BACKENDS} 중 하나여야 합니다: {profile['sparse']!r}")
    if profile["sparse"] == "bge-m3" and model_slug(profile["embedding"]) != "bge-m3":
        raise ValueError(
            f"sparse: bge-m3는 임베딩이 BAAI/bge-m3일 때만 쓸 수 있습니다 "
            f"({difficulty}: {profile['embedding']}). sparse: bm25로 바꿔 주세요."
        )
    return profile


""" 
1. 함수 정의

def collection_name(
    cfg: dict[str, Any], *, domain: str | None = None, difficulty: str | None = None, profile: str | None = None
) -> str:

- cfg: dict[str, Any]: load_config()가 돌려준 설정 dict이다. :dict[str, Any]는 "키는 문자열, 값은 아무거나"라는 타입 힌트이고, 실행에는 영향이 없다.
- *: 이 표시 뒤의 인자는 반드시 이름을 붙여서 넘겨야한다.
    1. 됨: collection_name(cfg, domain = "shipping", difficulty = "basic")
    2. 에러: collection_name(cfg, "shipping", "basic")
    => 인자 순서를 헷갈려서 domain과 difficulty가 뒤바뀌는 실수를 막아준다. 
- str | None = None: 문자열이나 None을 받을 수 있고, 안 넘기면 None이다. 
- -> str: 돌려주는 값이 문자열이라는 표시이다. 

2. 도메인, 난이도 확정 

domain, difficulty = resolve_scope(cfg, domain, difficulty)

- resolve_scope가 하는 일은 세가지이다. 
    1. 기본값 채우기: None이면 naive.yaml의 값을 쓴다. (shipping, basic)
    2. 정리: 앞뒤 공백을 지우고 소문자로 바꾼다. ("Basic" -> "basic")
    3. 검사: 허용된 값이 아니면 에러를 낸다. 
        - domain: shipping, finance, manunfacturing
        - difficulty: basic, intermediate, advanced 
- 함수가 값 두 개를 튜플로 돌려주고, 그걸 domain, difficulty에 나눠 담는다. 

3. 모델 이름표 만들기 

tag = model_slug(model_profile(profile or difficulty)["embedding"])

1. profile or difficulty 

"""
def collection_name(
    cfg: dict[str, Any], *, domain: str | None = None, difficulty: str | None = None, profile: str | None = None
) -> str:
    domain, difficulty = resolve_scope(cfg, domain, difficulty)
    tag = model_slug(model_profile(profile or difficulty)["embedding"])
    return f"{cfg['collection']['prefix']}_{domain}_{difficulty}_{tag}"

"""
1. levels 변수에는 문자열 리스트가 들어온다. 
2. -> models.Filter 는 이 함수가 models.Filter 객체를 돌려준다는 표시다. (타입 힌트)

models.MatchAny(any=list(levels)): 이 값들 중 하나라도 일치하면 통과라는 조건이다. levels가 리스트 ["para", "table"]
"""
def level_filter(levels: list[str]) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key="metadata.level", match=models.MatchAny(any=list(levels)))]
    )


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=4)
def _embeddings(model_name: str, device: str, normalize: bool, query_prefix: str = ""):
    from langchain_huggingface import HuggingFaceEmbeddings

    query_kwargs = {"normalize_embeddings": normalize}
    if query_prefix:
        query_kwargs["prompt"] = query_prefix
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": _resolve_device(device)},
        encode_kwargs={"normalize_embeddings": normalize},
        query_encode_kwargs=query_kwargs,
    )


@lru_cache(maxsize=1)
def _remote_client(url: str, api_key: str | None):
    from qdrant_client import QdrantClient

    return QdrantClient(url=url, api_key=api_key, timeout=60)


@lru_cache(maxsize=1)
def local_client(path: str = str(LOCAL_QDRANT_PATH)):
    from qdrant_client import QdrantClient

    Path(path).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=path)


def _connect_client(name: str):
    url = (os.getenv("QDRANT_URL") or "").strip()
    if url:
        client = _remote_client(url, os.getenv("QDRANT_API_KEY") or None)
        if client.collection_exists(name):
            return client
    if LOCAL_QDRANT_PATH.exists():
        client = local_client()
        if client.collection_exists(name):
            return client
    where = "Qdrant 서버와 로컬 색인" if url else "로컬 색인 (QDRANT_URL이 비어 있음)"
    raise RuntimeError(
        f"{where}에 컬렉션 {name!r}이 없습니다. 아직 적재되지 않은 주제·난이도이거나 "
        "models.yaml의 모델 설정과 적재된 모델이 다를 수 있습니다."
    )


def _connect_store(cfg: dict[str, Any], name: str, profile: dict[str, str]):
    from langchain_qdrant import QdrantVectorStore

    emb_cfg = cfg["embedding"]
    embedding = _embeddings(profile["embedding"], emb_cfg["device"], emb_cfg["normalize"], profile["query_prefix"])
    client = _connect_client(name)
    return QdrantVectorStore(
        client=client,
        collection_name=name,
        embedding=embedding,
        vector_name=cfg["collection"].get("vector_name", ""),
    )


def build_naive_retriever(
    *,
    domain: str | None = None,
    difficulty: str | None = None,
    top_k: int | None = None,
    levels: list[str] | None = None,
    profile: str | None = None,
    config_path: Path = CONFIG_PATH,
) -> BaseRetriever:
    cfg = load_config(config_path)
    search = cfg["search"]
    domain, difficulty = resolve_scope(cfg, domain, difficulty)
    name = collection_name(cfg, domain=domain, difficulty=difficulty, profile=profile)
    store = _connect_store(cfg, name, model_profile(profile or difficulty))
    return store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": top_k or search["top_k"],
            "filter": level_filter(levels or search["levels"]),
        },
    )


def _main() -> None:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description="retrieval_a Naive 검색 결과 확인")
    parser.add_argument("question")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--difficulty", default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--profile", default=None, help="models.yaml의 모델 설정 이름 (기본: 난이도 이름)")
    args = parser.parse_args()

    retriever = build_naive_retriever(
        domain=args.domain, difficulty=args.difficulty, top_k=args.k, profile=args.profile
    )
    for i, doc in enumerate(retriever.invoke(args.question), start=1):
        m = doc.metadata
        print(f"[S{i}] {m.get('book_id')}.pdf {m.get('page')}쪽 ({m.get('level')}, {m.get('chunk_id')})")
        print("    " + doc.page_content[:200].replace("\n", " "))


if __name__ == "__main__":
    _main()
