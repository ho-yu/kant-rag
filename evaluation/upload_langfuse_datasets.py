import json
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langfuse import get_client
from langfuse.api import NotFoundError


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

EXPECTED_FIELDS = {
    "reference_answer", "required_facts", "forbidden_claims",
    "expected_action", "evidence_groups", "evidence_required",
    "grading_notes", "bad_claims",
}

langfuse = get_client()

try:
    if not langfuse.auth_check():
        raise RuntimeError("Langfuse 연결 설정을 확인해주세요.")

    for path in sorted((ROOT / "data").glob("*/*/evaluation/*.json")):
        # 질문 전용 파일은 정답 포함 파일과 중복되므로 제외
        if path.stem.endswith("_questions"):
            continue

        difficulty = path.parents[2].name
        domain = path.parents[1].name.removesuffix("(not_use)")
        split = path.stem.removesuffix("_tasks").replace("_", "-")
        dataset_name = f"{domain}-{difficulty}-{split}-v1"

        rows = json.loads(path.read_text(encoding="utf-8-sig"))

        # 기존 데이터셋은 사용하고, 없으면 생성
        try:
            langfuse.get_dataset(name=dataset_name)
        except NotFoundError:
            langfuse.create_dataset(name=dataset_name)

        for row in rows:
            langfuse.create_dataset_item(
                dataset_name=dataset_name,
                id=str(uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"kant-rag/{dataset_name}/{row['id']}",
                )),
                input={"question": row["question"]},
                expected_output={
                    k: v for k, v in row.items()
                    if k in EXPECTED_FIELDS
                },
                metadata={
                    **{
                        k: v for k, v in row.items()
                        if k not in EXPECTED_FIELDS | {"id", "question"}
                    },
                    "question_id": row["id"],
                    "domain": domain,
                    "difficulty": difficulty,
                    "source_file": path.relative_to(ROOT).as_posix(),
                },
            )

        print(f"{dataset_name}: {len(rows)}문항 등록·갱신")

finally:
    langfuse.shutdown()