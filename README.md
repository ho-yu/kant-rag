## 1. Git 커밋 메시지 규칙

`타입: 한 줄 설명` 형식으로 설명은 한글로 짧게 쓴다. 

| 타입 | 언제 | 예시 |
|---|---|---|
| `feat` | 새 기능이나 코드 추가 | `feat: PDF 페이지별 청킹 함수 추가` |
| `fix` | 버그 수정 | `fix: BM25 한국어 토크나이저 오류 수정` |
| `docs` | 문서만 수정 | `docs: 폴더 구조 설계 문서 추가` |
| `chore` | 설정, 의존성, `.gitignore`, config 변경 | `chore: .gitignore 및 .env.example 추가` |
| `refactor` | 동작은 같고 구조만 정리 | `refactor: retriever 인터페이스 분리` |