"""
rag_pipeline.py
ChromaDB + LangChain + Gemini API 기반 RAG 진단 리포트 파이프라인
"""

import os
import re
import time
import numpy as np
import chromadb
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════════
#  설정
# ══════════════════════════════════════════════════════════════════
RAG_DIR         = os.path.join(os.path.dirname(__file__), 'rag')
REGULATIONS_TXT = os.path.join(RAG_DIR, 'pump_regulations.txt')
CHROMA_DIR      = os.path.join(RAG_DIR, 'chroma_db')
COLLECTION_NAME = 'pump_regulations'
EMBED_DIM       = 512
# 검색 조각 수: 환경변수 RAG_TOP_K (기본 5). 임베딩 종류 바꾼 뒤에는 `python rag_pipeline.py --reset` 권장.
TOP_K           = max(1, int(os.getenv('RAG_TOP_K', '5')))
GEMINI_MODEL    = 'gemini-2.5-flash'

# 근거 점검 시 제외할 일반어·프롬프트 고정어 (환각 휴리스틱용)
_GROUNDING_STOP = frozenset({
    '입니다', '합니다', '됩니다', '있습니다', '없습니다', '따라서', '때문에', '경우', '필요', '조치', '규정',
    '관련', '이상', '탐지', '모델', '장비', '상태', '평가', '분석', '리포트', '작성', '기준', '대상', '확인',
    '점검', '결과', '진단', '심각도', '우선', '즉각', '단기', '장기', '총평', '전문적', '실용적', '근거',
    '제시', '포함', '다음', '항목', '원인', '패턴', '명시', '순서', '권고', '건강', '유지보수', '엔지니어',
    '산업용', '펌프', '센서', '데이터', '재구성', '오차', '윈도우', '인덱스', '지속', '최대', '평균',
    '성능', '지표', '정보', '바탕', '작성해주세요', '문서', '참고', '아래', '위해', '통해', '대한', '하는',
    '있는', '없는', '같은', '위한', '대한', '및', '등', '수', '것', '때',
})


# ══════════════════════════════════════════════════════════════════
#  임베딩 (외부 모델 없이 로컬 동작)
# ══════════════════════════════════════════════════════════════════
class SimpleEmbedder:
    """
    네트워크 없이 동작하는 키워드 기반 임베딩.
    sentence-transformers 설치 시 교체 가능.
    """
    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            vec = np.zeros(self.dim)
            words = re.findall(r'[\w가-힣]+', text)
            for word in words:
                for i in range(min(len(word), 3)):
                    h = abs(hash(word[i:])) % self.dim
                    vec[h] += 1.0 / (i + 1)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            results.append(vec.tolist())
        return results


class VertexTextEmbedder:
    """
    Vertex AI Text Embeddings (한국어 규정에 맞게 기본: multilingual).
    GCP_PROJECT_ID / GCP_LOCATION 필요. 실패 시 상위에서 SimpleEmbedder로 폴백.
    """
    def __init__(self):
        import vertexai
        from vertexai.language_models import TextEmbeddingModel

        project = os.getenv('GCP_PROJECT_ID')
        if not project:
            raise RuntimeError('GCP_PROJECT_ID 가 설정되지 않았습니다.')
        location = os.getenv('GCP_LOCATION', 'asia-northeast3')
        vertexai.init(project=project, location=location)
        model_id = os.getenv('VERTEX_EMBEDDING_MODEL', 'text-multilingual-embedding-002')
        self._model = TextEmbeddingModel.from_pretrained(model_id)
        self._batch = max(1, int(os.getenv('RAG_EMBED_BATCH', '16')))
        self._sleep = float(os.getenv('RAG_EMBED_SLEEP_SEC', '0.25'))

    def embed(self, texts: List[str]) -> List[List[float]]:
        safe = [t if (t and str(t).strip()) else ' ' for t in texts]
        out: List[List[float]] = []
        for i in range(0, len(safe), self._batch):
            batch = safe[i:i + self._batch]
            embs = self._model.get_embeddings(batch)
            for e in embs:
                out.append(list(e.values))
            if i + self._batch < len(safe) and self._sleep > 0:
                time.sleep(self._sleep)
        return out


def _make_embedder():
    """RAG_EMBEDDER=vertex 이면 Vertex, 아니면 Simple. vertex 실패 시 Simple."""
    mode = os.getenv('RAG_EMBEDDER', 'simple').strip().lower()
    if mode == 'vertex':
        try:
            emb = VertexTextEmbedder()
            print('[RAG] 임베딩: Vertex (text-multilingual 등)')
            return emb
        except Exception as e:
            print(f'[RAG] Vertex 임베딩 불가 → SimpleEmbedder 사용: {e}')
    print('[RAG] 임베딩: SimpleEmbedder (로컬 해시·키워드)')
    return SimpleEmbedder()


# ══════════════════════════════════════════════════════════════════
#  텍스트 청킹
# ══════════════════════════════════════════════════════════════════
def chunk_by_section(text: str) -> List[Dict[str, str]]:
    """마크다운 헤더 기준 섹션 단위 청킹"""
    chunks = []
    current_section = '일반'
    current_content: List[str] = []

    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('###') or line.startswith('##'):
            if current_content:
                content = ' '.join(current_content)
                if len(content) > 20:
                    chunks.append({
                        'section': current_section,
                        'content': content
                    })
            current_section = re.sub(r'^#+\s*', '', line).strip()
            current_content = []
        elif not line.startswith('#'):
            # 마크다운 강조 제거
            clean = re.sub(r'\*\*(.+?)\*\*', r'\1', line)
            current_content.append(clean)

    if current_content:
        content = ' '.join(current_content)
        if len(content) > 20:
            chunks.append({'section': current_section, 'content': content})

    return chunks


# ══════════════════════════════════════════════════════════════════
#  ChromaDB 관리
# ══════════════════════════════════════════════════════════════════
class RegulationDB:
    def __init__(self):
        self.embedder = _make_embedder()
        os.makedirs(CHROMA_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = None
        self._init_collection()

    def _init_collection(self):
        existing = [c.name for c in self.client.list_collections()]
        if COLLECTION_NAME in existing:
            self.collection = self.client.get_collection(COLLECTION_NAME)
            print(f'[RAG] 기존 ChromaDB 로드 ({self.collection.count()}개 청크)')
        else:
            self._build()

    def _build(self):
        """규정 문서 벡터화 + 저장"""
        if not os.path.exists(REGULATIONS_TXT):
            raise FileNotFoundError(f'규정 문서를 찾을 수 없습니다: {REGULATIONS_TXT}')

        with open(REGULATIONS_TXT, encoding='utf-8') as f:
            text = f.read()

        chunks = chunk_by_section(text)
        if not chunks:
            raise ValueError('청킹 결과가 비어 있습니다.')

        self.collection = self.client.create_collection(COLLECTION_NAME)
        docs  = [c['content']  for c in chunks]
        metas = [{'section': c['section']} for c in chunks]
        ids   = [f'chunk_{i}' for i in range(len(chunks))]
        embs  = self.embedder.embed(docs)

        self.collection.add(
            documents=docs,
            ids=ids,
            embeddings=embs,
            metadatas=metas
        )
        print(f'[RAG] ChromaDB 초기화 완료 ({len(chunks)}개 청크 저장)')

    def retrieve(self, query: str, n: int = TOP_K) -> List[Dict]:
        q_emb = self.embedder.embed([query])
        res = self.collection.query(
            query_embeddings=q_emb,
            n_results=min(n, self.collection.count())
        )
        return [
            {'section': meta['section'], 'content': doc}
            for doc, meta in zip(res['documents'][0], res['metadatas'][0])
        ]

    def reset(self):
        """DB 초기화 (재구축 시 사용)"""
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self._build()


# ══════════════════════════════════════════════════════════════════
#  프롬프트 빌더
# ══════════════════════════════════════════════════════════════════
def build_query(segments: List[Dict], top3_sensors: Optional[List[Tuple[str, float]]] = None) -> str:
    high = [s for s in segments if s['severity'] == 'HIGH']
    mid  = [s for s in segments if s['severity'] == 'MEDIUM']
    low  = [s for s in segments if s['severity'] == 'LOW']
    parts = []
    if high:
        parts.append(f'HIGH 심각도 이상 {len(high)}건 즉각 조치')
    if mid:
        parts.append(f'MEDIUM 심각도 이상 {len(mid)}건')
    if low:
        parts.append(f'LOW 이상 {len(low)}건')
    parts.append('원인 조치 규정 진단')
    q = ' '.join(parts)
    if top3_sensors:
        names = ' '.join(t[0] for t in top3_sensors)
        q = f'{q} {names} 센서 점검 진동 온도 압력 유량'
    return q


def build_prompt(
    segments: List[Dict],
    metrics:  Dict,
    model_name: str,
    contexts: List[Dict],
    top3_sensors: Optional[List[Tuple[str, float]]] = None,
) -> str:
    # 이상 구간 요약 (최대 6건)
    seg_lines = []
    for i, s in enumerate(segments[:6]):
        seg_lines.append(
            f'  구간{i+1}: 인덱스 {s["start"]}~{s["end"]} '
            f'(지속 {s["duration"]}윈도우) | '
            f'심각도={s["severity"]} | 최대오차={s["max_err"]:.5f}'
        )
    seg_text = '\n'.join(seg_lines) if seg_lines else '  탐지된 이상 구간 없음'

    if top3_sensors:
        t3_lines = [
            f'  {i+1}. {name} (스케일 공간 평균 재구성 오차 MSE ≈ {val:.6f})'
            for i, (name, val) in enumerate(top3_sensors)
        ]
        top3_block = (
            '[오토인코더 기반 의심 센서 — 재구성 오차 기여도 상위 3개]\n'
            + '\n'.join(t3_lines)
            + '\n  위 센서를 우선적으로 규정 조항과 대조해 원인 후보를 논의하세요. '
            '물리적 부품 고장으로 단정하지 마세요.'
        )
    else:
        top3_block = (
            '[오토인코더 기반 의심 센서]\n'
            '  (이상 윈도우 없음 또는 미산출 — 일반 규정 검토만 수행)'
        )

    # 규정 컨텍스트
    ctx_text = '\n\n'.join([
        f'[{c["section"]}]\n{c["content"]}'
        for c in contexts
    ])

    # 지표 문자열 (혼동·해석 포함 시 LLM이 과장·오해하지 않도록)
    m = metrics
    metrics_lines = [
        f'F1={m.get("f1", "N/A")} | ROC-AUC={m.get("roc_auc", "N/A")} | PR-AUC={m.get("pr_auc", "N/A")}',
    ]
    if 'precision' in m and 'recall' in m:
        metrics_lines.append(
            f'Precision={m.get("precision", "N/A")} | Recall={m.get("recall", "N/A")}'
        )
    if all(k in m for k in ('tn', 'fp', 'fn', 'tp')):
        metrics_lines.append(
            f'윈도우 혼동 TN={m["tn"]}, FP={m["fp"]}, FN={m["fn"]}, TP={m["tp"]}'
        )
    if m.get('metrics_note'):
        metrics_lines.append(f'[지표 해석] {m["metrics_note"]}')
    metrics_text = '\n'.join(metrics_lines)

    caution = (
        'F1이 0에 가깝거나 한쪽 클래스(라벨)만 있으면 "시스템이 무용지물"처럼 단정하지 마세요. '
        '윈도우 라벨 정의·데이터 구간·임계값(슬라이더)에 따라 달라지는 참고 지표임을 분명히 하세요.'
    )

    output_rules = (
        '[출력 규칙 — 반드시 준수]\n'
        '- 아래 [관련 장비 정비 규정] 텍스트에 **없는** 구체적 수치·부품 모델명·조항 번호·날짜를 **지어내지 마세요**.\n'
        '- 규정에 근거할 때는 검색된 조각에 실제로 나온 표현만 인용하거나, 일반적 권고로만 서술하세요.\n'
        '- Top-3 센서는 **재구성 오차 기여도 후보**일 뿐, 특정 부품 고장으로 단정하지 마세요.\n'
        '- 지표(F1·AUC 등)가 낮거나 N/A이면, 그 **한계**를 한 문장으로 밝히고 과장된 신뢰 표현을 피하세요.'
    )

    return f"""당신은 산업용 펌프 장비 유지보수 전문 엔지니어입니다.
딥러닝 이상 탐지 시스템({model_name})이 다음 결과를 생성했습니다.

[탐지 성능 지표]
{metrics_text}

{caution}

{output_rules}

{top3_block}

[탐지된 이상 구간 (총 {len(segments)}건)]
{seg_text}

[관련 장비 정비 규정]
{ctx_text}

위 정보를 바탕으로 아래 항목을 포함하는 진단 리포트를 한국어로 작성해주세요:

1. 이상 원인 분석 (센서 패턴과 규정 기준 근거 명시)
2. 심각도 평가 및 우선 조치 순서
3. 규정에 따른 즉각 / 단기 / 장기 조치 권고
4. 장비 전반적 건강 상태 총평

전문적이고 실용적인 언어로 작성하되, **검색된 규정 본문에 기반한 부분**과 **일반 권고**를 구분해 읽을 수 있게 작성해주세요."""


def light_grounding_check(
    report: str,
    contexts: List[Dict],
    allow_names: Optional[List[str]] = None,
) -> str:
    """
    답변에 등장한 토큰이 검색된 규정 청크(문자열)에 있는지 샘플 검사.
    완전한 환각 탐지가 아니라 PoC 수준 가드레일.
    """
    allow_names = allow_names or []
    allow_lower = {a.lower() for a in allow_names}
    blob = '\n'.join(c['content'] for c in contexts)
    blob_l = blob.lower()

    raw = re.findall(r'[가-힣]{3,}|[A-Za-z][A-Za-z0-9_\-]{2,}', report)
    missing: List[str] = []
    seen = set()
    for w in raw:
        if w in _GROUNDING_STOP:
            continue
        key = w.lower() if w.isascii() else w
        if key in seen:
            continue
        seen.add(key)
        if allow_names and w in allow_names:
            continue
        if re.match(r'^sensor_\d+$', w, re.I) and w.lower() in allow_lower:
            continue
        in_blob = w in blob if not w.isascii() else (w in blob or w.lower() in blob_l)
        if not in_blob:
            missing.append(w)
        if len(missing) >= 18:
            break

    if not missing:
        return (
            '\n\n---\n*근거 점검(자동):* 이번에 검색·주입된 규정 조각에 대해, '
            '답변에서 뽑은 주요 토큰(일반어 제외)이 문자열로 존재하는지 샘플 확인했습니다. '
            '의미 동일·요약은 반영되지 않을 수 있습니다.\n'
        )

    shown = ', '.join(missing[:15])
    more = f' 외 {len(missing) - 15}개' if len(missing) > 15 else ''
    return (
        f'\n\n---\n⚠️ **근거 점검(자동):** 아래 용어는 **이번에 검색된 규정 조각**에 동일 문자열이 없습니다. '
        f'요약·일반 상식·환각일 수 있으니 매뉴얼 원문과 대조하세요.\n\n`{shown}`{more}\n'
    )


# ══════════════════════════════════════════════════════════════════
#  Gemini API 호출
# ══════════════════════════════════════════════════════════════════
def call_gemini(prompt: str) -> str:
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel

        # .env에서 불러온 값을 변수에 할당
        project_id = os.getenv('GCP_PROJECT_ID')
        location = os.getenv('GCP_LOCATION', 'asia-northeast3') # 기본값을 서울로 변경
        
        # Vertex AI 초기화
        vertexai.init(
            project=project_id,
            location=location
        )
        
        # 모델 설정 (현재 gemini-2.0-flash 사용 중이시네요!)
        model = GenerativeModel('gemini-2.5-flash') # 또는 'gemini-2.0-flash-exp'
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        print(f'[RAG] Vertex AI 오류: {e} → 규칙 기반 리포트로 대체')
        return _rule_based_report(prompt)
        

def _rule_based_report(prompt: str) -> str:
    """API 키 없을 때 규칙 기반 리포트 자동 생성"""
    # 프롬프트에서 핵심 정보 추출
    high_count  = len(re.findall(r'HIGH', prompt))
    med_count   = len(re.findall(r'MEDIUM', prompt))
    low_count   = len(re.findall(r'LOW', prompt))

    f1_match = re.search(r'F1=([\d.]+|N/A)', prompt)
    f1 = float(f1_match.group(1)) if f1_match and f1_match.group(1) != 'N/A' else 0.0

    severity_summary = []
    if high_count:
        severity_summary.append(f'HIGH {high_count}건')
    if med_count:
        severity_summary.append(f'MEDIUM {med_count}건')
    if low_count:
        severity_summary.append(f'LOW {low_count}건')
    summary = ', '.join(severity_summary) if severity_summary else '이상 없음'

    f1_hint = (
        f'윈도우 라벨이 있을 때 F1≈{f1:.4f} 입니다. F1이 매우 낮으면 구간·임계값·라벨 정의 영향일 수 있어 '
        '재구성 오차 탐지 결과와 별도로 해석하세요.'
    )

    return f"""# 펌프 장비 이상 탐지 진단 리포트 (규칙 기반)

## 1. 이상 원인 분석
탐지된 이상 구간: {summary}
{f1_hint}
{"HIGH 등급 이상이 감지되어 즉각적인 점검이 필요합니다. 제3조에 따라 두 개 이상의 센서 동시 이상 또는 재구성 오차 임계값 3배 초과 시 즉각 조치 대상입니다." if high_count else ""}
{"MEDIUM 등급 이상이 감지되었습니다. 제3조에 따라 24시간 이내 점검 일정을 수립하시기 바랍니다." if med_count else ""}

## 2. 심각도 평가 및 우선 조치 순서
{"1. HIGH 등급 구간 즉각 점검 (운전 중단 검토)" if high_count else ""}
{"2. MEDIUM 등급 구간 24시간 이내 점검" if med_count else ""}
{"3. LOW 등급 구간 정기 점검 시 확인" if low_count else ""}

## 3. 조치 권고
- **즉각**: {"HIGH 등급 이상 구간 운전 중단 및 담당 엔지니어 호출" if high_count else "현재 즉각 조치 불필요"}
- **단기**: 베어링, 시일, 임펠러 상태 점검 (제7조 기준)
- **장기**: 예방 정비 주기 단축 검토, 센서 이상 이력 누적 분석

## 4. 장비 건강 상태 총평
{"⚠️ 주의 필요 — 즉각 점검 권고" if high_count else "✅ 정상 범위 — 정기 모니터링 유지"}
라벨이 있는 경우에도 F1은 윈도우·임계값 설정에 민감합니다. 신뢰도는 {'높음' if f1 > 0.8 else ('참고(낮음)' if f1 < 0.15 else '보통')}으로만 참고하세요.

*본 리포트는 규칙 기반 자동 생성 결과입니다. Gemini API 연동 시 더 정밀한 분석이 제공됩니다.*
"""


# ══════════════════════════════════════════════════════════════════
#  메인 인터페이스
# ══════════════════════════════════════════════════════════════════
# 싱글턴
_db: RegulationDB = None

def get_db() -> RegulationDB:
    global _db
    if _db is None:
        _db = RegulationDB()
    return _db


def generate_report(
    segments:   List[Dict],
    metrics:    Dict,
    model_name: str,
    top3_sensors: Optional[List[Tuple[str, float]]] = None,
) -> str:
    """
    이상 탐지 결과 → RAG 검색 → Gemini 리포트 생성

    Args:
        segments:   extract_segments()의 반환값
        metrics:    {'f1': float, 'roc_auc': float, 'pr_auc': float}
        model_name: 'CNN1D-AE (Tuned)' 등
        top3_sensors: AE 기반 의심 센서 (이름, MSE) — 검색 쿼리·프롬프트에 주입

    Returns:
        자연어 진단 리포트 (markdown)
    """
    if not segments:
        return '탐지된 이상 구간이 없습니다. 장비가 정상 범위에서 운전 중입니다.'

    db = get_db()
    query    = build_query(segments, top3_sensors)
    contexts = db.retrieve(query)
    prompt   = build_prompt(segments, metrics, model_name, contexts, top3_sensors)
    report   = call_gemini(prompt)

    if '규칙 기반' in report:
        return report

    allow = [t[0] for t in top3_sensors] if top3_sensors else []
    return report + light_grounding_check(report, contexts, allow_names=allow)


def run_retrieval_eval(queries: Optional[List[str]] = None) -> None:
    """Chroma retrieve만 출력 — Before/After 표 작성용."""
    qs = queries or [
        'HIGH 심각도 이상 1건 즉각 조치 원인 조치 규정 진단',
        '진동 온도 압력 유량 센서 점검',
        '펌프 이상 운전 정비 규정',
        '베어링 진동 이상',
        '즉각 조치 및 보고',
    ]
    db = get_db()
    print(f'[RAG] eval | TOP_K={TOP_K} | embedder={type(db.embedder).__name__}')
    for q in qs:
        hits = db.retrieve(q, n=TOP_K)
        secs = [h['section'] for h in hits]
        print(f'\nQ: {q}\n  → sections ({len(secs)}): {secs}')


# ══════════════════════════════════════════════════════════════════
#  CLI 초기화 (python rag_pipeline.py --init | --reset | --eval)
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import sys
    if '--eval' in sys.argv:
        run_retrieval_eval()
    elif '--reset' in sys.argv:
        print('[RAG] ChromaDB reset (컬렉션 삭제 후 재빌드). 임베딩 종류를 바꿨다면 필수.')
        _db = RegulationDB()
        _db.reset()
        print('[RAG] 완료')
    elif '--init' in sys.argv:
        print('[RAG] ChromaDB 준비 (없으면 생성)...')
        get_db()
        print('[RAG] 완료')
    else:
        # 파이프라인 테스트
        print('[RAG] 파이프라인 테스트 시작...')
        test_segments = [
            {'start': 100, 'end': 115, 'duration': 15,
             'max_err': 0.0892, 'mean_err': 0.0612, 'severity': 'HIGH'},
            {'start': 380, 'end': 390, 'duration': 10,
             'max_err': 0.0341, 'mean_err': 0.0280, 'severity': 'MEDIUM'},
        ]
        test_metrics = {'f1': 0.8595, 'roc_auc': 0.9962, 'pr_auc': 0.9519}
        report = generate_report(test_segments, test_metrics, 'CNN1D-AE (Tuned)')
        print('\n' + '='*60)
        print(report)
        print('='*60)
        print('[RAG] 테스트 완료')
