"""
rag_pipeline.py
ChromaDB + LangChain + Gemini API 기반 RAG 진단 리포트 파이프라인
"""

import os
import re
import numpy as np
import chromadb
from typing import List, Dict
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
TOP_K           = 3
GEMINI_MODEL    = 'gemini-2.5-flash'


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
        self.embedder = SimpleEmbedder()
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
def build_query(segments: List[Dict]) -> str:
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
    return ' '.join(parts)


def build_prompt(
    segments: List[Dict],
    metrics:  Dict,
    model_name: str,
    contexts: List[Dict]
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

    # 규정 컨텍스트
    ctx_text = '\n\n'.join([
        f'[{c["section"]}]\n{c["content"]}'
        for c in contexts
    ])

    # 지표 문자열
    metrics_text = (
        f'F1={metrics.get("f1","N/A")} | '
        f'ROC-AUC={metrics.get("roc_auc","N/A")} | '
        f'PR-AUC={metrics.get("pr_auc","N/A")}'
    )

    return f"""당신은 산업용 펌프 장비 유지보수 전문 엔지니어입니다.
딥러닝 이상 탐지 시스템({model_name})이 다음 결과를 생성했습니다.

[탐지 성능 지표]
{metrics_text}

[탐지된 이상 구간 (총 {len(segments)}건)]
{seg_text}

[관련 장비 정비 규정]
{ctx_text}

위 정보를 바탕으로 아래 항목을 포함하는 진단 리포트를 한국어로 작성해주세요:

1. 이상 원인 분석 (센서 패턴과 규정 기준 근거 명시)
2. 심각도 평가 및 우선 조치 순서
3. 규정에 따른 즉각 / 단기 / 장기 조치 권고
4. 장비 전반적 건강 상태 총평

전문적이고 실용적인 언어로 작성하되, 해당 규정 조항을 근거로 제시해주세요."""


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

    f1_match = re.search(r'F1=([\d.]+)', prompt)
    f1 = float(f1_match.group(1)) if f1_match else 0.0

    severity_summary = []
    if high_count:
        severity_summary.append(f'HIGH {high_count}건')
    if med_count:
        severity_summary.append(f'MEDIUM {med_count}건')
    if low_count:
        severity_summary.append(f'LOW {low_count}건')
    summary = ', '.join(severity_summary) if severity_summary else '이상 없음'

    return f"""# 펌프 장비 이상 탐지 진단 리포트 (규칙 기반)

## 1. 이상 원인 분석
탐지된 이상 구간: {summary}
모델 F1 점수 {f1:.4f} 기준으로 이상 패턴이 감지되었습니다.
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
모델 탐지 성능(F1={f1:.4f})을 고려할 때 탐지 결과의 신뢰도는 {'높음' if f1 > 0.8 else '보통'} 수준입니다.

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
) -> str:
    """
    이상 탐지 결과 → RAG 검색 → Gemini 리포트 생성

    Args:
        segments:   extract_segments()의 반환값
        metrics:    {'f1': float, 'roc_auc': float, 'pr_auc': float}
        model_name: 'CNN1D-AE (Tuned)' 등

    Returns:
        자연어 진단 리포트 (markdown)
    """
    if not segments:
        return '탐지된 이상 구간이 없습니다. 장비가 정상 범위에서 운전 중입니다.'

    db = get_db()
    query    = build_query(segments)
    contexts = db.retrieve(query)
    prompt   = build_prompt(segments, metrics, model_name, contexts)
    report   = call_gemini(prompt)
    return report


# ══════════════════════════════════════════════════════════════════
#  CLI 초기화 (python rag_pipeline.py --init)
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import sys
    if '--init' in sys.argv or '--reset' in sys.argv:
        print('[RAG] ChromaDB 초기화 시작...')
        if _db is not None and '--reset' in sys.argv:
            _db.reset()
        else:
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
