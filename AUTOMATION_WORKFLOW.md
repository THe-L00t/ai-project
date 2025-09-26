# 🔄 AI 프로젝트 자동화 워크플로우

## 📋 개요
AI 코인 거래 프로젝트의 코드 수정 시 자동으로 실행되는 워크플로우입니다.

## 🔀 자동화 트리거 조건

### 1. 핵심 파일 수정 시 자동 실행
- `smart_hybrid_ai.py` - 메인 하이브리드 AI
- `adaptive_learning_ai.py` - 적응형 학습 AI
- `main.py` / `main_v2.py` - 메인 실행 파일
- `requirements.txt` - 의존성 패키지
- `project_config.md` - 프로젝트 설정

### 2. 에러 수정 완료 시
- timestamp 관련 에러
- API 연결 에러
- 거래 실행 에러
- 학습 모델 에러

## 🎯 자동화 실행 단계

### Step 1: 코드 검증
```bash
# Python 문법 오류 확인
python3 -m py_compile smart_hybrid_ai.py

# 의존성 패키지 확인
pip3 check
```

### Step 2: 데스크톱 단축키 갱신
#### 갱신 대상 파일:
- `/Users/the-l00t/Desktop/start_ai_learning.command`
- `/Users/the-l00t/Desktop/view_ai_logs.command`

#### 갱신 내용:
1. **타임스탬프 업데이트**
   ```bash
   # 스크립트 헤더에 갱신 시간 추가
   # CoinTradingAI 학습 시작 스크립트 (자동 갱신: 2025-09-27 12:30:45)
   ```

2. **파일 존재 확인 강화**
   ```bash
   # 핵심 AI 파일들 존재 확인
   if [ ! -f "smart_hybrid_ai.py" ]; then
       echo "❌ 필수 파일을 찾을 수 없습니다!"
       exit 1
   fi
   ```

3. **의존성 자동 설치**
   ```bash
   # requirements.txt 기반 패키지 설치
   if [ -f "requirements.txt" ]; then
       pip3 install -r requirements.txt --quiet
   fi
   ```

4. **실행 권한 재설정**
   ```bash
   chmod +x /Users/the-l00t/Desktop/*.command
   ```

### Step 3: Git 커밋 자동화

#### 커밋 메시지 형식:
```
[타입]: 간단한 요약 (50자 이내)

상세 설명:
- 변경 이유: [timestamp 에러 수정]
- 주요 수정 내용: [record_trade_result 함수 파라미터 매핑 오류 해결]
- 테스트 결과: [로그에서 timestamp 에러 제거 확인]
- 영향 범위: [smart_hybrid_ai.py 매도 로직]

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>
```

#### 커밋 타입 분류:
- `fix`: 버그 수정
- `feat`: 새 기능 추가
- `improve`: 기존 기능 개선
- `refactor`: 코드 리팩토링
- `config`: 설정 파일 변경
- `docs`: 문서 업데이트

### Step 4: 변경사항 로깅
```bash
# 자동화 로그 파일에 기록
echo "$(date): [자동화] timestamp 에러 수정 완료" >> logs/automation.log
echo "  - 수정 파일: smart_hybrid_ai.py:703-710" >> logs/automation.log
echo "  - 커밋 ID: $(git rev-parse HEAD)" >> logs/automation.log
```

## 📊 자동화 규칙 상세

### Git 커밋 규칙
1. **원자적 커밋**: 하나의 논리적 변경사항당 하나의 커밃
2. **분리 원칙**: 에러 수정과 기능 개선은 별도 커밋
3. **설정 관리**: 프로젝트 설정 변경은 독립 커밋

### 데스크톱 단축키 갱신 규칙
1. **타임스탬프**: 매 갱신 시 현재 시간 기록
2. **검증 로직**: 실행 전 필수 파일 존재 확인
3. **환경 설정**: Python 경로와 가상환경 자동 감지
4. **에러 처리**: 실행 실패 시 상세 오류 메시지

### 예외 처리
1. **Git 커밋 실패**
   - 충돌 발생 시 수동 해결 안내
   - 권한 오류 시 SSH 키 확인 요청

2. **단축키 갱신 실패**
   - 파일 권한 오류 시 chmod 재실행
   - 경로 오류 시 수동 확인 요청

3. **테스트 실패**
   - 문법 오류 시 커밋 중단
   - 의존성 오류 시 requirements.txt 확인

## 🔍 모니터링 및 알림

### 성공 지표
- ✅ 코드 문법 검증 통과
- ✅ 단축키 파일 갱신 완료
- ✅ Git 커밋 성공
- ✅ 자동화 로그 기록 완료

### 실패 처리
- ❌ 상세 에러 메시지 출력
- ❌ 실패 단계 및 원인 로깅
- ❌ 수동 개입 필요 사항 안내

### 통계 수집
- 일일 자동화 실행 횟수
- 성공/실패 비율
- 주요 에러 유형 분석
- 코드 수정 빈도 추적

## 🎮 사용법

### 수동 실행
```bash
# 프로젝트 디렉토리에서
cd /Users/the-l00t/Project/CoinTradingAI

# 자동화 워크플로우 실행
# (향후 automation_workflow.sh 스크립트 생성 예정)
```

### 자동 실행 확인
```bash
# 자동화 로그 확인
tail -f logs/automation.log

# 최근 커밋 확인
git log --oneline -10

# 단축키 갱신 상태 확인
ls -la /Users/the-l00t/Desktop/*.command
```

---

**마지막 업데이트**: 2025-09-27
**다음 예정 개선사항**:
- automation_workflow.sh 스크립트 생성
- 실시간 알림 시스템 구축
- 성능 지표 대시보드 연동