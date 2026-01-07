# AI 면접 코치 - 영상 분석

영상 면접 시뮬레이션을 통해 표정, 시선, 자세를 분석하고 피드백을 제공합니다.

## 분석 항목

| 항목 | 기술 | 분석 내용 |
|------|------|-----------|
| 표정 | DeepFace | 감정 분석 (긴장, 자신감, 미소) |
| 시선 | MediaPipe FaceMesh | 카메라 응시 비율 |
| 자세 | MediaPipe Pose | 자세 안정성, 손 제스처 |

## 프로젝트 구조

```
interview-coach/
├── app.py                 # FastAPI 백엔드
├── requirements.txt       # pip 의존성
├── environment.yml        # conda 환경 설정
├── README.md
└── frontend/              # React 프론트엔드
    ├── package.json
    ├── vite.config.js
    ├── index.html
    └── src/
        ├── main.jsx
        ├── App.jsx
        └── App.css
```

## 설치 방법 (Conda)

### 1. Conda 환경 생성
```bash
conda env create -f environment.yml
conda activate interview-coach
```

### 2. 백엔드 서버 실행
```bash
python app.py
# 서버: http://localhost:8000
```

### 3. 프론트엔드 실행 (새 터미널)
```bash
cd frontend
npm install
npm run dev
# 앱: http://localhost:3000
```

## API 사용법

### 영상 분석
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@interview.mp4"
```

### 응답 예시
```json
{
  "emotion": {
    "average_emotions": {
      "happy": 15.2,
      "neutral": 65.8,
      "sad": 5.1,
      "angry": 3.2,
      "fear": 8.4,
      "surprise": 1.8,
      "disgust": 0.5
    },
    "dominant_emotion": "neutral",
    "feedback": "차분하고 안정적입니다 ✅",
    "samples_analyzed": 45
  },
  "gaze": {
    "center_gaze_ratio": 78.5,
    "feedback": "카메라를 잘 응시하고 있어요 ✅",
    "samples_analyzed": 89
  },
  "posture": {
    "posture_stability": 85.2,
    "posture_feedback": "자세가 안정적이에요 ✅",
    "hand_gesture_level": "적당",
    "gesture_feedback": "적절한 제스처입니다 ✅",
    "samples_analyzed": 89
  },
  "overall": {
    "score": 82.5,
    "feedback": "전반적으로 훌륭한 면접 태도입니다! 🎉"
  }
}
```

## 기술 스택

- **Backend**: FastAPI
- **AI Models**: DeepFace, MediaPipe
- **Video Processing**: OpenCV

## 주의사항

- Python 3.10 권장 (3.12는 호환 이슈 가능)
- 첫 실행 시 DeepFace 모델 다운로드로 시간 소요
- GPU 있으면 더 빠름 (없어도 동작)
