import os
# 환경 변수 설정
os.environ['TF_USE_LEGACY_KERAS'] = '0'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import shutil
import uuid
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import librosa
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
import subprocess
import tempfile

# 1. FastAPI 앱 초기화
app = FastAPI(title="AI Interviewer API", description="AI 면접관의 백엔드 API 서비스")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 모델 및 도구 초기화
print("Loading Models... This may take a while.")

device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'CUDA' if device == 0 else 'CPU'}")

# STT
print("Loading Whisper...")
stt_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=device)

# 감정 인식 (Transformers)
print("Loading Emotion Model...")
emotion_pipe = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", device=device)

# LLM (속도 개선을 위해 0.5B 모델 사용)
print("Loading LLM...")
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Mediapipe
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

print("Models Loaded Successfully!")

# 3. 분석 함수 정의 (app.py에서 이식)

def analyze_emotions(video_path: str, sample_rate: int = 90) -> dict: # 60 -> 90 (3초에 1번)
    """영상에서 표정/감정 분석"""
    cap = cv2.VideoCapture(video_path)
    emotions_log = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % sample_rate == 0:
            try:
                # 리사이징으로 속도 향상
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                result = DeepFace.analyze(
                    small_frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True,
                    detector_backend='opencv' # 가벼운 백엔드 사용
                )
                emotions_log.append(result[0]['emotion'])
            except Exception:
                pass

        frame_count += 1

    cap.release()

    if not emotions_log:
        return {"error": "표정 인식 실패"}

    avg_emotions = {}
    for key in emotions_log[0].keys():
        avg_emotions[key] = round(sum(e[key] for e in emotions_log) / len(emotions_log), 2)

    dominant = max(avg_emotions, key=avg_emotions.get)
    
    emotion_feedback = {
        "happy": "밝은 표정이 좋습니다",
        "neutral": "차분하고 안정적입니다",
        "sad": "조금 더 밝은 표정을 지어보세요",
        "angry": "표정이 딱딱해 보일 수 있어요",
        "fear": "긴장한 것처럼 보여요",
        "surprise": "자연스러운 표정을 유지하세요",
        "disgust": "표정 관리가 필요해요"
    }

    return {
        "average_emotions": avg_emotions,
        "dominant_emotion": dominant,
        "feedback": emotion_feedback.get(dominant, ""),
        "samples_analyzed": len(emotions_log)
    }

def get_iris_center(landmarks, indices, img_w, img_h):
    points = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in indices]
    return np.mean(points, axis=0)

def analyze_gaze(video_path: str, sample_rate: int = 45) -> dict: # 15 -> 45 (1.5초에 1번)
    """시선 추적"""
    cap = cv2.VideoCapture(video_path)
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    center_x = img_w / 2

    # 리사이징 비율 계산 (얼굴 인식이 잘 되는 선에서 축소)
    scale_factor = 0.5 
    
    gaze_results = []
    frame_count = 0

    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_rate == 0:
                # Mediapipe는 원본 해상도 유지 권장하지만 너무 크면 느림 -> 적당히 리사이징
                small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
                rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    left_iris = get_iris_center(landmarks, LEFT_IRIS, img_w, img_h)
                    right_iris = get_iris_center(landmarks, RIGHT_IRIS, img_w, img_h)
                    avg_iris_x = (left_iris[0] + right_iris[0]) / 2
                    
                    is_looking_center = abs(avg_iris_x - center_x) < (img_w * 0.15)
                    gaze_results.append(is_looking_center)

            frame_count += 1

    cap.release()

    if not gaze_results:
        return {"error": "시선 인식 실패"}

    center_ratio = sum(gaze_results) / len(gaze_results) * 100
    
    if center_ratio >= 70:
        feedback = "카메라를 잘 응시하고 있어요"
    elif center_ratio >= 50:
        feedback = "카메라를 조금 더 바라봐주세요"
    else:
        feedback = "시선이 많이 흔들려요. 카메라를 응시해주세요"

    return {
        "center_gaze_ratio": round(center_ratio, 1),
        "feedback": feedback,
        "samples_analyzed": len(gaze_results)
    }

def analyze_posture(video_path: str, sample_rate: int = 60) -> dict: # 15 -> 60 (2초에 1번)
    """자세 및 제스처 분석"""
    cap = cv2.VideoCapture(video_path)
    shoulder_positions = []
    hand_movements = []
    prev_wrist_left = None
    prev_wrist_right = None
    frame_count = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_rate == 0:
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # 어깨
                    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
                    shoulder_positions.append(shoulder_y)

                    # 손목
                    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

                    if prev_wrist_left and prev_wrist_right:
                        left_move = np.sqrt((left_wrist.x - prev_wrist_left.x)**2 + (left_wrist.y - prev_wrist_left.y)**2)
                        right_move = np.sqrt((right_wrist.x - prev_wrist_right.x)**2 + (right_wrist.y - prev_wrist_right.y)**2)
                        hand_movements.append((left_move + right_move) / 2)

                    prev_wrist_left = left_wrist
                    prev_wrist_right = right_wrist

            frame_count += 1

    cap.release()

    if not shoulder_positions:
        return {"error": "자세 인식 실패"}

    shoulder_std = np.std(shoulder_positions)
    stability = max(0, 100 - shoulder_std * 500)
    
    avg_movement = np.mean(hand_movements) if hand_movements else 0
    if avg_movement > 0.05:
        gesture_level = "많음"
        gesture_feedback = "손 제스처가 많아요. 조금 줄여보세요"
    elif avg_movement > 0.02:
        gesture_level = "적당"
        gesture_feedback = "적절한 제스처입니다"
    else:
        gesture_level = "적음"
        gesture_feedback = "자연스러운 제스처를 추가해보세요"

    if stability >= 80:
        posture_feedback = "자세가 안정적이에요"
    elif stability >= 60:
        posture_feedback = "자세가 약간 흔들려요"
    else:
        posture_feedback = "자세를 고정하고 안정감을 유지하세요"

    return {
        "posture_stability": round(stability, 1),
        "posture_feedback": posture_feedback,
        "hand_gesture_level": gesture_level,
        "gesture_feedback": gesture_feedback,
        "samples_analyzed": len(shoulder_positions)
    }

def analyze_interview_video(video_path: str) -> dict:
    """종합 비디오 분석"""
    results = {
        "emotion": analyze_emotions(video_path),
        "gaze": analyze_gaze(video_path),
        "posture": analyze_posture(video_path)
    }

    scores = []
    if "dominant_emotion" in results["emotion"]:
        emotion = results["emotion"]["dominant_emotion"]
        emotion_scores = {"happy": 90, "neutral": 85, "surprise": 70, "sad": 50, "angry": 40, "fear": 45, "disgust": 40}
        scores.append(emotion_scores.get(emotion, 60))

    if "center_gaze_ratio" in results["gaze"]:
        scores.append(results["gaze"]["center_gaze_ratio"])

    if "posture_stability" in results["posture"]:
        scores.append(results["posture"]["posture_stability"])

    overall_score = round(sum(scores) / len(scores), 1) if scores else 0
    
    if overall_score >= 80:
        overall_feedback = "전반적으로 훌륭한 면접 태도입니다!"
    elif overall_score >= 60:
        overall_feedback = "좋은 편이지만 개선할 부분이 있어요"
    else:
        overall_feedback = "연습이 더 필요해요. 피드백을 참고해주세요"

    results["overall"] = {
        "score": overall_score,
        "feedback": overall_feedback
    }
    return results

def extract_audio_from_video(video_path: str) -> str:
    """ffmpeg로 오디오 추출"""
    try:
        audio_path = tempfile.mktemp(suffix=".wav")
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            "-y", audio_path
        ]
        # 윈도우에서 콘솔 창 뜨지 않게 설정
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        
        subprocess.run(cmd, capture_output=True, text=True, startupinfo=startupinfo)
        
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            return audio_path
        return None
    except Exception as e:
        print(f"오디오 추출 실패: {e}")
        return None

# 4. 데이터 모델 정의
class QuestionRequest(BaseModel):
    topic: str
    difficulty: str

class QuestionResponse(BaseModel):
    question: str

# 5. 엔드포인트 구현

@app.post("/generate_question", response_model=QuestionResponse)
async def generate_question_api(req: QuestionRequest):
    if not req.topic:
        raise HTTPException(status_code=400, detail="Topic is required")

    prompt = f"당신은 면접관입니다. '{req.topic}' 직무/주제와 관련된 면접 질문을 딱 하나만 던지세요. 난이도는 {req.difficulty}입니다. 절대 답변 예시나 빈칸 채우기 형식으로 만들지 말고, 지원자에게 묻는 '의문문' 형식의 질문 하나만 출력하세요. 인사말은 생략합니다."
    
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([input_text], return_tensors="pt").to(llm_model.device)

    generated_ids = llm_model.generate(inputs.input_ids, max_new_tokens=100)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    question_text = response.split("assistant\n")[-1].strip()

    return QuestionResponse(question=question_text)

@app.post("/analyze")
async def analyze_interview_api(
    file: UploadFile = File(None),  # 비디오 또는 오디오 파일
    text_answer: str = Form(None),
    difficulty: str = Form("초급"),
    question: str = Form(...)
):
    has_file = file is not None
    has_text = text_answer is not None and len(text_answer.strip()) > 0

    if not has_file and not has_text:
        raise HTTPException(status_code=400, detail="File or text answer is required")

    response_data = {
        "stt_result": "",
        "audio_emotion": None,
        "video_analysis": None,
        "feedback": "",
        "best_answer": ""
    }

    temp_file_path = None
    extracted_audio_path = None

    try:
        if has_file:
            # 파일 저장
            file_ext = file.filename.split(".")[-1]
            temp_file_path = f"temp_{uuid.uuid4()}.{file_ext}"
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # 비디오 분석 시도 (실패 시 오디오 파일로 간주)
            is_video = False
            try:
                cap = cv2.VideoCapture(temp_file_path)
                if cap.isOpened():
                    # 프레임이 읽히는지 확인
                    ret, _ = cap.read()
                    if ret:
                        is_video = True
                cap.release()
            except:
                pass

            target_audio_path = temp_file_path

            if is_video:
                print(">>> [Step 1] 비디오 분석 시작...")
                try:
                    video_results = analyze_interview_video(temp_file_path)
                    response_data["video_analysis"] = video_results
                    print(">>> [Step 1] 비디오 분석 완료")
                    
                    # 오디오 추출
                    print(">>> [Step 2] 오디오 추출 시작...")
                    extracted = extract_audio_from_video(temp_file_path)
                    if extracted:
                        extracted_audio_path = extracted
                        target_audio_path = extracted
                        print(">>> [Step 2] 오디오 추출 완료")
                    else:
                        print(">>> [Step 2] 오디오 추출 실패 (파일이 없거나 코덱 문제)")
                except Exception as e:
                    print(f"FAILED: 비디오 분석 중 오류: {e}")

            # 오디오/음성 분석
            if os.path.exists(target_audio_path):
                print(">>> [Step 3] 오디오(STT/감정) 분석 시작...")
                try:
                    audio_array, _ = librosa.load(target_audio_path, sr=16000)
                    
                    def run_emotion_analysis():
                        print("   -> 감정 분석 중...")
                        results = emotion_pipe(target_audio_path)
                        emotion_probs = {r['label']: float(r['score']) for r in results}
                        top_emotion = results[0]['label'] if results else 'unknown'
                        return top_emotion, emotion_probs

                    with ThreadPoolExecutor() as executor:
                        print("   -> STT 변환 중...")
                        future_stt = executor.submit(lambda: stt_pipe(audio_array)["text"])
                        future_emotion = executor.submit(run_emotion_analysis)
                        
                        response_data["stt_result"] = future_stt.result()
                        print("   -> STT 완료")
                        top, probs = future_emotion.result()
                        print("   -> 감정 분석 완료")
                        response_data["audio_emotion"] = {"top_emotion": top, "probabilities": probs}
                    print(">>> [Step 3] 오디오 분석 종합 완료")

                except Exception as e:
                    print(f"FAILED: 오디오 분석 실패: {e}")
                    if not response_data["stt_result"] and not is_video:
                         response_data["stt_result"] = "(음성 분석 실패)"

        else:
            response_data["stt_result"] = text_answer

        # LLM 피드백
        print(">>> [Step 4] LLM 피드백 생성 중...")
        top_emo = "텍스트 모드"
        if response_data["audio_emotion"]:
            top_emo = response_data["audio_emotion"]["top_emotion"]
        elif response_data["video_analysis"] and "emotion" in response_data["video_analysis"]:
             # 비디오 감정이 있으면 그것도 참고 가능하지만 여기선 음성 감정 우선하거나 병기
             dominant = response_data["video_analysis"]["emotion"].get("dominant_emotion", "")
             top_emo = f"영상표정:{dominant}"

        prompt_feedback = f"""당신은 면접관입니다.
[질문]: {question}
[지원자 답변]: {response_data["stt_result"]}
[감지된 감정/태도]: {top_emo}

지원자의 답변 내용과 태도(감정 상태)에 대해 평가하고, 개선할 점을 구체적으로 조언해주세요."""

        messages_f = [{"role": "user", "content": prompt_feedback}]
        input_f = tokenizer.apply_chat_template(messages_f, tokenize=False, add_generation_prompt=True)
        inputs_f = tokenizer([input_f], return_tensors="pt").to(llm_model.device)
        generated_ids_f = llm_model.generate(inputs_f.input_ids, max_new_tokens=400)
        response_data["feedback"] = tokenizer.batch_decode(generated_ids_f, skip_special_tokens=True)[0].split("assistant\n")[-1].strip()
        print(">>> [Step 4] LLM 피드백 완료")

        # 모범 답안
        print(">>> [Step 5] 모범 답안 생성 중...")
        prompt_answer = f"""당신은 면접관입니다.
[질문]: {question}
이 질문에 대해 지원자가 할 수 있는 가장 이상적이고 논리적인 '만점짜리 모범 답변'을 스크립트 형태로 작성해주세요."""
        
        messages_a = [{"role": "user", "content": prompt_answer}]
        input_a = tokenizer.apply_chat_template(messages_a, tokenize=False, add_generation_prompt=True)
        inputs_a = tokenizer([input_a], return_tensors="pt").to(llm_model.device)
        generated_ids_a = llm_model.generate(inputs_a.input_ids, max_new_tokens=400)
        response_data["best_answer"] = tokenizer.batch_decode(generated_ids_a, skip_special_tokens=True)[0].split("assistant\n")[-1].strip()
        print(">>> [Step 5] 완료")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # 파일 정리
        if temp_file_path and os.path.exists(temp_file_path):
            try: os.remove(temp_file_path)
            except: pass
        if extracted_audio_path and os.path.exists(extracted_audio_path):
             try: os.remove(extracted_audio_path)
             except: pass

    return response_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
