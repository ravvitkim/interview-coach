import { useState, useRef, useEffect } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import './App.css';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const API_BASE = "http://localhost:8000";

interface VideoAnalysis {
  emotion?: {
    average_emotions?: Record<string, number>;
    dominant_emotion?: string;
    feedback?: string;
  };
  gaze?: {
    center_gaze_ratio?: number;
    feedback?: string;
  };
  posture?: {
    posture_stability?: number;
    posture_feedback?: string;
    gesture_feedback?: string;
  };
  overall?: {
    score: number;
    feedback: string;
  };
}

interface AnalysisResult {
  stt_result: string;
  audio_emotion?: {
    top_emotion: string;
    probabilities: Record<string, number>;
  };
  video_analysis?: VideoAnalysis;
  feedback: string;
  best_answer: string;
}

function App() {
  // States
  const [topic, setTopic] = useState('');
  const [difficulty, setDifficulty] = useState('초급');
  const [question, setQuestion] = useState('');
  const [textAnswer, setTextAnswer] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  // Recording states
  const [isRecording, setIsRecording] = useState(false);
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [hasRecording, setHasRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  // Refs
  const previewRef = useRef<HTMLVideoElement>(null);
  const recordedRef = useRef<HTMLVideoElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  // 질문 생성
  const generateQuestion = async () => {
    if (!topic) {
      alert("희망 직무를 입력해주세요!");
      return;
    }

    setIsGenerating(true);
    try {
      const res = await fetch(`${API_BASE}/generate_question`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topic, difficulty })
      });
      const data = await res.json();
      setQuestion(data.question);
    } catch (err) {
      alert("질문 생성 실패: " + (err as Error).message);
    } finally {
      setIsGenerating(false);
    }
  };

  // 카메라 시작
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      streamRef.current = stream;
      if (previewRef.current) {
        previewRef.current.srcObject = stream;
      }
      setIsCameraOn(true);
      setHasRecording(false);
    } catch (err) {
      alert("카메라/마이크 접근 권한이 필요합니다.");
    }
  };

  // 녹화 시작
  const startRecording = () => {
    if (!streamRef.current) return;

    chunksRef.current = [];
    const mediaRecorder = new MediaRecorder(streamRef.current);
    mediaRecorderRef.current = mediaRecorder;

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        chunksRef.current.push(event.data);
      }
    };

    mediaRecorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: 'video/webm' });
      setRecordedBlob(blob);
      if (recordedRef.current) {
        recordedRef.current.src = URL.createObjectURL(blob);
      }
      setHasRecording(true);
    };

    mediaRecorder.start();
    setIsRecording(true);
  };

  // 녹화 종료
  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
  };

  // 다시 녹화
  const resetRecording = () => {
    setRecordedBlob(null);
    setHasRecording(false);
    if (recordedRef.current) {
      recordedRef.current.src = '';
    }
  };

  // 파일 업로드
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setUploadedFile(file);
    }
  };

  // 분석 요청
  const analyzeInterview = async () => {
    if (!question) {
      alert("먼저 질문을 생성해주세요.");
      return;
    }

    const finalBlob = recordedBlob || uploadedFile;
    if (!finalBlob && !textAnswer) {
      alert("녹화, 파일 업로드, 또는 텍스트 답변 중 하나는 필수입니다.");
      return;
    }

    setIsAnalyzing(true);
    setResult(null);

    const formData = new FormData();
    formData.append("question", question);
    formData.append("difficulty", difficulty);
    if (textAnswer) formData.append("text_answer", textAnswer);
    if (finalBlob) {
      const ext = finalBlob.type.includes('mp4') ? 'mp4' : 'webm';
      formData.append("file", finalBlob, `recording.${ext}`);
    }

    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        body: formData
      });

      if (!res.ok) throw new Error("서버 에러 발생");

      const data: AnalysisResult = await res.json();
      setResult(data);
    } catch (err) {
      alert("분석 실패: " + (err as Error).message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // 차트 데이터 생성
  const getChartData = () => {
    let labels: string[] = [];
    let values: number[] = [];
    let labelName = "감정 분석";

    if (result?.video_analysis?.emotion?.average_emotions) {
      const emos = result.video_analysis.emotion.average_emotions;
      labels = Object.keys(emos);
      values = Object.values(emos);
      labelName = "영상 표정 (%)";
    } else if (result?.audio_emotion?.probabilities) {
      const emos = result.audio_emotion.probabilities;
      labels = Object.keys(emos);
      values = Object.values(emos).map(v => Number((v * 100).toFixed(1)));
      labelName = "음성 톤 (%)";
    }

    return {
      labels,
      datasets: [{
        label: labelName,
        data: values,
        backgroundColor: 'rgba(99, 102, 241, 0.6)',
        borderColor: 'rgba(99, 102, 241, 1)',
        borderWidth: 1,
        borderRadius: 4
      }]
    };
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: { beginAtZero: true, max: 100 }
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <h1 className="header-title">
            <i className="fas fa-robot"></i> AI Interview Coach
          </h1>
          <span className="header-subtitle">실시간 면접 코칭 시스템</span>
        </div>
      </header>

      <main className="main-content">
        <div className="grid-container">
          {/* Left Column */}
          <div className="left-column">
            {/* 면접 설정 */}
            <section className="card">
              <h2 className="section-title">1. 면접 설정</h2>
              <div className="form-group">
                <label className="label">희망 직무 / 주제</label>
                <input
                  type="text"
                  className="input"
                  placeholder="예: 데이터 분석가, 마케팅, 자기소개"
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                />
              </div>
              <div className="form-group">
                <label className="label">질문 난이도</label>
                <select
                  className="input"
                  value={difficulty}
                  onChange={(e) => setDifficulty(e.target.value)}
                >
                  <option value="초급">초급 (30초 이내)</option>
                  <option value="중급">중급 (60초 이내)</option>
                  <option value="고급">고급 (90초 이내)</option>
                </select>
              </div>
              <button
                className="btn btn-primary"
                onClick={generateQuestion}
                disabled={isGenerating}
              >
                {isGenerating ? (
                  <><i className="fas fa-spinner fa-spin"></i> 생성 중...</>
                ) : (
                  <><i className="fas fa-dice"></i> 면접 질문 생성하기</>
                )}
              </button>
            </section>

            {/* 답변 녹화 */}
            <section className="card">
              <h2 className="section-title">2. 답변 하기</h2>

              {/* 질문 표시 */}
              <div className="form-group">
                <label className="label">면접 질문</label>
                <div className="question-display">
                  {question || "질문을 먼저 생성해주세요."}
                </div>
              </div>

              {/* 비디오 영역 */}
              <div className="video-container">
                <video
                  ref={previewRef}
                  autoPlay
                  muted
                  className={`video ${!hasRecording && isCameraOn ? '' : 'hidden'}`}
                />
                <video
                  ref={recordedRef}
                  controls
                  className={`video ${hasRecording ? '' : 'hidden'}`}
                />
                {!isCameraOn && !hasRecording && (
                  <div className="video-placeholder">
                    <i className="fas fa-video"></i>
                    <p>카메라를 켜주세요</p>
                  </div>
                )}
                {isRecording && (
                  <div className="recording-indicator">● Recording</div>
                )}
              </div>

              {/* 녹화 버튼들 */}
              <div className="button-group">
                {!isCameraOn && !hasRecording && (
                  <button className="btn btn-secondary" onClick={startCamera}>
                    <i className="fas fa-camera"></i> 카메라 켜기
                  </button>
                )}
                {isCameraOn && !isRecording && !hasRecording && (
                  <button className="btn btn-danger" onClick={startRecording}>
                    <i className="fas fa-circle"></i> 녹화 시작
                  </button>
                )}
                {isRecording && (
                  <button className="btn btn-dark" onClick={stopRecording}>
                    <i className="fas fa-stop"></i> 녹화 종료
                  </button>
                )}
                {hasRecording && (
                  <button className="btn btn-info" onClick={resetRecording}>
                    <i className="fas fa-redo"></i> 다시 하기
                  </button>
                )}
              </div>

              <div className="divider">
                <span>또는 파일 업로드</span>
              </div>

              <input
                type="file"
                accept="video/*,audio/*"
                className="file-input"
                onChange={handleFileChange}
              />

              <div className="form-group">
                <label className="label">답변 텍스트 (영상/음성 없을 시)</label>
                <textarea
                  className="textarea"
                  rows={2}
                  placeholder="녹화가 어렵다면 텍스트로 입력하세요."
                  value={textAnswer}
                  onChange={(e) => setTextAnswer(e.target.value)}
                />
              </div>

              <button
                className="btn btn-success btn-large"
                onClick={analyzeInterview}
                disabled={isAnalyzing}
              >
                {isAnalyzing ? (
                  <><i className="fas fa-spinner fa-spin"></i> 분석 중...</>
                ) : (
                  <><i className="fas fa-chart-line"></i> 분석 시작하기</>
                )}
              </button>
            </section>
          </div>

          {/* Right Column */}
          <div className="right-column">
            <section className="card result-card">
              <h2 className="section-title">3. 분석 결과 리포트</h2>

              {/* 로딩 */}
              {isAnalyzing && (
                <div className="loading-overlay">
                  <div className="loader"></div>
                  <p className="loading-text">AI가 면접 내용을 분석 중입니다...</p>
                  <p className="loading-subtext">영상 길이에 따라 시간이 걸릴 수 있습니다.</p>
                </div>
              )}

              {/* 결과 */}
              {result && !isAnalyzing && (
                <div className="result-content">
                  {/* STT 결과 */}
                  <div className="result-section">
                    <h3 className="result-label">
                      <i className="fas fa-comment-alt"></i> 답변 내용 (STT)
                    </h3>
                    <div className="stt-box">{result.stt_result || "(내용 없음)"}</div>
                  </div>

                  {/* 비디오 분석 */}
                  {result.video_analysis?.overall && (
                    <div className="video-analysis-section">
                      <div className="score-grid">
                        <div className="score-box">
                          <span className="score-label">종합 점수</span>
                          <span className="score-value">{result.video_analysis.overall.score}</span>
                          <span className="score-unit">점</span>
                        </div>
                        <div className="score-feedback">
                          <p>{result.video_analysis.overall.feedback}</p>
                        </div>
                      </div>

                      <div className="detail-grid">
                        <div className="detail-box">
                          <div className="detail-label">😃 표정</div>
                          <div className="detail-value">
                            {result.video_analysis.emotion?.dominant_emotion || '-'}
                          </div>
                          <div className="detail-feedback">
                            {result.video_analysis.emotion?.feedback}
                          </div>
                        </div>
                        <div className="detail-box">
                          <div className="detail-label">👁️ 시선</div>
                          <div className="detail-value">
                            {result.video_analysis.gaze?.center_gaze_ratio || 0}%
                          </div>
                          <div className="detail-feedback">
                            {result.video_analysis.gaze?.feedback}
                          </div>
                        </div>
                        <div className="detail-box">
                          <div className="detail-label">🧘 자세</div>
                          <div className="detail-value">
                            {result.video_analysis.posture?.posture_stability || 0}점
                          </div>
                          <div className="detail-feedback">
                            {result.video_analysis.posture?.posture_feedback}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* 감정 차트 */}
                  {(result.video_analysis?.emotion?.average_emotions || result.audio_emotion?.probabilities) && (
                    <div className="result-section">
                      <h3 className="result-label">
                        <i className="fas fa-heart"></i> 감정/태도 분석
                      </h3>
                      <div className="chart-container">
                        <Bar data={getChartData()} options={chartOptions} />
                      </div>
                    </div>
                  )}

                  {/* 피드백 */}
                  <div className="feedback-box feedback-yellow">
                    <h3 className="feedback-title">
                      <i className="fas fa-lightbulb"></i> AI 코치 피드백
                    </h3>
                    <p className="feedback-content">{result.feedback}</p>
                  </div>

                  {/* 모범 답안 */}
                  <div className="feedback-box feedback-green">
                    <h3 className="feedback-title">
                      <i className="fas fa-check-circle"></i> 추천 모범 답안
                    </h3>
                    <p className="feedback-content">{result.best_answer}</p>
                  </div>
                </div>
              )}

              {/* 초기 상태 */}
              {!result && !isAnalyzing && (
                <div className="empty-state">
                  <i className="fas fa-clipboard-list"></i>
                  <p>분석 결과가 여기에 표시됩니다.</p>
                </div>
              )}
            </section>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;