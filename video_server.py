import os
import cv2
import asyncio
import json
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.websockets import WebSocketState
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import uvicorn
from multi_modal_analyzer import MultiModalAnalyzer
from config import VideoConfig, ServerConfig, LOG_CONFIG, ARCHIVE_DIR, RAGConfig
from utility import search_similar, client, chat_request, model
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=LOG_CONFIG['level'],
    format=LOG_CONFIG['format'],
    handlers=[
        logging.FileHandler(LOG_CONFIG['handlers'][0]['filename'], encoding='utf-8'),
        logging.StreamHandler()
    ]
)

app = FastAPI(title="智能视频监控预警系统")
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

system_state = {"status": "idle"}
processor: Optional["VideoProcessor"] = None
archiver: Optional["VideoArchiver"] = None
class MonitorRequest(BaseModel):
    source_type: str
    video_path: Optional[str] = ""

class VideoProcessor:
    def __init__(self, video_source):
        if isinstance(video_source, str) and video_source.isdigit():
            video_source = int(video_source)
        self.video_source = video_source
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {video_source}")

        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self._fps <= 1.0 or self._fps > 120:
            self._fps = 25.0
        print(f"[初始化] 视频源 FPS = {self._fps}")

        self.buffer = deque(maxlen=int(self._fps * VideoConfig.BUFFER_DURATION))
        self.executor = ThreadPoolExecutor()
        self.analyzer = MultiModalAnalyzer()
        self.last_analysis = datetime.now().timestamp()
        self._running = True
        self.lock = asyncio.Lock()
        self.frame_queue = asyncio.Queue()
        self.start_push_queue = 0

    @property
    def fps(self) -> float:
        return self._fps

    async def frame_generator(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                await asyncio.sleep(1)
                continue
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            self.buffer.append({
                "frame": frame,
                "timestamp": datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            })

            if self.start_push_queue:
                await self.frame_queue.put(frame)

            yield frame
            await asyncio.sleep(max(0, 1 / self._fps))
    async def start_processing(self):
        count = 0
        print("[开始监控] 启动视频帧读取...")
        async for frame in self.frame_generator():
            asyncio.create_task(archiver.write_frame(frame))
            count += 1

            if (datetime.now().timestamp() - self.last_analysis) >= VideoConfig.ANALYSIS_INTERVAL:
                print(f"[触发分析] 帧数：{count}")
                asyncio.create_task(self.trigger_analysis())
                self.last_analysis = datetime.now().timestamp()
                count = 0

    async def trigger_analysis(self):
        try:
            async with self.lock:
                clip = list(self.buffer)
                if not clip:
                    return
                result = await self.analyzer.analyze(
                    [f["frame"] for f in clip],
                    self.fps,
                    (clip[0]['timestamp'], clip[-1]['timestamp'])
                )
                print(f"[分析完成] 结果：{result['alert']}")
                if result["alert"] != "无异常":
                    await AlertService.notify(result)
        except Exception as e:
            logging.error(f"[分析失败] {str(e)}")

class VideoArchiver:
    def __init__(self):
        self.current_writer: Optional[cv2.VideoWriter] = None
        self.last_split = datetime.now()

    async def write_frame(self, frame: np.ndarray):
        if self._should_split():
            self._create_new_file()
        if self.current_writer:
            self.current_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def _should_split(self):
        return (datetime.now() - self.last_split).total_seconds() >= VideoConfig.VIDEO_INTERVAL

    def _create_new_file(self):
        if self.current_writer:
            self.current_writer.release()
        filename = f"{ARCHIVE_DIR}/{datetime.now().strftime('%Y%m%d_%H%M')}.mp4"
        self.current_writer = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*'avc1'),
            25,
            (640, 360)
        )
        self.last_split = datetime.now()
class AlertService:
    _connections = set()

    @classmethod
    async def register(cls, websocket: WebSocket):
        await websocket.accept()
        cls._connections.add(websocket)

    @classmethod
    async def notify(cls, data: Dict):
        message = json.dumps({
            "timestamp": datetime.now().isoformat(),
            **data
        })
        for conn in list(cls._connections):
            try:
                if conn.client_state == WebSocketState.CONNECTED:
                    await conn.send_text(message)
                else:
                    cls._connections.remove(conn)
            except Exception:
                cls._connections.remove(conn)

@app.get("/")
def serve_index():
    return FileResponse("index.html")

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as f:
        content = await file.read()
        f.write(content)
    return {"success": True, "path": file_location}

@app.post("/start_monitoring")
async def start_monitoring(data: MonitorRequest):
    global processor, archiver, system_state

    if system_state["status"] == "running":
        return {"message": "系统已在运行中"}

    video_source = data.video_path if data.source_type != "device" else int(data.video_path)
    try:
        processor = VideoProcessor(video_source)
    except Exception as e:
        return {"error": str(e)}, 500

    archiver = VideoArchiver()
    asyncio.create_task(processor.start_processing())
    system_state["status"] = "running"
    return {"message": "监控已启动"}

@app.post("/pause_monitoring")
async def pause_monitoring():
    system_state["status"] = "paused"
    return {"message": "已暂停"}

@app.post("/stop_monitoring")
async def stop_monitoring():
    global processor, archiver, system_state
    system_state["status"] = "stopped"
    if processor:
        processor._running = False
        processor.cap.release()
        processor = None
    return {"message": "已停止"}

@app.get("/status")
async def get_status():
    return {"status": system_state["status"]}
@app.get("/list_cameras")
async def list_cameras():
    devices = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            devices.append(f"本机摄像头 {i}")
            cap.release()
    return {"devices": devices}

@app.get("/get_history")
async def get_history(page: int = 1, size: int = 5):
    try:
        with open(RAGConfig.HISTORY_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        total = len(lines)
        start = (page - 1) * size
        end = start + size
        return {
            "total": total,
            "page": page,
            "size": size,
            "items": lines[start:end]
        }
    except FileNotFoundError:
        return {"total": 0, "page": page, "size": size, "items": []}

@app.websocket("/alerts")
async def alert_websocket(websocket: WebSocket):
    await AlertService.register(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        pass

@app.websocket("/video_feed")
async def video_feed(websocket: WebSocket):
    await websocket.accept()
    if processor:
        processor.start_push_queue = 1
        print("[视频推流] WebSocket 已连接")
        try:
            while system_state["status"] == "running":
                frame = await processor.frame_queue.get()
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), VideoConfig.JPEG_QUALITY])
                await websocket.send_bytes(buffer.tobytes())
        except WebSocketDisconnect:
            print("[WebSocket] 客户端断开")
        finally:
            processor.start_push_queue = 0
            processor.frame_queue = asyncio.Queue()

@app.get("/search_similar")
async def search_similar(query: str, top_k: int = 5):
    # 使用 SentenceTransformer 生成查询嵌入
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query).tolist()  # 生成查询的嵌入向量

    # 从 Qdrant 中查询相似的描述
    results = client.search(
        collection_name=RAGConfig.QDRANT_COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k
    )

    # 返回相似的描述
    return {"results": [result.payload['description'] for result in results]}


@app.get("/ask_question")
async def ask_question(query: str):
    # 将用户问题转化为查询嵌入向量
    query_embedding = model.encode(query).tolist()

    # 从 Qdrant 中检索相关的历史描述
    results = client.search(
        collection_name=RAGConfig.QDRANT_COLLECTION_NAME,
        query_vector=query_embedding,
        limit=500  # 返回与问题最相关的5条历史记录
    )

    # 将检索到的历史记录进行整合，生成回答
    relevant_descriptions = [result.payload['description'] for result in results]
    context = "\n".join(relevant_descriptions)

    # 基于检索到的上下文生成回答
    prompt = f"""
    你是一个智能视频监控系统的虚拟助手，专门负责根据监控视频历史记录回答问题。你的任务是根据历史监控内容，结合用户的问题，提供简洁、精准的回答。

    以下是历史监控内容，描述了视频中的人物、行为、异常事件等：

    {context}

    用户的问题是：
    {query}

    请根据历史监控内容中的描述回答以下问题：
    - 确保你的回答简洁明确。
    - 如果视频内容中涉及到具体人物或异常事件，提及相关细节。
    - 如果问题无法从历史内容中得到答案，请明确说明。

    请提供清晰且信息充实的回答，帮助用户理解视频监控中的关键细节。
    """

    try:
        # 使用 Moonshot 或 Qwen 模型生成回答
        answer = await chat_request(prompt)
        return {"answer": answer}
    except Exception as e:
        return {"error": f"问题处理失败：{str(e)}"}


if __name__ == "__main__":
    uvicorn.run(
        app="video_server:app",
        host=ServerConfig.HOST,
        port=ServerConfig.PORT,
        reload=ServerConfig.RELOAD
    )
