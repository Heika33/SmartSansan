import os
import cv2
import asyncio
import json
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.websockets import WebSocketState
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import uvicorn
from multi_modal_analyzer import MultiModalAnalyzer
from config import VideoConfig, ServerConfig, ARCHIVE_DIR, RAGConfig
from utility import (
    chat_request,
    load_history_records,
    filter_history_records,
    summarize_records,
    format_match_payload,
    build_context_from_records,
    validate_records_with_llm,
)
from core.query_parser import parse_question, describe_filters, has_monitor_context
from core.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(title="智能视频监控预警系统")
app.mount("/static", StaticFiles(directory="static"), name="static")
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
        logger.info("[初始化] 视频源 FPS = %.2f", self._fps)

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
        logger.info("[开始监控] 启动视频帧读取...")
        async for frame in self.frame_generator():
            asyncio.create_task(archiver.write_frame(frame))
            count += 1

            if (datetime.now().timestamp() - self.last_analysis) >= VideoConfig.ANALYSIS_INTERVAL:
                logger.info("[触发分析] 帧数：%s", count)
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
                logger.info("[分析完成] 结果：%s", result['alert'])
                if result["alert"] != "无异常":
                    await AlertService.notify(result)
        except Exception as e:
            logger.error("[分析失败] %s", e, exc_info=True)

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
        logger.info("[视频推流] WebSocket 已连接")
        try:
            while system_state["status"] == "running":
                frame = await processor.frame_queue.get()
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), VideoConfig.JPEG_QUALITY])
                await websocket.send_bytes(buffer.tobytes())
        except WebSocketDisconnect:
            logger.info("[WebSocket] 客户端断开")
        finally:
            processor.start_push_queue = 0
            processor.frame_queue = asyncio.Queue()

def _should_use_monitor_mode(parsed: Dict[str, Any]) -> bool:
    filters = parsed.get("filters") or {}
    keywords = parsed.get("keywords") or []
    time_range = parsed.get("time_range")

    has_filter = any(filters.values())
    has_keywords = bool(keywords)
    has_time = bool(time_range)
    intent = parsed.get("intent")

    monitor_context = has_monitor_context(parsed.get("raw_query", ""))

    if intent == "exists":
        reason = "intent_exists"
    elif has_filter or has_keywords:
        reason = "filters"
    elif has_time:
        reason = "time_range"
    elif monitor_context:
        reason = "monitor_context"
    else:
        reason = None

    parsed["monitor_reason"] = reason
    return reason is not None


@app.get("/ask_question")
async def ask_question(query: str):
    parsed = parse_question(query)

    intent = parsed.get("intent", "describe")
    intro_mode = parsed.get("intro_mode")

    if intro_mode:
        logger.info("[QA] 系统介绍问题: %s", query)
        intro_answer = (
            "我是智能视频监控预警系统助手，可以帮助你：\n"
            "1. 选择摄像头、RTSP 流或上传视频进行监控。\n"
            "2. 实时查看画面、接收异常警报，并查询历史记录。\n"
            "3. 通过自然语言提问了解特定事件发生次数、时间或对象。\n"
            "4. 检索历史描述，协助分析异常、生成简报。\n"
            "若需更多帮助，可描述你想了解的功能。"
        )
        return {
            "mode": "monitor",
            "intent": "describe",
            "answer": intro_answer,
            "matches": [],
            "stats": {},
            "sources": [],
            "note": "系统功能说明",
        }

    if not _should_use_monitor_mode(parsed):
        logger.info("[QA] 通用模式问题: %s", query)
        general_prompt = f"""
你是一个友好的助手，请直接回答用户问题，避免提及你使用的模型或无法访问实时数据的事实。回答时聚焦于监控系统相关的知识或常规生活常识，不要讨论实时新闻。
用户问题：{query}
"""
        try:
            answer = await chat_request(general_prompt)
            return {
                "mode": "general",
                "answer": answer.strip(),
                "matches": [],
                "stats": {},
                "sources": [],
                "note": None,
                "intent": "general",
            }
        except Exception as exc:
            logger.error("通用问答失败: %s", exc, exc_info=True)
            return {"error": "问答服务暂时不可用，请稍后再试。"}

    history_records = load_history_records(limit=500)
    matched_records = filter_history_records(
        history_records,
        filters=parsed.get("filters"),
        time_range=parsed.get("time_range"),
        keywords=parsed.get("keywords"),
    )
    validated_records = await validate_records_with_llm(query, matched_records, max_items=25)

    stats = summarize_records(validated_records, intent) if intent in {"count", "exists"} else {}

    if not validated_records:
        note = "历史记录中没有与该问题相关的内容。"
        if parsed.get("time_range"):
            note = "指定时间范围内没有找到任何监控记录。"
        logger.info("[QA] 监控模式无匹配: %s", query)
        return {
            "mode": "monitor",
            "answer": note,
            "matches": [],
            "stats": {},
            "sources": [],
            "note": note,
            "intent": intent,
        }

    context_segments = build_context_from_records(query, validated_records, top_k=6)
    context = "\n".join(f"- {segment}" for segment in context_segments if segment)
    filter_summary = describe_filters(parsed.get("filters", {}), parsed.get("time_range"))

    prompt = f"""
你是智能视频监控系统的分析助手，需要结合历史监控记录回答问题。请勿提及你所使用的模型或无法访问实时数据等内容，只依据提供的摘要给出事实性回答。

[过滤条件]
{filter_summary}

[相关监控摘要]
{context}

[用户问题]
{query}

请用简洁、明确的中文回答，必要时给出出现次数、时间点或关联对象。若无法从给定内容中得出答案，请坦诚说明，并引用提供的摘要内容。
"""

    try:
        answer = await chat_request(prompt)
        return {
            "mode": "monitor",
            "answer": answer.strip(),
            "matches": format_match_payload(validated_records),
            "stats": stats,
            "sources": context_segments,
            "note": None,
            "intent": intent,
        }
    except Exception as exc:
        logger.error("问答服务调用失败: %s", exc, exc_info=True)
        return {"error": "问答服务暂时不可用，请稍后再试。"}


if __name__ == "__main__":
    uvicorn.run(
        app="video_server:app",
        host=ServerConfig.HOST,
        port=ServerConfig.PORT,
        reload=ServerConfig.RELOAD
    )
