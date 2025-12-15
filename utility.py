import base64
import asyncio
import requests
import cv2
import time
import numpy as np
import json
import httpx
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
from config import APIConfig, RAGConfig, VideoConfig
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer  # 导入 SentenceTransformer
from core.logging import get_logger

# 创建 Qdrant 客户端
client = QdrantClient(url=RAGConfig.QDRANT_HOST)

# 初始化 SentenceTransformer 模型
model = SentenceTransformer('all-MiniLM-L6-v2')
logger = get_logger(__name__)

def frames_to_base64(frames, fps, timestamps):
    width = frames[0].shape[1]
    height = frames[0].shape[0]
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_writer = cv2.VideoWriter('./video_warning/output.mp4', fourcc, fps, (width, height))

    for frame in frames:
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        video_writer.write(frame)
    video_writer.release()

    with open('./video_warning/output.mp4', 'rb') as video_file:
        video_base64 = base64.b64encode(video_file.read()).decode('utf-8')
    return video_base64

async def video_chat_async_limit_frame(text, frames, timestamps, fps=20):
    video_base64 = frames_to_base64(frames, fps, timestamps)
    url = APIConfig.QWEN_API_URL
    headers = {
        "Content-Type": "application/json",
        "authorization": APIConfig.QWEN_API_KEY
    }
    model = APIConfig.QWEN_MODEL

    data_image = []
    frame_count = int(VideoConfig.BUFFER_DURATION)
    for i in range(frame_count):
        frame = frames[(len(frames)//frame_count)*i]
        image_path = 'output_frame.jpg'
        cv2.imwrite(image_path, frame)
        with open(image_path, 'rb') as file:
            image_base64 = "data:image/jpeg;base64," + base64.b64encode(file.read()).decode('utf-8')
        data_image.append(image_base64)

    content = [{"type": "text", "text": text}] + [{"type": "image_url", "image_url": {"url": i}} for i in data_image]
    data = {
        "model": model,
        "vl_high_resolution_images": False,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        response = await client.post(url, headers=headers, json=data)
        response_data = response.json()
        return response_data['choices'][0]['message']['content']

async def video_chat_async(text, frames, timestamps, fps=20):
    video_base64 = frames_to_base64(frames, fps, timestamps)
    url = APIConfig.QWEN_API_URL
    headers = {
        "Content-Type": "application/json",
        "authorization": APIConfig.QWEN_API_KEY
    }
    model = APIConfig.QWEN_MODEL

    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": f"data:video/mp4;base64,{video_base64}"
                        }
                    }
                ]
            }
        ],
        "stop_token_ids": [151645, 151643]
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(APIConfig.REQUEST_TIMEOUT)) as client:
        response = await client.post(url, headers=headers, json=data)
        response_data = response.json()
        return response_data['choices'][0]['message']['content']

async def chat_request(message, stream=False):
    url = APIConfig.MOONSHOT_API_URL
    model = APIConfig.MOONSHOT_MODEL

    messages = [{"role": "user", "content": message}]
    headers = {
        "content-Type": "application/json",
        "authorization": APIConfig.MOONSHOT_API_KEY
    }
    data = {
        "messages": messages,
        "model": model,
        "repetition_penalty": APIConfig.REPETITION_PENALTY,
        "temperature": APIConfig.TEMPERATURE,
        "top_p": APIConfig.TOP_P,
        "top_k": APIConfig.TOP_K,
        "stream": stream
    }

    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(APIConfig.REQUEST_TIMEOUT)) as client:
                response = await client.post(url, headers=headers, json=data)
                response_data = response.json()

                if "choices" in response_data:
                    return response_data['choices'][0]['message']['content']
                elif "error" in response_data:
                    code = response_data["error"].get("code", "unknown")
                    message = response_data["error"].get("message", "API返回错误")
                    raise RuntimeError(f"Moonshot API 错误 [{code}]: {message}")
                else:
                    raise RuntimeError(f"Moonshot API 无效响应结构: {response_data}")

        except Exception as e:
            wait_time = 2 ** attempt
            logger.warning(
                "[chat_request] 第 %s 次调用失败，将在 %ss 后重试。错误信息: %s",
                attempt + 1,
                wait_time,
                e,
                exc_info=True,
            )
            await asyncio.sleep(wait_time)

    raise RuntimeError("Moonshot API 多次调用失败，请检查配额或稍后再试")

def insert_txt(docs, table_name):
    url = RAGConfig.VECTOR_API_URL
    data = {
        "docs": docs,
        "table_name": table_name
    }
    response = requests.post(url, json=data)
    return response.json()

def create_collection():
    try:
        client.create_collection(
            collection_name=RAGConfig.COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # 使用 MiniLM-L6-v2 生成 384 维向量
        )
    except Exception as e:
        logger.error("Collection creation failed: %s", e, exc_info=True)

# 向 qdrant 插入描述向量
def insert_description(description: str):
    # 使用 SentenceTransformer 生成嵌入
    embedding = model.encode(description)

    # 将描述和向量插入 qdrant 数据库
    client.upsert(
        collection_name=RAGConfig.QDRANT_COLLECTION_NAME,
        points=[{
            "id": np.random.randint(100000),
            "vector": embedding.tolist(),
            "payload": {"description": description}
        }]
    )

# 查询相似的描述
def search_similar(query_description: str, top_k: int = 5):
    embedding = model.encode(query_description)

    results = client.search(
        collection_name=RAGConfig.COLLECTION_NAME,
        query_vector=embedding.tolist(),
        limit=top_k
    )

    return [result.payload['description'] for result in results]


def load_history_records(limit: int = 500) -> List[Dict[str, Optional[str]]]:
    try:
        with open(RAGConfig.HISTORY_FILE, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        return []

    if limit:
        lines = lines[-limit:]

    records = []
    for line in lines:
        timestamp = _extract_timestamp(line)
        records.append({"timestamp": timestamp, "description": line})
    return deduplicate_records(records)


def filter_history_records(
    records: List[Dict[str, Optional[str]]],
    filters: Optional[Dict[str, List[str]]] = None,
    time_range: Optional[Dict[str, str]] = None,
    keywords: Optional[List[str]] = None,
) -> List[Dict[str, Optional[str]]]:
    filtered = []
    start = _parse_iso_time(time_range.get("start")) if time_range else None
    end = _parse_iso_time(time_range.get("end")) if time_range else None

    required_tokens = []
    if filters:
        for values in filters.values():
            required_tokens.extend(values)
    if keywords:
        required_tokens.extend(keywords)
    required_tokens = list(dict.fromkeys(required_tokens))

    for record in records:
        text = record["description"]
        ts_str = record["timestamp"]
        ts = _parse_guess_time(ts_str) if ts_str else None

        if (start or end) and not ts:
            # 没有时间戳的记录无法确认是否在范围内，直接丢弃
            continue

        if start and ts and ts < start:
            continue
        if end and ts and ts > end:
            continue

        if required_tokens and not all(token in text for token in required_tokens):
            continue

        filtered.append(record)
    return filtered


def summarize_records(records: List[Dict[str, Optional[str]]], intent: str) -> Dict[str, object]:
    if not records:
        return {}

    stats: Dict[str, object] = {"count": len(records)}
    timestamps = []
    for record in records:
        if not record.get("timestamp"):
            continue
        parsed = _parse_guess_time(record["timestamp"])
        if parsed:
            timestamps.append(parsed)
    if timestamps:
        timestamps.sort()
        stats["first_seen"] = timestamps[0].strftime("%Y-%m-%d %H:%M:%S")
        stats["last_seen"] = timestamps[-1].strftime("%Y-%m-%d %H:%M:%S")
    if intent == "exists" and records:
        stats["exists"] = "是"
    elif intent == "exists":
        stats["exists"] = "否"
    return stats


def format_match_payload(records: List[Dict[str, Optional[str]]], limit: int = 10) -> List[Dict[str, Optional[str]]]:
    limited = records[:limit]
    payload = []
    for rec in limited:
        item = {
            "timestamp": rec.get("timestamp"),
            "description": rec["description"]
        }
        if rec.get("llm_reason"):
            item["reason"] = rec["llm_reason"]
        payload.append(item)
    return payload


def build_context_from_records(query: str, matches: List[Dict[str, Optional[str]]], top_k: int = 5) -> List[str]:
    if matches:
        return [record["description"] for record in matches[:top_k]]
    try:
        query_embedding = model.encode(query).tolist()
        results = client.search(
            collection_name=RAGConfig.QDRANT_COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k
        )
        return [result.payload.get("description", "") for result in results]
    except Exception as exc:
        logger.warning("向量检索失败: %s", exc)
        return []


def _extract_timestamp(text: str) -> Optional[str]:
    structured = _extract_structured_chinese_datetime(text)
    if structured:
        return structured.isoformat(sep=" ")

    patterns = [
        (r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", "%Y-%m-%d %H:%M:%S"),
        (r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", "%Y-%m-%d-%H-%M-%S"),
        (r"\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}", "%Y/%m/%d %H:%M:%S"),
    ]
    for regex, fmt in patterns:
        match = re.search(regex, text)
        if match:
            value = match.group(0)
            try:
                datetime.strptime(value, fmt)
                return value
            except ValueError:
                continue
    return None


def _parse_guess_time(value: str) -> Optional[datetime]:
    if not value:
        return None

    structured = _extract_structured_chinese_datetime(value)
    if structured:
        return structured

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d-%H-%M-%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _extract_structured_chinese_datetime(text: str) -> Optional[datetime]:
    """
    解析诸如“2025年4月16日上午9点53分2秒”或“2025年4月16日 09时53分”等中文日期。
    """
    regex = (
        r"(?P<year>\d{4})年(?P<month>\d{1,2})月(?P<day>\d{1,2})日?"
        r"(?:[^\d]*(?P<ampm>上午|下午|中午|凌晨|晚上|傍晚)?)?"
        r"(?:[^\d]*(?P<hour>\d{1,2})点)?"
        r"(?:[^\d]*(?P<minute>\d{1,2})分)?"
        r"(?:[^\d]*(?P<second>\d{1,2})秒)?"
    )
    match = re.search(regex, text)
    if not match:
        return None

    year = int(match.group("year"))
    month = int(match.group("month"))
    day = int(match.group("day"))
    hour = int(match.group("hour")) if match.group("hour") else 0
    minute = int(match.group("minute")) if match.group("minute") else 0
    second = int(match.group("second")) if match.group("second") else 0

    ampm = match.group("ampm")
    if ampm:
        if ampm in {"下午", "晚上", "傍晚"} and hour < 12:
            hour += 12
        if ampm == "中午" and hour < 12:
            hour = 12
        if ampm == "凌晨" and hour == 12:
            hour = 0

    try:
        return datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None


def deduplicate_records(records: List[Dict[str, Optional[str]]]) -> List[Dict[str, Optional[str]]]:
    seen: Set[str] = set()
    unique: List[Dict[str, Optional[str]]] = []
    for record in records:
        key = f"{record.get('timestamp')}-{hash(record['description'])}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


async def validate_records_with_llm(query: str, records: List[Dict[str, Optional[str]]], max_items: int = 20) -> List[Dict[str, Optional[str]]]:
    """Use LLM to confirm whether candidate records match the query semantics."""
    if not records:
        return []

    subset = records[:max_items]
    candidates_text = "\n".join(
        f"{idx}. [{rec.get('timestamp') or '未知时间'}] {rec['description']}"
        for idx, rec in enumerate(subset, start=1)
    )
    prompt = f"""
请你作为监控日志审核员，根据用户问题判断候选记录是否相关。
用户问题：{query}
候选记录：
{candidates_text}

请严格按照 JSON 输出，格式如下：
{{
  "matches": [
    {{"index": 1, "match": true, "reason": "涉及黑衣男子"}},
    ...
  ]
}}
只参考候选记录，不要编造额外内容；若不匹配请给出简单原因。
"""
    try:
        response = await chat_request(prompt)
        data = json.loads(response)
        match_map = {
            item["index"]: item for item in data.get("matches", []) if isinstance(item, dict)
        }
    except Exception as exc:
        logger.warning("LLM 匹配判定失败，返回空集: %s", exc)
        return []

    validated: List[Dict[str, Optional[str]]] = []
    for idx, record in enumerate(subset, start=1):
        item = match_map.get(idx)
        if not item or not item.get("match"):
            continue
        enriched = dict(record)
        if item.get("reason"):
            enriched["llm_reason"] = item["reason"]
        validated.append(enriched)
    return validated


def _parse_iso_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
