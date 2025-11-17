import base64
import asyncio
import requests
import cv2
import time
import numpy as np
import json
import httpx
from config import APIConfig, RAGConfig, VideoConfig
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer  # 导入 SentenceTransformer

# 创建 Qdrant 客户端
client = QdrantClient(url=RAGConfig.QDRANT_HOST)

# 初始化 SentenceTransformer 模型
model = SentenceTransformer('all-MiniLM-L6-v2')

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
            print(f"[chat_request] 第 {attempt + 1} 次调用失败，将在 {wait_time}s 后重试。错误信息: {e}")
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
        print(f"Collection creation failed: {e}")

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


