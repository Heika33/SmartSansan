import os
import time
import json
import cv2
import numpy as np
import datetime
import asyncio
from config import RAGConfig
from utility import video_chat_async_limit_frame, chat_request, insert_txt
from prompt import prompt_detect, prompt_summary, prompt_vieo
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from core.logging import get_logger


logger = get_logger(__name__)


class MultiModalAnalyzer:
    def __init__(self):
        self.message_queue = []
        self.time_step_story = []
        # 使用 SentenceTransformer 加载嵌入模型
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # 初始化 Qdrant 客户端
        self.client = QdrantClient(url=RAGConfig.QDRANT_HOST)

        # 如果集合不存在，才创建集合
        if not self.collection_exists(RAGConfig.QDRANT_COLLECTION_NAME):
            self.create_collection(RAGConfig.QDRANT_COLLECTION_NAME)

    def collection_exists(self, collection_name: str):
        """检查 Qdrant 集合是否存在"""
        try:
            collections = self.client.get_collections()
            return collection_name in collections
        except Exception as e:
            logger.error("检查集合是否存在失败: %s", e, exc_info=True)
            return False

    def create_collection(self, collection_name: str):
        """创建 Qdrant 集合"""
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "size": 384,  # MiniLM-L6-v2 生成 384 维向量
                    "distance": "Cosine"
                }
            )
            logger.info("集合 %s 创建成功", collection_name)
        except Exception as e:
            logger.error("Qdrant集合创建失败: %s", e, exc_info=True)

    def trans_date(self, date_str):
        year, month, day, hour, minute, second = date_str.split('-')
        am_pm = "上午" if int(hour) < 12 else "下午"
        hour_12 = hour if hour == '12' else str(int(hour) % 12)
        return f"{year}年{int(month)}月{int(day)}日{am_pm}{hour_12}点（{hour}时）{int(minute)}分{int(second)}秒"

    async def analyze(self, frames, fps=20, timestamps=None):
        start_time = time.time()
        histroy = "录像视频刚刚开始。"
        Recursive_summary = ""

        for i in self.message_queue:
            histroy = (
                    "历史视频内容总结:" + Recursive_summary +
                    "\n\n当前时间段：" + i['start_time'] + " - " + i['end_time'] +
                    "\n该时间段视频描述如下：" + i['description'] +
                    "\n\n该时间段异常提醒:" + i['is_alert']
            )

        try:
            results = await asyncio.gather(
                chat_request(prompt_summary.format(histroy=histroy)),
                video_chat_async_limit_frame(prompt_vieo, frames, timestamps, fps=fps)
            )
            Recursive_summary = results[0]
            description = results[1]
        except Exception as e:
            logger.error("[分析失败] 多模态 API 响应错误: %s", e, exc_info=True)
            return {"alert": "分析失败，请检查大模型API状态"}

        if not timestamps:
            return description

        date_flag = self.trans_date(timestamps[0]) + "："

        try:
            # 1. 将描述保存到 video_histroy_info.txt 文件
            with open(RAGConfig.HISTORY_FILE, 'a', encoding='utf-8') as file:
                file.write(date_flag + description + '\n')
            # 存储视频描述到 qdrant 或文件
            if RAGConfig.ENABLE_RAG:
                self.insert_description(date_flag + description)
            # else:
            #     # 写入到 video_histroy_info.txt 文件
            #     with open(RAGConfig.HISTORY_FILE, 'a', encoding='utf-8') as file:
            #         file.write(date_flag + description + '\n')
        except Exception as e:
            logger.error("[历史记录保存失败] %s", e, exc_info=True)

        text = prompt_detect.format(
            Recursive_summary=Recursive_summary,
            current_time=timestamps[0] + " - " + timestamps[-1],
            latest_description=description
        )

        try:
            alert = await chat_request(text)
        except Exception as e:
            logger.error("[异常检测失败] %s", e, exc_info=True)
            alert = "分析失败"

        logger.info("[分析完成] 耗时 %.2fs | 异常：%s", time.time() - start_time, alert)

        if "无异常" not in alert:
            current_time = timestamps[0]
            file_str = f"waring_{current_time}"
            new_file_name = f"video_warning/{file_str}.mp4"
            os.rename("./video_warning/output.mp4", new_file_name)

            frame = frames[0]
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(f"video_warning/{file_str}.jpg", frame)

            return {
                "alert": f"<span style=\"color:red;\">{alert}</span>",
                "description": f' 当前10秒监控消息描述：\n{description}\n\n 历史监控内容:\n{Recursive_summary}',
                "video_file_name": f"{file_str}.mp4",
                "picture_file_name": f"{file_str}.jpg"
            }

        self.message_queue.append({
            'start_time': timestamps[0],
            'end_time': timestamps[1],
            'description': description,
            'is_alert': alert
        })
        self.message_queue = self.message_queue[-15:]

        return {"alert": "无异常"}

    def insert_description(self, description: str):
        # 使用 SentenceTransformer 生成嵌入
        embedding = self.model.encode(description)
        # 确保生成的向量维度为 384，避免维度不匹配
        assert len(embedding) == 384, f"Expected vector dimension of 384, but got {len(embedding)}"

        # 将描述和向量插入 qdrant 数据库
        self.client.upsert(
            collection_name=RAGConfig.QDRANT_COLLECTION_NAME,
            points=[{
                "id": np.random.randint(100000),  # 随机 ID，可以根据需要修改
                "vector": embedding.tolist(),  # 转换为列表格式
                "payload": {"description": description}
            }]
        )

    def search_similar(self, query_description: str, top_k: int = 5):
        """根据查询描述在 Qdrant 中进行相似性搜索"""
        embedding = self.model.encode(query_description)

        results = self.client.search(
            collection_name=RAGConfig.QDRANT_COLLECTION_NAME,
            query_vector=embedding.tolist(),
            limit=top_k
        )

        return [result.payload['description'] for result in results]
