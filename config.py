from typing import Dict, Any
import logging

# 视频处理配置
class VideoConfig:
    VIDEO_INTERVAL = 1800  # 视频分段时长(秒)
    ANALYSIS_INTERVAL = 5  # 分析间隔(秒)
    BUFFER_DURATION = 11  # 滑窗分析时长（秒）
    WS_RETRY_INTERVAL = 3  # WebSocket重连间隔(秒)
    MAX_WS_QUEUE = 100  # 消息队列最大容量
    JPEG_QUALITY = 70  # JPEG压缩质量

class APIConfig:
    # 通义千问API配置
    QWEN_API_KEY = "sk-299d7e485d134cd2a0893b03945bd961"
    QWEN_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    QWEN_MODEL = "qwen-vl-max-2025-01-25"

    # Moonshot语言模型 API配置
    MOONSHOT_API_KEY = "sk-Pc8cbpbVTYTCu91icQfMvhRmrjJNEBrluAC4iFe2Jpd09YgQ"
    # MOONSHOT_API_KEY = "sk-IoAFdeEvjKKFgbLsPm8smtK04Qj57a5ThXrESAwvOllRtREM"
    MOONSHOT_API_URL = "https://api.moonshot.cn/v1/chat/completions"
    MOONSHOT_MODEL = "moonshot-v1-32k"

    # API请求配置
    REQUEST_TIMEOUT = 60.0 # 请求超时时间（秒）
    TEMPERATURE = 0.5 # 温度
    TOP_P = 0.01
    TOP_K = 20
    REPETITION_PENALTY = 1.05


# RAG系统配置
class RAGConfig:
    ENABLE_RAG = True
    HISTORY_FILE = "video_histroy_info.txt"
    QDRANT_HOST = "http://localhost:6333"  # qdrant 服务的地址
    QDRANT_COLLECTION_NAME = "video_descriptions"  # 向量数据存储的集合名称
    VECTOR_API_URL = "http://localhost:8085/add_text/"  # 用于插入文本的API接口地址


# 存档配置
ARCHIVE_DIR = "archive"

# 服务器配置
class ServerConfig:
    HOST = "0.0.0.0"
    PORT = 16532
    RELOAD = True
    WORKERS = 1


# 日志配置
LOG_CONFIG = {
    'level': logging.INFO,
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'handlers': [
        {'type': 'file', 'filename': 'code.log'},
        {'type': 'stream'}
    ]
}


def update_config(args: Dict[str, Any]) -> None:
    """使用命令行参数更新配置"""
    # 视频处理配置更新
    for key in ['video_interval', 'analysis_interval', 'buffer_duration',
                'ws_retry_interval', 'max_ws_queue', 'jpeg_quality']:
        if key in args:
            setattr(VideoConfig, key.upper(), args[key])

    # 服务器配置
    for key in ['host', 'port', 'reload', 'workers']:
        if key in args:
            setattr(ServerConfig, key.upper(), args[key])

    # API配置更新
    for key in ['qwen_api_key', 'qwen_api_url', 'qwen_model',
                'moonshot_api_key', 'moonshot_api_url', 'moonshot_model',
                'request_timeout', 'temperature', 'top_p', 'top_k',
                'repetition_penalty']:
        if key in args:
            setattr(APIConfig, key.upper(), args[key])

    # RAG配置更新
    for key in ['enable_rag', 'vector_api_url', 'history_file']:
        if key in args:
            setattr(RAGConfig, key.upper(), args[key])


