# -*- coding: utf-8 -*-
"""
迪士尼RAG助手 - Web API 服务

提供前端聊天界面所需的 REST 接口
"""
import os
import importlib.util
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
IMAGE_DIR = os.path.join(BASE_DIR, "disney_knowledge_base", "images")

# 避免本地代理干扰 DashScope API 调用
_no_proxy = os.environ.get("NO_PROXY", "")
if "dashscope.aliyuncs.com" not in _no_proxy:
    os.environ["NO_PROXY"] = (_no_proxy + ",dashscope.aliyuncs.com").lstrip(",")

_query_path = os.path.join(BASE_DIR, "5-disney_query.py")
_spec = importlib.util.spec_from_file_location("disney_query", _query_path)
query_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(query_module)

load_index = query_module.load_index
rag_ask_structured = query_module.rag_ask_structured
get_dashscope_api_key = query_module.get_dashscope_api_key
is_valid_api_key = query_module.is_valid_api_key

app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app)

index = None
metadata = None


def ensure_index_loaded():
    """懒加载 FAISS 索引"""
    global index, metadata
    if index is None or metadata is None:
        index, metadata = load_index()


@app.route("/")
def index_page():
    """返回聊天主页"""
    return send_file(os.path.join(STATIC_DIR, "disney_chat.html"))


@app.route("/api/health")
def health():
    """健康检查"""
    api_key = get_dashscope_api_key()
    return jsonify({
        "status": "ok",
        "api_key_configured": is_valid_api_key(api_key),
        "api_key_prefix": api_key[:7] + "..." if len(api_key) > 7 else ""
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    """处理用户提问"""
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "请输入问题"}), 400

    api_key = get_dashscope_api_key()
    if not is_valid_api_key(api_key):
        return jsonify({
            "error": (
                "未读取到有效的 API Key。"
                "代码读取的变量名是 DASHSCOPE_API_KEY（全大写+下划线），"
                "不是 dashscope-api-key。"
                "请在启动服务的同一终端执行: export DASHSCOPE_API_KEY=sk-你的真实密钥"
            )
        }), 500

    try:
        ensure_index_loaded()
        result = rag_ask_structured(query, index, metadata, k=3)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"处理失败: {str(e)}"}), 500


@app.route("/media/images/<path:filename>")
def serve_image(filename):
    """提供知识库图片访问"""
    return send_from_directory(IMAGE_DIR, filename)


if __name__ == "__main__":
    api_key = get_dashscope_api_key()
    if not is_valid_api_key(api_key):
        print("警告: 未读取到有效的 DASHSCOPE_API_KEY")
        print("变量名必须是 DASHSCOPE_API_KEY，不能是 dashscope-api-key")
        print("请在同一终端执行: export DASHSCOPE_API_KEY=sk-你的真实密钥")
    else:
        print(f"API Key 已加载: {api_key[:7]}...")
    ensure_index_loaded()
    port = int(os.getenv("PORT", 5001))
    print(f"服务地址: http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
