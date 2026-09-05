import os
from typing import Any
import requests

from utils.logger_handler import logger
from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path

rag=RagSummarizeService()

external_data={}

@tool(description='从向量存储中检索参考资料')
def rag_summarize(query:str)->str:
    return rag.rag_summarize(query)

@tool(description="根据用户提供的城市名称查询当前真实天气")
def get_weather(city: str) -> dict[str, Any]:
    city = city.strip()

    if not city:
        return {
            "status": "error",
            "message": "城市名称不能为空，请先询问用户所在城市。",
        }

    try:
        location_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": city,
                "count": 1,
                "language": "zh",
                "format": "json",
            },
            timeout=10,
        )
        location_response.raise_for_status()

        locations = location_response.json().get("results", [])
        if not locations:
            return {
                "status": "not_found",
                "message": f"没有找到城市：{city}",
            }

        location = locations[0]
        latitude = location["latitude"]
        longitude = location["longitude"]

        weather_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,relative_humidity_2m,precipitation,"
                    "weather_code,wind_speed_10m"
                ),
                "timezone": "auto",
            },
            timeout=10,
        )
        weather_response.raise_for_status()

        current = weather_response.json().get("current")
        if not current:
            return {
                "status": "error",
                "message": "天气服务没有返回当前天气。",
            }

        return {
            "status": "ok",
            "city": location["name"],
            "province": location.get("admin1", ""),
            "country": location.get("country", ""),
            "temperature_celsius": current.get("temperature_2m"),
            "relative_humidity_percent": current.get("relative_humidity_2m"),
            "precipitation_mm": current.get("precipitation"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "weather_code": current.get("weather_code"),
            "observation_time": current.get("time"),
        }

    except (requests.RequestException, ValueError, KeyError):
        logger.exception("查询城市 %s 的天气失败", city)
        return {
            "status": "error",
            "message": "天气服务暂时不可用，请稍后重试。",
        }

@tool(description="检查能否自动获取用户城市；本应用需要用户主动提供城市")
def get_user_location() -> str:
    return "无法自动获取用户城市，请询问用户所在城市。"

@tool(description="获取当前演示用户ID")
def get_user_id() -> str:
    return "1001"

@tool(description="获取演示报告月份，格式为YYYY-MM")
def get_current_month() -> str:
    return "2025-01"

def generate_external_data():
    if not external_data:
        external_data_path=get_abs_path(agent_conf['external_data_path'])

        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f'外部数据文件{external_data_path}不存在')

        with open(external_data_path,'r',encoding='utf-8') as f:
            for line in f.readlines()[1:]:
                arr:list[str]=line.strip().split(',')

                user_id:str=arr[0].replace('"','')
                feature:str=arr[1].replace('"','')
                efficiency:str=arr[2].replace('"','')
                consumables:str=arr[3].replace('"','')
                comparsion:str=arr[4].replace('"','')
                time:str=arr[5].replace('"','')

                if user_id not in external_data:
                    external_data[user_id]={}

                external_data[user_id][time]={
                    '特征':feature,
                    '效率':efficiency,
                    '耗材':consumables,
                    '对比':comparsion
                }

@tool(
    description=(
        "查询指定用户在指定月份的机器人使用记录。"
        "返回结果包含查询状态、用户ID、月份和使用数据。"
    )
)
def fetch_external_data(user_id: str, month: str) -> dict:
    generate_external_data()

    record = external_data.get(user_id, {}).get(month)

    if record is None:
        logger.warning(
            "[fetch_external_data]未检索到用户 %s 在 %s 的使用记录",
            user_id,
            month,
        )

        return {
            "status": "not_found",
            "message": "没有查询到该用户在指定月份的使用记录",
            "user_id": user_id,
            "month": month,
            "data": None,
        }

    return {
        "status": "ok",
        "message": "查询成功",
        "user_id": user_id,
        "month": month,
        "data": record,
    }

@tool(description='无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息')
def fill_context_for_report():
    return "fill_context_for_report已调用"
