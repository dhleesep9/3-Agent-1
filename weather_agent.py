import os
import json
import requests
import openai
from dotenv import load_dotenv


"""
🌤️ 9주차 과제: 오늘의 날씨를 알려주는 Agent (Python 스크립트 버전)

이 파일은 Jupyter Notebook(`9주차-과제-날씨-Agent.ipynb`)에 있던 코드를
그대로 Python 스크립트 형태로 옮긴 것입니다.

사용 예시:
    python weather_agent.py
"""


# === 1. 환경 설정 및 OpenAI 클라이언트 초기화 ===
load_dotenv()

client = openai.OpenAI()
MODEL = "gpt-4o-mini"


# === 2. Tool: 오늘 날씨 조회 함수 ===
def get_today_weather(city: str) -> str:
    """
    지정된 도시의 현재 날씨 정보를 가져옵니다.

    Args:
        city: 도시 이름 (예: "Seoul", "Busan", "Tokyo")

    Returns:
        날씨 정보를 담은 JSON 문자열
        예: {"weather": "Clear", "temp": 25.5, "feels_like": 26.0, "humidity": 40}
    """
    # 과제에서 제공된 OpenWeatherMap API KEY
    API_KEY = "cc8369ec59d8c647f4797cec0fc9fef1"
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&appid={API_KEY}&units=metric&lang=kr"
    )

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # 필요한 정보 추출
        weather_main = data.get("weather", [{}])[0].get("main", "Unknown")
        weather_description = data.get("weather", [{}])[0].get(
            "description", "Unknown"
        )
        temp = data.get("main", {}).get("temp")
        feels_like = data.get("main", {}).get("feels_like")
        humidity = data.get("main", {}).get("humidity")

        # JSON 문자열로 반환
        result = {
            "weather": weather_main,
            "description": weather_description,
            "temp": temp,
            "feels_like": feels_like,
            "humidity": humidity,
        }
        return json.dumps(result, ensure_ascii=False)
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            return json.dumps(
                {"error": f"'{city}' 도시를 찾을 수 없습니다."}, ensure_ascii=False
            )
        else:
            return json.dumps(
                {"error": f"API 요청 중 에러 발생: {e}"}, ensure_ascii=False
            )
    except Exception as e:
        return json.dumps(
            {"error": f"API 요청 중 에러 발생: {e}"}, ensure_ascii=False
        )


# === 3. JSON 스키마 정의 (Tool Spec) ===
weather_tool_schema = {
    "type": "function",
    "function": {
        "name": "get_today_weather",
        "description": "지정된 도시의 현재 날씨 정보(날씨 상태, 온도, 체감 온도, 습도 등)를 조회합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "날씨를 조회할 도시 이름 (예: Seoul, Busan, Tokyo, New York)",
                }
            },
            "required": ["city"],
        },
    },
}


# === 4. 사용 가능한 함수를 매핑 ===
available_functions = {
    "get_today_weather": get_today_weather,
}


def run_agent_for_city(city: str) -> str:
    """
    주어진 도시 이름에 대해
    '오늘 {city} 날씨 어때?' 라는 질문을 던지고,
    Tool을 이용해 최종 자연어 답변을 반환합니다.
    """
    user_message = f"오늘 {city} 날씨 어때?"
    messages = [{"role": "user", "content": user_message}]
    tools = [weather_tool_schema]

    print("💬 1단계: LLM에 Tool 호출 요청")
    response = client.chat.completions.create(
        model=MODEL, messages=messages, tools=tools, tool_choice="auto"
    )
    response_message = response.choices[0].message
    messages.append(response_message)

    tool_calls = response_message.tool_calls
    if tool_calls:
        print("✅ LLM이 Tool 사용을 결정했습니다.")
        print("\n🔧 2단계: 결정된 Tool을 실제로 실행")
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            print(
                f"- 실행 함수: {function_name}("
                + ", ".join([f"{k}={v}" for k, v in function_args.items()])
                + ")"
            )

            function_response = function_to_call(**function_args)
            print(f"- 실행 결과: {function_response}")

            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )

        print("\n💭 3단계: Tool 실행 결과를 바탕으로 최종 답변 생성")
        final_response = client.chat.completions.create(
            model=MODEL, messages=messages
        )
        agent_answer = final_response.choices[0].message.content
        print("\n===== Agent의 최종 답변 =====")
        print(agent_answer)
        return agent_answer
    else:
        print("❌ LLM이 Tool 사용을 결정하지 않았습니다.")
        final_response = client.chat.completions.create(
            model=MODEL, messages=messages
        )
        agent_answer = final_response.choices[0].message.content
        print("\n===== Agent의 최종 답변 =====")
        print(agent_answer)
        return agent_answer


def main():
    # 기본 예시: 서울 날씨
    run_agent_for_city("서울")

    # 추가로 테스트해보고 싶으면 아래 주석을 해제해서 사용하세요.
    # for city in ["Busan", "Tokyo", "New York"]:
    #     print("\n" + "=" * 50)
    #     print(f"🌍 {city} 날씨 조회")
    #     print("=" * 50)
    #     run_agent_for_city(city)


if __name__ == "__main__":
    main()


