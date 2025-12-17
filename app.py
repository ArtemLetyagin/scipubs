from flask import Flask, render_template, request, jsonify
from datetime import datetime
from scipubs_mas_cli.main import run_one_query, render_result
# import sys
# sys.path.append("scipubs_mas_cli")
import shutil
import os
import pandas as pd


app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    # Рендерим HTML с простым окном чата
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    Маршрут, который вызывается с фронтенда после отправки сообщения.
    Ожидает JSON: { "message": "..." }
    Возвращает JSON: { "reply": "..." }
    """
    shutil.rmtree("exports")
    os.mkdir("exports")

    data = request.get_json(force=True)
    user_message = data.get("message", "").strip()
    history = data.get("history", [])  # если захочешь слать историю с фронта

    if not user_message:
        return jsonify({"error": "Пустое сообщение"}), 400

    # Вызов твоего LLM-агента
    state, user_query = run_one_query(user_message, interactive=True)

    if state.get("needs_clarification"):
        question_raw = state["question_raw"]
        anayst_result = state.get("clarification_prompt")
        plot_base64 = False
    else:
        question_raw = state["question_raw"]
        anayst_result = state["analyst_result"].summary
        plot_base64 = state["analyst_result"].plot_base64

    html_table = False

    render_result(state, user_query)
    
    if len([file for file in os.listdir("exports") if file.endswith(".csv")]) > 0:
        df = pd.read_csv("exports/" + [file for file in os.listdir("exports") if file.endswith(".csv")][0])
        html_table = df.to_html(index=False, classes='table table-striped')
    # ===== TEST =====
    # anayst_result = "model answer"
    # with open("plot.png", "rb") as image_file:
    #     import base64
    #     plot_base64 = base64.b64encode(image_file.read()).decode("ascii")#.decode('utf-8')
    #     df = pd.read_csv("exports/" + [file for file in os.listdir("exports") if file.endswith(".csv")][0])
    #     html_table = df.to_html(index=False, classes='table table-striped')

    # ===== RETURN =====
    return jsonify(
        {
            "reply": anayst_result,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "plot_base64": plot_base64,
            "html_table": html_table
        }
    )


if __name__ == "__main__":
    # Запуск сервера
    app.run(host="0.0.0.0", port=5000, debug=True)


"""
построй мне таблицу из самых цитируемых работ по machine learning за 2024 год, используй openalex
построй мне таблицу из самых цитируемых работ по machine learning за 2024 год
Построй динамику числа публикаций по теме deep learning с 2010 по 2024 год, используй openalex
Построй динамику числа публикаций по теме deep learning с 2010 по 2024 год
Сравни динамику числа публикаций по deep learning с числом публикаций по nlp с 2020 до 2024 года, используй openalex
Сравни динамику числа публикаций по deep learning с числом публикаций по nlp с 2020 до 2024 года
покажи, в каких журналах чаще всего публикуют работы по теме deep learning, используй openalex
покажи, в каких журналах чаще всего публикуют работы по теме deep learning
"""

"""
покажи, в каких журналах чаще всего публикуют работы по теме deep learning
покажи то же самое но про machine learning за 2024 год

какая погода в Питере
"""