from flask import Flask, render_template, jsonify, abort
import os
import json

app = Flask(__name__)

# 📘 首頁（可自行修改成介紹頁或 redirect）


@app.route('/')
def home():
    return render_template('index.html')

# 📄 作業報告頁（HW4-1, HW4-2 說明）


@app.route('/report')
def report():
    return render_template('report.html')

# 📊 訓練成果頁（顯示 reward 圖）


@app.route('/results')
def results():
    return render_template('results.html')


@app.route('/api/rewards/<model>')
def api_rewards(model):
    path = f'data/{model}_rewards.json'
    if os.path.exists(path):
        with open(path) as f:
            return jsonify(json.load(f))
    else:
        abort(404)


# ✅ 啟動
if __name__ == '__main__':
    app.run(debug=True)
