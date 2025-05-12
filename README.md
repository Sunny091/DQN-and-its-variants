# 🧠 Deep Q-Learning and Its Variants – HW4 Interactive Showcase

This project presents an interactive web interface built with Flask to showcase and compare various Deep Q-Learning (DQN) algorithms as implemented in HW4 of a Deep Reinforcement Learning course.

It combines model training (in PyTorch / PyTorch Lightning), reward tracking, and real-time visualization via Plotly.

---

## 📦 Features

-   🧪 **Train and evaluate**: Naive DQN, Double DQN, Dueling DQN, and Lightning DQN
-   📉 **Interactive reward curve** with zoom/range/scale toggle
-   🌐 **Flask Web UI** with clean Bootstrap layout
-   📊 **Live JSON-based data loading** for real-time result rendering

---

## 📁 Project Structure

```bash
├── app.py                       # Main Flask app
├── templates/
│   ├── base.html               # Common layout template
│   ├── index.html              # Landing page
│   ├── report.html             # Theory + result summary
│   └── results.html            # Reward curve visualizations (interactive)
├── static/
│   └── plots/                  # Reward curve images (optional)
├── data/
│   ├── naive_rewards.json
│   ├── double_rewards.json
│   ├── dueling_rewards.json
│   └── lightning_rewards.json  # Episode reward lists per model
├── training/
│   ├── dqn_train.py            # PyTorch implementation of Naive/Double/Dueling
│   ├── naive_train.py          # Baseline Naive DQN
│   └── lightning_train.py      # PyTorch Lightning version
└── README.md                   # This file
```

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Make sure to include:

-   `flask`
-   `pytorch-lightning`
-   `matplotlib`
-   `plotly`

### 2. Train the models

Train and generate reward curves:

```bash
python training/naive_train.py
python training/dqn_train.py    # For double / dueling
python training/lightning_train.py
```

> This will generate reward plots (PNG) and `.json` files under `data/`

### 3. Launch the Flask server

```bash
python app.py
```

Visit: [http://localhost:5000](http://localhost:5000)

---

## 📄 Page Overview

| Route                  | Purpose                               |
| ---------------------- | ------------------------------------- |
| `/`                    | Home page (navigation)                |
| `/report`              | Theoretical explanation (HW4-1~4)     |
| `/results`             | Interactive reward plot comparison    |
| `/api/rewards/<model>` | JSON API for each model’s reward data |

---

## 🔍 Model Summary

| Model         | Technique                       | Benefit                               |
| ------------- | ------------------------------- | ------------------------------------- |
| Naive DQN     | Basic Q-learning                | Simple, fast but unstable             |
| Double DQN    | Decoupled target selection      | Reduces overestimation bias           |
| Dueling DQN   | Value & Advantage decomposition | More stable for value-dominant states |
| Lightning DQN | PyTorch Lightning + scheduler   | Cleaner training, easier to extend    |

---

## 📷 Screenshots

-   📊 `/results`: Reward curves with zoom, pan, log scale toggle
-   🧠 `/report`: Architecture comparisons and loss formula

---

## 📌 Notes

-   Data files are loaded dynamically – missing reward JSONs will silently fail
-   You can export plots as images via the Plotly menu

---

## 🤖 Author

Developed as part of a university course on Deep Reinforcement Learning (Spring 2025).

---

## 🧠 License

MIT License

Feel free to adapt and extend this code for your own RL experiments!
