import joblib

model = joblib.load("stress_model.pkl")

tests = [
    "deadline pressure is high",
    "I feel relaxed today",
    "I am tired and frustrated"
]

for t in tests:
    print(t, "→", model.predict([t])[0])
