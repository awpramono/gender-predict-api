from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# Load model yang sudah dilatih
model = load_model("model_gender_lstm.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

max_len = 10  # Sesuaikan dengan panjang input training

# Fungsi untuk memproses input dan prediksi
def predict_gender(nama):
    seq = tokenizer.texts_to_sequences([nama])
    seq_padded = pad_sequences(seq, maxlen=max_len)
    prediction = model.predict(seq_padded)[0][0]
    return "Perempuan" if prediction > 0.5 else "Laki-laki"

# API endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    nama = data.get("nama", "")

    if not nama:
        return jsonify({"error": "Nama tidak boleh kosong!"}), 400

    gender = predict_gender(nama)
    return jsonify({"nama": nama, "gender": gender})

# Menjalankan aplikasi
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
