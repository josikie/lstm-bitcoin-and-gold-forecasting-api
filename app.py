## Import Library
# numpy untuk manipulasi struktur data 
# tensorflow memuat dan menjalankan model LSTM
# joblib untuk memuat skaler 
# yfinance untuk download data terkini emas dan bitcoin
# flask untuk membuat server flask
# request membaca data dari pengguna
# jsonify untuk mengubah data menjadi format JSON
import numpy as np
import tensorflow as tf
import joblib
import yfinance as yf
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from flask_cors import CORS
from datetime import datetime, timedelta
from tensorflow.keras.layers import Dense
from curl_cffi.requests import Session

print("Library Berhasil di import")

# inisiasi server flask api
app = Flask(__name__)
CORS(app)


class CustomDense(Dense):
    def __init__(self, *args, **kwargs):
        # Buang paksa parameter yang bikin server VPS ngambek
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

# Daftarkan kelas kustom ini ke objek loader TensorFlow
custom_objects = {'Dense': CustomDense}

# muat model dan skaler
try: 
    # btc_model = load_model("lstm_bitcoin_model.h5", compile=False, custom_objects=custom_objects)
    # gold_model = load_model("lstm_gold_model.h5", compile=False, custom_objects=custom_objects)
    btc_model = load_model("lstm_bitcoin_model.h5", compile=False)
    gold_model = load_model("lstm_gold_model.h5", compile=False)
    btc_scaler = joblib.load("btc_scaler.pkl")
    gold_scaler = joblib.load("gold_scaler.pkl")
    print("Model dan skaler dimuat")
except Exception as e:
    print(f"Gagal memuat file aset: {e}")

# Endpoint untuk menampilkan dashboard
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint untuk menampilkan BTC
@app.route('/bitcoin')
def bitcoin():
    return render_template('bitcoin.html')

# Endpoint untuk menampilkan gold
@app.route('/gold')
def gold():
    return render_template('gold.html')

# Endpoint ini mengambil parameter url asset (btc atau gold)
@app.route('/predict', methods=['GET'])
def predict():
    try:
        asset = request.args.get('asset', 'btc').lower()
        if asset == 'btc':
            ticker = "BTC-USD"
            model = btc_model
            scaler = btc_scaler
        elif asset == 'gold':
            ticker = 'GC=F'
            model = gold_model
            scaler = gold_scaler
        else:
            return jsonify({"status": "error", "message": "Aset tidak valid"}), 400
        # Gunakan adapter bawaan untuk mencegah timeout jabat tangan SSL
        yfinance_data = yf.download(tickers=ticker, period="45d", interval="1d")
        if yfinance_data.empty or len(yfinance_data) < 30:
            return jsonify({"status": "error", "message": "Data gagal dimuat"}), 500
        # ratain kolom bertingkat yfinance
        yfinance_data.columns = yfinance_data.columns.get_level_values(0)
        # ambil 30 hari data untuk memprediksi keesokan hari (lock satu harga)
        df_features = yfinance_data[['Close', 'Volume']].iloc[-31:-1]
        current_sequence = df_features.values.astype(float).tolist()

        # normalisasi data menggunakan skaler
        scaled_sequence = scaler.transform(np.array(current_sequence))

        # reshape jadi 3d format 
        input_3d = scaled_sequence.reshape(1, 30, 2)

        # prediksi
        prediction = model.predict(input_3d, verbose=0)
        predictions = prediction.flatten().tolist()
        

        # mengubah angka hasil normalisasi ke harga asli
        dummy_matrix = np.zeros((7,2))
        dummy_matrix[:, 0] = predictions
        real_prediction_prices = scaler.inverse_transform(dummy_matrix)
        predictions_prices = real_prediction_prices[:,0].tolist()

        if asset == 'gold':
            # 1. Ambil harga asli emas hari terakhir dari pasar (misal: $4341.00)
            last_real_price = yfinance_data['Close'].iloc[-1].item()
            
            # 2. Hitung selisih harga absolut antara tebakan pertama model dengan harga asli pasar
            absolute_gap = predictions_prices[0] - last_real_price
            
            # 3. Geser seluruh 7 hari prediksi ke bawah sebesar nilai gap tersebut
            # Trik ini menjamin hari pertama akan PAS di harga pasar saat ini ($4300-an)
            # DAN 100% mempertahankan lekukan naik-turun serta bentuk asli grafik Anda!
            adjusted_prices = []
            for price in predictions_prices:
                adjusted_prices.append(price - absolute_gap)
                
            predictions_prices = [float(p) for p in adjusted_prices]

        # data tanggal
        base_date = datetime.now()
        dates_list = [(base_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)]

        # respon api
        return jsonify({
            "status": "success",
            "asset" : "Bitcoin (BTC)" if asset == 'btc' else "Emas (Gold)",
            "ticker": ticker,
            "tomorrow_prediction": round(predictions_prices[0], 2),
            "seven_days_projection": [round(price, 2) for price in predictions_prices],
            "dates": dates_list
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Endpoint mengembalikan data bitcoin dan emas 5 tahun terakhir
@app.route('/five_years_data', methods=['GET'])
def five_years_data():
    try:
        # Tarik data 5 tahun kebelakang (5y) dengan interval bulanan (1mo)
        btc_raw =  yf.download("BTC-USD", period="5y", interval="1mo")
        gold_raw = yf.download("GC=F", period="5y", interval="1mo")
        
        # # Ratakan kolom bertingkat yfinance
        btc_raw.columns = btc_raw.columns.get_level_values(0)
        gold_raw.columns = gold_raw.columns.get_level_values(0)
        
        # Ambil kolom Close, buang baris kosong, dan ambil indeks tanggalnya
        btc_dates = btc_raw.index.strftime('%Y-%m').tolist()
        btc_prices = btc_raw['Close'].dropna().values.astype(float).tolist()
        
        gold_dates = gold_raw.index.strftime('%Y-%m').tolist()
        gold_prices = gold_raw['Close'].dropna().values.astype(float).tolist()
        
        return jsonify({
            "status": "success",
            "btc": {"labels": btc_dates, "prices": btc_prices},
            "gold": {"labels": gold_dates, "prices": gold_prices}
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# Endpoint untuk memanggil data harga bitcoin hari ini
@app.route('/btc_today', methods=['GET'])
def bitcoin_today():
    try:
        data_raw = yf.download("BTC-USD", period="5d", interval="1d")
        data_raw.columns = data_raw.columns.get_level_values(0)
        
        # Mengambil harga btc pada menit terakhir dari yfinance
        harga_btc = float(data_raw['Close'].iloc[-1])
        
        return jsonify({
            "status": "success",
            "asset": "Bitcoin (BTC-USD)",
            "today_price": round(harga_btc, 2)
        }), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# Endpoint untuk memanggil data harga emas hari ini
@app.route('/gold_today', methods=['GET'])
def gold_today():
    try:        
        # Mengunduh data pasar Emas ETF (GLD) 
        data_raw = yf.download("GC=F", period="4d", interval="1d")
        data_raw.columns = data_raw.columns.get_level_values(0)
        
        # Mengambil harga emas pada menit terakhir dari yfinance
        harga_gold = float(data_raw['Close'].dropna().iloc[-1])
        
        return jsonify({
            "status": "success",
            "asset": "Gold",
            "today_price": round(harga_gold, 2)
        }), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
# buat server di lokal
if __name__ == '__main__':
    app.run(debug=True, port=5000)