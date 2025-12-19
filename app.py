import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Muat model yang sudah disimpan
model = joblib.load('model.pkl')

# Fungsi untuk mengonversi input waktu layar berdasarkan pedoman kesehatan
def convert_screen_time(x):
    if pd.isna(x):
        return np.nan
    x = str(x)
    if "< 2" in x:
        return 1.5
    elif "2-4" in x:
        return 3.0
    elif "> 4" in x:
        return 5.0
    else:
        return np.nan

# Fungsi untuk kesimpulan berdasarkan prediksi
def generate_conclusion(predicted_sleep_time):
    if predicted_sleep_time < 6:
        return "Waktu tidur Anda lebih rendah dari rekomendasi sehat (7-9 jam). Sebaiknya tingkatkan durasi tidur untuk kesehatan yang lebih baik."
    elif 6 <= predicted_sleep_time <= 8:
        return "Waktu tidur Anda berada dalam rentang yang sehat. Pertahankan gaya hidup sehat ini."
    else:
        return "Waktu tidur Anda lebih dari yang disarankan. Pertimbangkan untuk mengurangi waktu tidur yang berlebihan untuk menjaga kesehatan."

# Menambahkan CSS untuk styling kustom
st.markdown(
    """
    <style>
    /* Background Gradient untuk aplikasi */
    .stApp {
        background: linear-gradient(135deg, #FFDEE9 0%, #B5FFFC 100%);
        padding: 20px;
    }

    /* Styling untuk judul */
    .stTitle {
        color: #2F4F4F;
        font-size: 36px;
        font-weight: 700;
        text-align: center;
        padding-bottom: 20px;
    }

    /* Styling untuk tombol */
    .stButton>button {
        background-color: #FF6347;
        color: white;
        padding: 12px 30px;
        border: none;
        cursor: pointer;
        border-radius: 20px;
        font-size: 18px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: background-color 0.3s ease, transform 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #FF4500;
        transform: scale(1.05);
    }

    /* Styling untuk kolom input */
    .stSelectbox, .stNumberInput {
        margin-bottom: 20px;
        font-size: 18px;
        padding: 10px;
        background-color: #f0f0f0;
        border-radius: 10px;
        border: 1px solid #ccc;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    /* Styling untuk input select dan number */
    .stSelectbox>div>div>input {
        font-size: 18px;
    }

    .stSelectbox>div>div>input:focus {
        outline: none;
        border: 1px solid #FF6347;
    }

    /* Styling untuk hasil kesimpulan */
    .stMarkdown {
        font-size: 18px;
        color: #333;
        font-weight: 600;
    }

    /* Styling untuk grafik */
    .stGraph {
        padding-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Menyimpan halaman yang sedang aktif menggunakan session_state
if 'page' not in st.session_state:
    st.session_state.page = "Tentang Aplikasi"  # Default page is "Tentang Aplikasi"

if 'user_name' not in st.session_state:
    st.session_state.user_name = None  # Default name is None

# Sidebar navigation
page = st.sidebar.radio("Pilih Halaman", ["Tentang Aplikasi", "Prediksi Waktu Tidur", "Hubungi Kami"])

# Set halaman yang dipilih
st.session_state.page = page

# Jika pengguna baru, tampilkan kolom input nama
if st.session_state.page == "Tentang Aplikasi" and st.session_state.user_name is None:
    st.title("Tentang Aplikasi")
    st.write(
        "Aplikasi ini memprediksi waktu tidur Anda berdasarkan beberapa faktor kesehatan seperti usia, jenis kelamin, kebiasaan olahraga, waktu penggunaan layar, dan lainnya. "
        "Masukkan nama Anda untuk melanjutkan."
    )
    # Kolom input untuk nama
    user_name = st.text_input("Masukkan Nama Anda")

    # Setelah nama dimasukkan, tampilkan pesan selamat datang
    if user_name:
        st.session_state.user_name = user_name
        st.success(f"Selamat datang, {user_name}!")
        # Secara otomatis mengarahkan ke halaman Prediksi Waktu Tidur
        st.session_state.page = "Prediksi Waktu Tidur"  # Navigasi ke halaman Prediksi Waktu Tidur
else:
    if st.session_state.page == "Prediksi Waktu Tidur":
        # Input fitur dari pengguna
        st.title("Prediksi Waktu Tidur Berdasarkan Fitur Kesehatan")

        col1, col2 = st.columns(2)

        with col1:
            usia_input = st.number_input("Usia (dalam tahun)", min_value=10, max_value=100, value=25)

        with col2:
            jenis_kelamin_input = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])

        # Input lainnya
        penyakit_fisik_input = st.selectbox("Penyakit Fisik", ["Ya", "Tidak"])
        screen_time_input = st.selectbox("Waktu Penggunaan Layar (per hari)", ["< 2 jam", "2-4 jam", "> 4 jam"])
        frekuensi_olahraga_input = st.selectbox("Frekuensi Olahraga", ["Jarang", "Kadang", "Sering"])
        rokok_alkohol_input = st.selectbox("Kebiasaan Merokok atau Minum Alkohol", ["Ya", "Tidak"])

        # Proses input
        input_data = {
            "usia": usia_input,
            "jenis_kelamin": jenis_kelamin_input,
            "penyakit_fisik": penyakit_fisik_input,
            "screen_time": convert_screen_time(screen_time_input),
            "frekuensi_olahraga": frekuensi_olahraga_input,
            "rokok_alkohol": rokok_alkohol_input
        }

        # Preprocess input
        input_df = pd.DataFrame([input_data])

        # Prediksi
        input_pred = model.predict(input_df)

        # Tampilkan hasil prediksi
        st.write(f"Prediksi waktu tidur Anda adalah: {input_pred[0]:.2f} jam")

        # Kesimpulan Berdasarkan Prediksi
        kesimpulan = generate_conclusion(input_pred[0])
        st.markdown(f"### Kesimpulan: {kesimpulan}")

        # Tampilkan grafik MAE dan MSE
        MAE_test = 0.496
        MSE_test = 0.452

        fig, ax = plt.subplots()
        ax.bar(["MAE", "MSE"], [MAE_test, MSE_test], color=["skyblue", "orange"])
        st.pyplot(fig)

    elif st.session_state.page == "Hubungi Kami":
        st.title("Hubungi Kami")
        st.write(
            "Jika Anda memiliki pertanyaan atau feedback, silakan hubungi kami di: \n"
            "Email: cobacoba@gmail.com"
        )
