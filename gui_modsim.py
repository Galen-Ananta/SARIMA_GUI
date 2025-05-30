# streamlit_sarima_gui_light.py - Bagian 1: Import & Sesi 1

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import scipy.stats as stats

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro

from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from pmdarima.arima import ARIMA as PMDARIMA

# ===== Caching Fungsi Berat =====
@st.cache_data
def run_auto_arima(ts, seasonal_period):
    model = pm.auto_arima(
        ts,
        seasonal=True,
        m=seasonal_period,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        max_p=3, max_q=3, max_d=2,
        max_P=2, max_Q=2, max_D=1
    )
    return model

@st.cache_data
def fit_sarimax(ts, order, seasonal_order):
    model = SARIMAX(ts, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False)

def predict_model(model, n_steps, index):
    try:
        if isinstance(model, PMDARIMA):
            forecast = model.predict(n_periods=n_steps)
        else:
            forecast = model.forecast(steps=n_steps)
        return pd.Series(forecast, index=index)
    except Exception as e:
        st.error(f"❌ Gagal melakukan prediksi: {e}")
        return pd.Series([np.nan]*n_steps, index=index)

st.set_page_config(page_title="SARIMA Time Series GUI", layout="wide")

# Sidebar navigasi
sesi = st.sidebar.radio("Navigasi Sesi", [
    "Sesi 1: Upload & Setup Waktu",
    "Sesi 2: Eksplorasi Data",
    "Sesi 3: Identifikasi Model",
    "Sesi 4: Pemodelan",
    "Sesi 5: Evaluasi & Uji Diagnostik",
    "Sesi 6: Forecast & Export"
])

# ===== Sesi 1: Upload & Setup Waktu =====
if sesi == "Sesi 1: Upload & Setup Waktu":
    st.title(":bar_chart: Sesi 1: Upload & Setup Waktu")

    st.title(":bar_chart: SARIMA GUI - Sesi 1: Upload & Setup Waktu")

    st.subheader("👥 Informasi Kelompok")
    st.markdown("""
    **Kelompok 2**
    - Galen Ananta (140110220006)
    - Rizky Febrian (140110220019)
    - Asrie Putri Janitha (140110220023)
    - Zhafir Alhaq Ali Aqil B. (140110220053)
    - Thania Nur Salsabila (140110220057)
    - Raisa Huria Pasha (140110220064)
    - Muhammad Ismail Sabiq (140110220066)

    📄 **Format CSV yang diterima:**
    - Hanya 1 kolom berisi nilai time series (tanpa header pun boleh)
    - Setelah upload, kamu akan diminta mengisi tanggal awal dan frekuensi data
    """)

    uploaded_file = st.file_uploader("📤 Upload CSV Time Series (1 kolom nilai)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file, header=None)
        ts_values = df.iloc[:, 0].dropna()

        st.success("✅ Data berhasil dimuat")
        st.write(ts_values.head())

        st.subheader("📅 Setup Waktu")
        start_date = st.date_input("Tanggal awal", pd.to_datetime("2020-01-01"))
        freq_str = st.selectbox("Frekuensi", ["D - Harian", "M - Bulanan", "Y - Tahunan"])
        freq_map = {"D - Harian": "D", "M - Bulanan": "M", "Y - Tahunan": "Y"}
        freq = freq_map[freq_str]

        try:
            date_index = pd.date_range(start=start_date, periods=len(ts_values), freq=freq)
            ts = pd.Series(ts_values.values, index=date_index)
            st.success("✅ Index waktu dibuat")

            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode="lines", name="Data"))
            fig.update_layout(title="Plot Time Series", xaxis_title="Waktu", yaxis_title="Nilai")
            st.plotly_chart(fig, use_container_width=True)

            # Split Train/Test
            st.subheader("📚 Split Training & Testing")
            ratio = st.slider("Persentase data training", 50, 95, 80, step=5)
            split_idx = int(len(ts) * (ratio / 100))
            train_ts = ts.iloc[:split_idx]
            test_ts = ts.iloc[split_idx:]

            st.write(f"Jumlah data training: {len(train_ts)}")
            st.write(f"Jumlah data testing: {len(test_ts)}")

            # Simpan ke session state
            st.session_state.ts = ts
            st.session_state.freq = freq
            st.session_state.start_date = start_date
            st.session_state.train_ts = train_ts
            st.session_state.test_ts = test_ts

        except Exception as e:
            st.error(f"❌ Gagal membuat index waktu: {e}")

# ===== Sesi 2: Eksplorasi Data =====
elif sesi == "Sesi 2: Eksplorasi Data":
    st.title(":bar_chart: Sesi 2: Eksplorasi Data")

    if "train_ts" not in st.session_state:
        st.warning("⚠️ Silakan upload data di Sesi 1 terlebih dahulu.")
        st.stop()

    ts = st.session_state.train_ts

    st.subheader("📉 Plot Data Training")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode="lines", name="Training Data"))
    fig.update_layout(title="Plot Data Training", xaxis_title="Waktu", yaxis_title="Nilai")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 QQ-Plot")
    fig2, ax = plt.subplots()
    stats.probplot(ts, dist="norm", plot=ax)
    st.pyplot(fig2)

# ===== Sesi 3: Identifikasi Model =====
elif sesi == "Sesi 3: Identifikasi Model":
    st.title(":mag: Sesi 3: Identifikasi Model")

    if "train_ts" not in st.session_state:
        st.warning("⚠️ Silakan upload data di Sesi 1 terlebih dahulu.")
        st.stop()

    ts = st.session_state.train_ts
    lag_slider = st.slider("Jumlah lag (untuk ACF & PACF)", 5, 40, 20)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔁 Plot ACF")
        fig_acf, ax_acf = plt.subplots()
        plot_acf(ts, lags=lag_slider, ax=ax_acf)
        st.pyplot(fig_acf)

    with col2:
        st.subheader("🔁 Plot PACF")
        fig_pacf, ax_pacf = plt.subplots()
        plot_pacf(ts, lags=lag_slider, ax=ax_pacf)
        st.pyplot(fig_pacf)

    st.subheader("📉 ADF Test (Original Series)")
    adf_result = adfuller(ts)
    st.write(f"ADF Statistic: {adf_result[0]:.4f}")
    st.write(f"p-value: {adf_result[1]:.4e}")
    if adf_result[1] < 0.05:
        st.success(f"✅ Kesimpulan: p-value = {adf_result[1]:.4e} < 0.05 → Data stasioner")
    else:
        st.warning(f"⚠️ Kesimpulan: p-value = {adf_result[1]:.4e} ≥ 0.05 → Data tidak stasioner")
    st.markdown("**Hipotesis:**")
    st.markdown("- H0: Data tidak stasioner")
    st.markdown("- H1: Data stasioner")
    st.markdown("**Syarat:** Tolak H0 jika p-value < 0.05")

    st.subheader("⚙️ Differencing")
    d = st.number_input("Orde differencing (d)", 0, 5, 1)
    D = st.number_input("Musiman (D)", 0, 5, 0)
    s = st.number_input("Periode Musiman (s)", 0, 24, 12)

    ts_diff = ts.diff(d).dropna()
    if D > 0:
        ts_diff = ts_diff.diff(s * D).dropna()

    st.subheader("📉 ADF Test Setelah Differencing")
    adf_result_diff = adfuller(ts_diff)
    st.write(f"ADF Statistic: {adf_result_diff[0]:.4f}")
    st.write(f"p-value: {adf_result_diff[1]:.4e}")
    if adf_result_diff[1] < 0.05:
        st.success(f"✅ Kesimpulan: p-value = {adf_result_diff[1]:.4e} < 0.05 → Data stasioner")
    else:
        st.warning(f"⚠️ Kesimpulan: p-value = {adf_result_diff[1]:.4e} ≥ 0.05 → Data tidak stasioner")
    st.markdown("**Hipotesis:**")
    st.markdown("- H0: Data tidak stasioner")
    st.markdown("- H1: Data stasioner")
    st.markdown("**Syarat:** Tolak H0 jika p-value < 0.05")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("ACF Setelah Differencing")
        fig_acf2, ax_acf2 = plt.subplots()
        plot_acf(ts_diff, lags=lag_slider, ax=ax_acf2)
        st.pyplot(fig_acf2)

    with col4:
        st.subheader("PACF Setelah Differencing")
        fig_pacf2, ax_pacf2 = plt.subplots()
        plot_pacf(ts_diff, lags=lag_slider, ax=ax_pacf2)
        st.pyplot(fig_pacf2)

    # Simpan hasil differencing
    st.session_state.ts_diff = ts_diff

# ===== Sesi 4: Pemodelan =====
elif sesi == "Sesi 4: Pemodelan":
    st.title(":wrench: Sesi 4: Pemodelan SARIMA")

    if "ts" not in st.session_state or "ts_diff" not in st.session_state:
        st.warning("⚠️ Harap selesaikan Sesi 1–3 terlebih dahulu.")
        st.stop()

    ts = st.session_state.train_ts
    ts_full = st.session_state.ts

    model_type = st.radio("Pilih jenis pemodelan", ["Auto SARIMA", "Manual SARIMA"])

    if model_type == "Manual SARIMA":
        st.markdown("**Masukkan parameter SARIMA:**")
        col1, col2 = st.columns(2)
        with col1:
            p = st.number_input("p", 0, 5, 1)
            d = st.number_input("d", 0, 2, 1)
            q = st.number_input("q", 0, 5, 1)
        with col2:
            P = st.number_input("P", 0, 5, 0)
            D = st.number_input("D", 0, 2, 0)
            Q = st.number_input("Q", 0, 5, 0)
            s = st.number_input("s (periode musiman)", 1, 24, 12)

        if st.button("🚀 Jalankan Manual SARIMA"):
            with st.spinner("🔧 Mem-fit model SARIMA..."):
                model = fit_sarimax(ts, order=(p, d, q), seasonal_order=(P, D, Q, s))
                st.session_state.model = model
                st.success(f"✅ Model: SARIMA({p},{d},{q}) x ({P},{D},{Q}){s}")

    else:
        seasonal_period = st.number_input("s (periode musiman)", 1, 24, 12, step=1)

        if st.button("🚀 Jalankan Auto SARIMA"):
            with st.spinner("🔄 Menjalankan Auto ARIMA..."):
                model = run_auto_arima(ts, seasonal_period)
                p, d, q = model.order
                P, D, Q, s = model.seasonal_order
                st.success(f"✅ Model: SARIMA({p},{d},{q}) x ({P},{D},{Q}){s}")
                st.session_state.model = model

    # Tampilkan ringkasan model jika sudah ada
    if "model" in st.session_state:
        st.subheader("📑 Ringkasan Model")

        model = st.session_state.model
        try:
            summary_df = pd.DataFrame(model.summary().tables[1].data[1:], columns=model.summary().tables[1].data[0])
            st.dataframe(summary_df)
        except:
            st.text(model.summary())

# ===== Sesi 5: Evaluasi & Uji Diagnostik =====
elif sesi == "Sesi 5: Evaluasi & Uji Diagnostik":
    st.title(":bar_chart: Sesi 5: Evaluasi & Uji Diagnostik")

    if "model" not in st.session_state or "train_ts" not in st.session_state:
        st.warning("⚠️ Silakan selesaikan sesi sebelumnya terlebih dahulu.")
        st.stop()

    model = st.session_state.model
    train_ts = st.session_state.train_ts
    test_ts = st.session_state.test_ts

    st.subheader("📈 Evaluasi Out-of-Sample (Testing Data)")

    forecast_test = predict_model(model, n_steps=len(test_ts), index=test_ts.index)

    try:
        mae_test = mean_absolute_error(test_ts, forecast_test)
        rmse_test = np.sqrt(mean_squared_error(test_ts, forecast_test))
        mape_test = np.mean(np.abs((test_ts - forecast_test) / test_ts)) * 100

        st.write(f"**MAE:** {mae_test:.4f}")
        st.write(f"**RMSE:** {rmse_test:.4f}")
        st.write(f"**MAPE:** {mape_test:.2f}%")

        # Tambahkan AIC dan BIC
        try:
            if hasattr(model, "aic") and not callable(model.aic):
                st.write(f"**AIC:** {model.aic:.2f}")
                st.write(f"**BIC:** {model.bic:.2f}")
            elif hasattr(model, "aic") and callable(model.aic):
                st.write(f"**AIC:** {model.aic():.2f}")
                st.write(f"**BIC:** {model.bic():.2f}")
        except Exception as e:
            st.warning(f"Gagal menampilkan AIC/BIC: {e}")

        # Plot hasil prediksi vs aktual
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_ts.index, y=train_ts, name="Training"))
        fig.add_trace(go.Scatter(x=test_ts.index, y=test_ts, name="Testing"))
        fig.add_trace(go.Scatter(x=forecast_test.index, y=forecast_test, name="Prediksi"))
        fig.update_layout(title="Prediksi vs Aktual (Training + Testing)", xaxis_title="Waktu", yaxis_title="Nilai")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Model tidak bisa menghasilkan prediksi untuk data testing. Error: {e}")

    # ===== OPSIONAL: Plot Prediksi dari Awal =====
    st.write("---")
    st.subheader("📈 Prediksi dari Awal (Full Range)")

    if st.checkbox("Tampilkan prediksi dari awal (in-sample + out-of-sample)"):
        with st.spinner("⏳ Menghitung prediksi dari awal..."):
            ts_all = pd.concat([train_ts, test_ts])
            n_train = len(train_ts)
            n_test = len(test_ts)

            try:
                if isinstance(model, PMDARIMA):
                    # Prediksi in-sample + out-sample secara terpisah
                    forecast_train = model.predict_in_sample()
                    forecast_test = model.predict(n_periods=n_test)
                    forecast_all = np.concatenate([forecast_train, forecast_test])
                else:
                    # SARIMAX: prediksi langsung dari awal hingga akhir
                    forecast_all = model.predict(start=0, end=n_train + n_test - 1)

                forecast_full = pd.Series(forecast_all, index=ts_all.index)

            except Exception as e:
                st.warning(f"Gagal menghasilkan prediksi dari awal: {e}")
                forecast_full = pd.Series([np.nan]*len(ts_all), index=ts_all.index)

            fig_full = go.Figure()
            fig_full.add_trace(go.Scatter(x=ts_all.index, y=ts_all, name="Aktual"))
            fig_full.add_trace(go.Scatter(x=ts_all.index, y=forecast_full, name="Prediksi"))
            fig_full.update_layout(title="Prediksi vs Aktual (Seluruh Periode)",
                                  xaxis_title="Waktu", yaxis_title="Nilai")
            st.plotly_chart(fig_full, use_container_width=True)

            # === Export CSV
            st.subheader("📁 Download Hasil Prediksi dari Awal")
            export_df = pd.concat([
                ts_all.rename("Aktual"),
                forecast_full.rename("Prediksi")
            ], axis=1)
            output_csv = export_df.reset_index().rename(columns={"index": "Tanggal"}).to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", output_csv, file_name="prediksi_dari_awal.csv", mime="text/csv")


# ===== Sesi 6: Forecast & Export =====
elif sesi == "Sesi 6: Forecast & Export":
    st.title(":crystal_ball: Sesi 6: Forecast & Export")

    if "ts" not in st.session_state or "model" not in st.session_state:
        st.warning("⚠️ Silakan selesaikan sesi sebelumnya terlebih dahulu.")
        st.stop()

    ts = st.session_state.ts
    train_ts = st.session_state.train_ts
    test_ts = st.session_state.test_ts
    model = st.session_state.model
    freq = st.session_state.freq

    st.subheader("🔮 Forecast Lengkap")
    horizon = st.number_input("Berapa langkah ke depan ingin diprediksi?", min_value=1, max_value=120, value=12)
    forecast_test = predict_model(model, n_steps=horizon, index=test_ts.index)

    if st.button("🚀 Jalankan Forecast & Plot Gabungan"):
        with st.spinner("⏳ Menghitung..."):
            try:
                # Plot hasil prediksi
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=train_ts.index, y=train_ts, name="Training"))
                fig.add_trace(go.Scatter(x=forecast_test.index, y=forecast_test, name="Prediksi"))
                fig.update_layout(title="Prediksi Waktu ke Depan", xaxis_title="Waktu", yaxis_title="Nilai")
                st.plotly_chart(fig, use_container_width=True)

                # === Export ===
                st.subheader("📁 Download Hasil Prediksi")
                # Gabungkan data Training + Forecast jadi satu dataframe
                forecast_df = forecast_test.rename("Forecast")
                training_df = train_ts.rename("Training")

                # Buat satu dataframe gabungan dengan kolom keterangan
                export_df = pd.concat([
                    pd.DataFrame({"Tanggal": training_df.index, "Nilai": training_df.values, "Tipe": "Training"}),
                    pd.DataFrame({"Tanggal": forecast_df.index, "Nilai": forecast_df.values, "Tipe": "Forecast"})
                ])

                # Export ke CSV
                output_csv = export_df.to_csv(index=False).encode("utf-8")

                # Tombol download
                st.download_button("📥 Download CSV", output_csv, file_name="forecast_output.csv", mime="text/csv")


            except Exception as e:
                st.error(f"❌ Gagal menghasilkan prediksi: {e}")
