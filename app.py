import joblib
import numpy as np
import streamlit as st

from preprocessing_inference import preprocess_text


# ================================
# Load all models + TF-IDF
# ================================
@st.cache_resource
def load_all():
    model_nb = joblib.load("model_nb.pkl")
    model_lr = joblib.load("model_lr.pkl")
    model_svm = joblib.load("model_svm.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    return {
        "Naive Bayes": model_nb,
        "Logistic Regression": model_lr,
        "SVM (LinearSVC)": model_svm,
    }, tfidf


models, tfidf = load_all()

# ================================
# Streamlit UI
# ================================
st.title("Analisis Sentimen Tokopedia (2025)")
st.write("Masukkan ulasan produk untuk memprediksi sentimen menggunakan 3 model ML.")

# Pilihan model
model_choice = st.selectbox(
    "Pilih Model", ["Naive Bayes", "Logistic Regression", "SVM (LinearSVC)"]
)
model = models[model_choice]

# Input user
text_input = st.text_area("Masukkan Ulasan:", height=120)


# ================================
# Fungsi WordCloud
# ================================
def generate_wordcloud(text):
    wc = WordCloud(width=600, height=300, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig


# ================================
# Tombol Prediksi
# ================================
if st.button("Prediksi Sentimen"):
    if text_input.strip() == "":
        st.warning("⚠ Masukkan teks ulasan terlebih dahulu.")
    else:
        # Preprocess
        processed = preprocess_text(text_input)

        # TF-IDF transform
        X_input = tfidf.transform([processed])

        # Predict
        pred = model.predict(X_input)[0]

        # Confidence score (LR & NB support prob)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_input)[0]
        else:
            # LinearSVC tidak punya prob → pakai decision_function
            df = model.decision_function(X_input)[0]
            proba = np.exp(df) / np.sum(np.exp(df))  # softmax manual

        # Mapping sentimen

        # Mapping sentimen (mendukung label int & string)
        label_map = {
            0: "😡 Negative",
            1: "😐 Neutral",
            2: "😊 Positive",
            "negative": "😡 Negative",
            "neutral": "😐 Neutral",
            "positive": "😊 Positive",
        }

        sentiment_text = label_map.get(pred, "Unknown")

        st.subheader("Hasil Prediksi")
        st.markdown(f"### {sentiment_text}")

        # Confidence score tampil
        st.subheader("Confidence Score")
        st.json(
            {
                "Negative": float(proba[0].round(4)),
                "Neutral": float(proba[1].round(4)),
                "Positive": float(proba[2].round(4)),
            }
        )
