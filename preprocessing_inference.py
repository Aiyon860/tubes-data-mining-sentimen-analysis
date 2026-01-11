import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Emoji regex
emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "]+",
    flags=re.UNICODE
)

# Sastrawi tools
stemmer = StemmerFactory().create_stemmer()
stopwords = set(StopWordRemoverFactory().get_stop_words())


# ==========================
# 1. KAMUS SLANG
# ==========================
slang_dict = {
    "brg": "barang", "dr": "dari", "dri": "dari",
    "udh": "sudah", "sdh": "sudah",
    "bgt": "banget", "bngt": "banget",
    "gk": "tidak", "ga": "tidak", "gak": "tidak",
    "nga": "tidak", "ngga": "tidak", "nggak": "tidak",
    "aq": "saya", "sy": "saya",
    "tp": "tapi", "tpi": "tapi",
    "dg": "dengan", "dgn": "dengan",
    "sm": "sama", "sgt": "sangat",
    "krn": "karena", "trs": "terus", "trus": "terus",
    "jd": "jadi", "lg": "lagi", "lgsg": "langsung"
}


# ==========================
# 2. FUNGSI NORMALISASI SLANG
# ==========================
def normalize_slang(tokens):
    """Mengubah kata slang menjadi kata baku menggunakan kamus slang"""
    return [slang_dict.get(t, t) for t in tokens]


# ==========================
# 3. PIPELINE PREPROCESSING
# ==========================
def preprocess_text(text):
    # Casefolding
    text = text.lower()

    # Ambil emoji
    emojis = emoji_pattern.findall(text)

    # Hapus karakter selain huruf (emoji dipertahankan)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Kembalikan emoji sebagai token
    if emojis:
        text = text + " " + " ".join(emojis)

    # Hilangkan spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenisasi
    tokens = text.split()

    # === NORMALISASI SLANG (tahap baru) ===
    tokens = normalize_slang(tokens)

    # Stopword Removal
    tokens = [t for t in tokens if t not in stopwords]

    # Stemming (emoji dilewati)
    tokens = [stemmer.stem(t) if t.isalpha() else t for t in tokens]

    return " ".join(tokens)
