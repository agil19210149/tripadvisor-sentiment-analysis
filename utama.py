import re
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import string
import streamlit.components.v1 as components

from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from streamlit_option_menu import option_menu


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Analisis Sentimen Hotel Sulawesi Utara",
    page_icon="🏨",
    layout="wide"
)
# =====================================================
# PATH FILE
# =====================================================

DATA_MENTAH = r"data/hotel_sulut.csv"

DATA_FINAL = r"data/hasil_analisis_final_bertopic.csv"

HASIL_MODEL = r"data/Hasil_Perbandingan_Model.csv"

BERTOPIC = r"data/hasil_analisis_final_bertopic.csv"

CASEFOLDING = r"data/Lampiran_1_CaseFolding_Filtering.csv"

TOKEN = r"data/Lampiran_2_Tokenization.csv"

STEMMING = r"data/Lampiran_3_Stemming.csv"

# =====================================================
# FILE ALGORITMA
# =====================================================

CM_NB = r"data/06_Hasil_Analisis/confusion_matrix_Naive_Bayes.png"

CM_KNN = r"data/06_Hasil_Analisis/confusion_matrix_KNN.png"

CM_DT = r"data/06_Hasil_Analisis/confusion_matrix_Decision_Tree.png"

REPORT_NB = r"data/06_Hasil_Analisis/classification_report_Naive_Bayes.csv"

REPORT_KNN = r"data/06_Hasil_Analisis/classification_report_KNN.csv"

REPORT_DT = r"data/06_Hasil_Analisis/classification_report_Decision_Tree.csv"

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FINAL)

@st.cache_data
def load_hotel():

    try:

        df = pd.read_csv(
            DATA_MENTAH,
            sep=";",
            encoding="utf-8-sig",
            on_bad_lines="skip"
        )

    except:

        df = pd.read_csv(
            DATA_MENTAH,
            engine="python",
            on_bad_lines="skip"
        )

    return df

@st.cache_data
def load_model():
    return pd.read_csv(HASIL_MODEL)

@st.cache_data
def load_topic():
    return pd.read_csv(BERTOPIC)


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    selected = option_menu(
        "Menu",
        [
            "Dashboard Sentimen",
            "BERTopic",
            "Perbandingan Algoritma",
            "Hasil Analisis"

        ],
        icons=[
            "bar-chart",
            "file-text",
            "graph-up",
            "file-earmark-text",

        ],
        default_index=0
    )


# =====================================================
# HOME
# =====================================================

if selected == "Home":

    st.title(
        "🏨 Analisis Sentimen Ulasan Hotel di Sulawesi Utara"
    )

    st.subheader(
        "Perbandingan Algoritma K-NN, Naïve Bayes, dan Decision Tree Menggunakan TF-IDF dan BERTopic"
    )

    st.markdown("---")

    st.markdown("""
    <div style='text-align: justify;'>

    Penelitian ini bertujuan untuk membandingkan performa
    algoritma K-Nearest Neighbor (K-NN), Naïve Bayes,
    dan Decision Tree dalam melakukan klasifikasi sentimen
    ulasan hotel di Sulawesi Utara.

    Data ulasan diperoleh dari platform TripAdvisor dan
    diproses melalui tahapan preprocessing teks, ekstraksi
    fitur menggunakan TF-IDF, klasifikasi sentimen,
    serta pemodelan topik menggunakan BERTopic.

    Hasil penelitian menunjukkan bahwa algoritma
    Decision Tree memberikan performa yang paling
    seimbang dalam menangani data yang tidak seimbang
    (class imbalance), sehingga dipilih sebagai model
    terbaik untuk proses analisis sentimen.

    </div>
    """,
    unsafe_allow_html=True)

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Jumlah Ulasan",
            "2.330"
        )

    with col2:
        st.metric(
            "Sentimen Positif",
            "1.803"
        )

    with col3:
        st.metric(
            "Sentimen Negatif",
            "527"
        )

    with col4:
        st.metric(
            "Model Terbaik",
            "Decision Tree"
        )



    st.subheader("Hasil Utama Penelitian")

    st.success(
        "Decision Tree memperoleh performa paling seimbang dengan akurasi 84,65%, Recall Negatif 0,61, dan F1-Score Negatif 0,65."
    )




# =====================================================
# DASHBOARD SENTIMEN
# =====================================================

elif selected == "Dashboard Sentimen":

    st.title("📊 Dashboard Sentimen")

    hotel = load_hotel()
    df = load_data()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Ringkasan Dataset",
        "🗂️ Dataset",
        "⚙️ Preprocessing",
        "🏷️ Distribusi Label"
        
    ])

    # =================================================
    # TAB 1
    # =================================================

    # =================================================
    # TAB 1
    # =================================================

    with tab1:

        df_final = pd.read_csv(
            r"data/06_FINALS/data_final_terlabel.csv"
        )

        data_awal = len(hotel)
        data_final = len(df_final)

        positif = (
            df_final["label"] == "Positif"
        ).sum()

        negatif = (
            df_final["label"] == "Negatif"
        ).sum()

        data_terhapus = (
            data_awal - data_final
        )

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Data Awal",
                f"{data_awal:,}"
            )

        with col2:
            st.metric(
                "Data Final",
                f"{data_final:,}"
            )

        with col3:
            st.metric(
                "Data Terhapus",
                f"{data_terhapus:,}"
            )

        with col4:
            st.metric(
                "Positif",
                f"{positif:,}"
            )

        with col5:
            st.metric(
                "Negatif",
                f"{negatif:,}"
            )

        st.divider()

        # ============================================
        # DISTRIBUSI SENTIMEN
        # ============================================

        st.subheader(
            "Distribusi Sentimen"
        )

        fig, ax = plt.subplots(
            figsize=(8,5)
        )

        sns.countplot(
            data=df_final,
            x="label",
            palette="Set2",
            ax=ax
        )

        for container in ax.containers:
            ax.bar_label(container)

        st.pyplot(fig)

        st.divider()

        # ============================================
        # DISTRIBUSI RATING
        # ============================================

        st.subheader(
            "Distribusi Rating"
        )

        fig, ax = plt.subplots(
            figsize=(8,5)
        )

        sns.countplot(
            data=df_final,
            x="rating",
            palette="Blues",
            ax=ax
        )

        for container in ax.containers:
            ax.bar_label(container)

        st.pyplot(fig)

        st.divider()

        # ============================================
        # PIE CHART SENTIMEN
        # ============================================

        st.subheader(
            "Persentase Sentimen"
        )

        sentimen = (
            df_final["label"]
            .value_counts()
        )

        fig, ax = plt.subplots(
            figsize=(7,7)
        )

        ax.pie(
            sentimen.values,
            labels=sentimen.index,
            autopct="%1.1f%%",
            startangle=90
        )

        st.pyplot(fig)

        st.divider()

        # ============================================
        # RINGKASAN DATASET
        # ============================================

        st.subheader(
            "Ringkasan Dataset"
        )

        ringkasan = pd.DataFrame({

            "Keterangan": [

                "Jumlah Data Awal",
                "Jumlah Data Final",
                "Data Terhapus",
                "Sentimen Positif",
                "Sentimen Negatif"

            ],

            "Jumlah": [

                data_awal,
                data_final,
                data_terhapus,
                positif,
                negatif

            ]

        })

        st.dataframe(
            ringkasan,
            use_container_width=True
        )

    # =================================================
    # TAB 2
    # =================================================

    with tab2:

            st.subheader(
                "Dataset Mentah"
            )

            df_mentah = pd.read_csv(
                r"data/01_Data_Mentah/hotel_sulut.csv",
                sep=";",
                encoding="utf-8-sig",
                on_bad_lines="skip"
            )

            st.metric(
                "Jumlah Data Mentah",
                f"{len(df_mentah):,}"
            )

            st.dataframe(
                df_mentah,
                use_container_width=True,
                height=400
            )

            st.divider()

            st.subheader(
                "Dataset Final (Setelah Filtering)"
            )

            df_final = pd.read_csv(
                r"data/06_FINALS/data_final_terlabel.csv"
            )

            st.metric(
                "Jumlah Data Final",
                f"{len(df_final):,}"
            )

            st.dataframe(
                df_final,
                use_container_width=True,
                height=400
            )
            

    # =================================================
    # TAB 3
    # =================================================

    with tab3:

        # =============================================
        # CASE FOLDING & FILTERING
        # =============================================

        st.subheader(
            "Case Folding & Filtering"
        )

        df_case = pd.read_csv(
            CASEFOLDING
        )

        st.dataframe(
            df_case,
            use_container_width=True,
            height=400
        )

        st.divider()

        # =============================================
        # TOKENIZATION
        # =============================================

        st.subheader(
            "Tokenization"
        )

        df_token = pd.read_csv(
            TOKEN
        )

        st.dataframe(
            df_token,
            use_container_width=True,
            height=400
        )

        st.divider()

        # =============================================
        # STEMMING
        # =============================================

        st.subheader(
            "Stemming"
        )

        df_stem = pd.read_csv(
            STEMMING
        )

        st.dataframe(
            df_stem,
            use_container_width=True,
            height=400
        )

        st.divider()

        # =============================================
        # DATA SETELAH LABELING
        # =============================================

        st.subheader(
            "Labeling"
        )

        df_label = pd.read_csv(
            r"data/06_FINALS/data_final_terlabel.csv"
        )

        st.dataframe(
            df_label,
            use_container_width=True,
            height=400
        )

    # =================================================
    # TAB 4 - LABELING
    # =================================================

    with tab4:

        st.subheader(
            "Distribusi Label Sentimen"
        )

        df_label = pd.read_csv(
            r"data/06_FINALS/data_final_terlabel.csv"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Jumlah Data",
                f"{len(df_label):,}"
            )

        with col2:
            st.metric(
                "Positif",
                f"{(df_label['label'] == 'Positif').sum():,}"
            )

        with col3:
            st.metric(
                "Negatif",
                f"{(df_label['label'] == 'Negatif').sum():,}"
            )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            distribusi = (
                df_label["label"]
                .value_counts()
                .reset_index()
            )

            distribusi.columns = [
                "Label",
                "Jumlah"
            ]

            st.dataframe(
                distribusi,
                use_container_width=True
            )

        with col2:

            fig, ax = plt.subplots(
                figsize=(5,4)
            )

            sns.countplot(
                data=df_label,
                x="label",
                palette="Set2",
                ax=ax
            )

            for container in ax.containers:
                ax.bar_label(container)

            st.pyplot(fig)

        st.divider()

        st.subheader(
            "Dataset Final Berlabel"
        )

        st.dataframe(
            df_label,
            use_container_width=True,
            height=400
        )



# =====================================================
# ANALISIS HOTEL
# =====================================================

elif selected == "Analisis Hotel":

    st.title("Analisis Hotel")

    hotel = load_hotel()

    st.subheader(
        "Top 10 Hotel Dengan Jumlah Ulasan Terbanyak"
    )

    # Cek kolom nama hotel
    if "placeInfo/name" in hotel.columns:

        top_hotel = (
            hotel["placeInfo/name"]
            .astype(str)
            .value_counts()
            .head(10)
            .reset_index()
        )

        top_hotel.columns = [
            "Hotel",
            "Jumlah Ulasan"
        ]

        st.dataframe(top_hotel)

        fig, ax = plt.subplots(figsize=(10,6))
        
        sns.barplot(
            data=top_hotel,
            y="Hotel",
            x="Jumlah Ulasan",
            ax=ax
        )

        st.pyplot(fig)

    else:

        st.error(
            "Kolom placeInfo/name tidak ditemukan"
        )

    st.subheader("Distribusi Rating")

    fig, ax = plt.subplots(figsize=(8,5))

    sns.countplot(
        data=hotel,
        x="rating",
        ax=ax
    )

    st.pyplot(fig)

# =====================================================
# BERTOPIC
# =====================================================

elif selected == "BERTopic":

    import streamlit.components.v1 as components

    st.title("🧠 Analisis Topik BERTopic")

    folder = r"data/06_Hasil_Analisis"

    def tampilkan_html(path_file, tinggi=800):

        try:

            with open(path_file, "r", encoding="utf-8") as f:

                html_content = f.read()

            components.html(
                html_content,
                height=tinggi,
                scrolling=True
            )

        except Exception as e:

            st.error(f"Gagal membuka file: {e}")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Topik Dominan",
        "🏆 Topik Teratas",
        "🗺️ Peta Jarak",
        "🔥 Heatmap",
        "🌳 Hierarki",
        "⚠️ Pain Points",
        "💪 Strengths",
        "📈 Sentimen per Topik"
    ])

    # =================================================
    # TOPIK DOMINAN
    # =================================================

    with tab1:

        st.subheader("10 Topik Paling Banyak Dibahas")

        topic = load_topic()

        topik = (
            topic["topic_name"]
            .value_counts()
            .head(10)
            .reset_index()
        )

        topik.columns = ["Topik", "Jumlah"]

        st.dataframe(
            topik,
            use_container_width=True
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.barplot(
            data=topik,
            x="Jumlah",
            y="Topik",
            ax=ax
        )

        st.pyplot(fig)

    # =================================================
    # TOPIK TERATAS
    # =================================================

    with tab2:

        st.subheader("Visualisasi Topik Teratas")

        tampilkan_html(
            rf"{folder}/topik_teratas.html",
            tinggi=900
        )

    # =================================================
    # PETA JARAK TOPIK
    # =================================================

    with tab3:

        st.subheader("Intertopic Distance Map")

        tampilkan_html(
            rf"{folder}/peta_jarak_topik.html",
            tinggi=950
        )

    # =================================================
    # HEATMAP TOPIK
    # =================================================
    with tab4:
        
        st.subheader("Heatmap Similarity Matrix")

        tampilkan_html(
            rf"{folder}/heatmap_topik.html",
            tinggi=950
        )

        st.image(
            rf"{folder}/similarity_matrix_final.png",
            use_container_width=True
        )

        st.info(
            """
            Similarity Matrix digunakan untuk melihat tingkat kemiripan
            antar topik yang dihasilkan oleh model BERTopic.

            Semakin tinggi nilai similarity maka semakin dekat hubungan
            antar topik tersebut. Sebaliknya, nilai similarity yang rendah
            menunjukkan bahwa topik berhasil dipisahkan dengan baik dan
            merepresentasikan aspek layanan yang berbeda.
            """
        )
    # =================================================
    # HIERARKI TOPIK
    # =================================================

    with tab5:

        st.subheader("🌳 Hierarki Topik")

        tampilkan_html(
            rf"{folder}/hierarki_topik.html",
            tinggi=900
        )


    # =================================================
    # PAIN POINTS
    # =================================================

    with tab6:

        st.subheader("Pain Points Pelanggan")

        st.image(
            rf"{folder}/pain_points_final.png",
            use_container_width=True
        )

        st.image(
            rf"{folder}/pain_points_final2.png",
            use_container_width=True
        )

    # =================================================
    # STRENGTHS
    # =================================================

    with tab7:

        st.subheader("Strengths Hotel")

        st.image(
            rf"{folder}/strengths_final.png",
            use_container_width=True
        )

        st.image(
            rf"{folder}/strengths_final3.png",
            use_container_width=True
        )

        with tab8:

            st.subheader(
                "Distribusi Sentimen pada Setiap Topik"
            )

            topic = load_topic()

            crosstab = pd.crosstab(
                topic["topic_id"],
                topic["label"]
            )

            fig, ax = plt.subplots(
                figsize=(10,6)
            )

            crosstab.plot(
                kind="bar",
                stacked=True,
                color=[
                    "#ff6f61",
                    "#6bbfb0"
                ],
                ax=ax
            )

            ax.set_title(
                "Distribusi Sentimen per Topik"
            )

            ax.set_xlabel(
                "Topik"
            )

            ax.set_ylabel(
                "Jumlah Ulasan"
            )

            ax.legend(
                title="Sentimen"
            )

            plt.xticks(
                rotation=45
            )

            st.pyplot(fig)

# =====================================================
# PERBANDINGAN ALGORITMA
# =====================================================

elif selected == "Perbandingan Algoritma":

    st.title("⚖️ Perbandingan Algoritma")

    model = load_model()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Classification Report",
        "🔳 Confusion Matrix",
        "☁️ WordCloud",
        "📊 Hasil Perbandingan"
    ])

    # =================================================
    # TAB 1 - CLASSIFICATION REPORT
    # =================================================

    with tab1:

        # =================================================
        # NAIVE BAYES
        # =================================================

        st.subheader("Classification Report - Naïve Bayes")

        df_nb = pd.DataFrame({
            "precision": [1.00, 0.78, 0.89, 0.83],
            "recall": [0.04, 1.00, 0.52, 0.78],
            "f1-score": [0.07, 0.87, 0.47, 0.69],
            "support": [108, 361, 469, 469]
        },
        index=[
            "Negatif",
            "Positif",
            "macro avg",
            "weighted avg"
        ])

        st.dataframe(df_nb, use_container_width=True)

        st.metric(
            "Accuracy Naïve Bayes",
            "77.83%"
        )

        st.divider()

        # =================================================
        # KNN
        # =================================================

        st.subheader("Classification Report - K-NN")

        df_knn = pd.DataFrame({
            "precision": [0.81, 0.86, 0.83, 0.85],
            "recall": [0.46, 0.97, 0.71, 0.85],
            "f1-score": [0.59, 0.91, 0.75, 0.84],
            "support": [108, 361, 469, 469]
        },
        index=[
            "Negatif",
            "Positif",
            "macro avg",
            "weighted avg"
        ])

        st.dataframe(df_knn, use_container_width=True)

        st.metric(
            "Accuracy K-NN",
            "85.07%"
        )

        st.divider()

        # =================================================
        # DECISION TREE
        # =================================================

        st.subheader("Classification Report - Decision Tree")

        df_dt = pd.DataFrame({
            "precision": [0.69, 0.89, 0.79, 0.84],
            "recall": [0.61, 0.92, 0.76, 0.85],
            "f1-score": [0.65, 0.90, 0.77, 0.84],
            "support": [108, 361, 469, 469]
        },
        index=[
            "Negatif",
            "Positif",
            "macro avg",
            "weighted avg"
        ])

        st.dataframe(df_dt, use_container_width=True)

        st.metric(
            "Accuracy Decision Tree",
            "84.65%"
        )
    
# =================================================
# TAB 2 - CONFUSION MATRIX
# =================================================

    with tab2:

        st.subheader("Confusion Matrix Naïve Bayes")

        st.image(
            CM_NB,
            use_container_width=True
        )

        st.divider()

        st.subheader("Confusion Matrix K-NN")

        st.image(
            CM_KNN,
            use_container_width=True
        )

        st.divider()

        st.subheader("Confusion Matrix Decision Tree")

        st.image(
            CM_DT,
            use_container_width=True
        )

# ============================================
# TAB 3 - WORDCLOUD
# ============================================

    with tab3:

        st.image(
            r"data/03_Hasil_Analisis/wordcloud_positif.png",
            use_container_width=True
        )

        st.divider()

        st.image(
            r"data/03_Hasil_Analisis/wordcloud_negatif.png",
            use_container_width=True
        )

# =================================================
# TAB 4 - GRAFIK
# =================================================

    with tab4:

        st.subheader("Tabel Hasil Perbandingan Algoritma")

        st.dataframe(
            model,
            use_container_width=True
        )

        terbaik = model.loc[
            model["Akurasi"].idxmax()
        ]

        st.subheader("Grafik Hasil Perbandingan Algoritma")

        metrics_df = pd.DataFrame({
            "Model": [
                "Naïve Bayes",
                "K-NN",
                "Decision Tree"
            ],
            "Akurasi": [
                0.7783,
                0.8507,
                0.8465
            ],
            "Precision": [
                1.0000,
                0.8065,
                0.6875
            ],
            "Recall": [
                0.0370,
                0.4630,
                0.6111
            ],
            "F1-Score": [
                0.0714,
                0.5882,
                0.6471
            ]
        })

        metrics_melt = metrics_df.melt(
            id_vars="Model",
            var_name="Metrik",
            value_name="Nilai"
        )

        fig, ax = plt.subplots(
            figsize=(12,6)
        )

        sns.barplot(
            data=metrics_melt,
            x="Metrik",
            y="Nilai",
            hue="Model",
            palette=[
                "#4E79A7",   # Naive Bayes
                "#F28E2B",   # KNN
                "#59A14F"    # Decision Tree
            ],
            ax=ax
        )

        ax.set_title(
            "Perbandingan Performa Algoritma",
            fontsize=14,
            fontweight="bold"
        )

        ax.set_ylabel("Nilai")
        ax.set_xlabel("")
        ax.set_ylim(0, 1.15)

        # ============================
        # LABEL ANGKA DI ATAS BAR
        # ============================

        for container in ax.containers:

            labels = []

            for v in container.datavalues:

                labels.append(
                    f"{v*100:.2f}%"
                )

            ax.bar_label(
                container,
                labels=labels,
                fontsize=9,
                fontweight="bold"
            )

        ax.legend(
            title="Algoritma"
        )

        st.pyplot(fig)

        st.markdown(
            """
            <div style="
                background-color:#F8FAFC;
                padding:20px;
                border-radius:10px;
                border-left:6px solid #2563EB;
                text-align:justify;
                line-height:1.8;
                color:#1E293B;
                font-size:15px;
            ">

            <h4 style="margin-top:0;color:#2563EB;">
            📌 Perbandingan Algoritma
            </h4>

            Berdasarkan grafik perbandingan performa algoritma, terlihat bahwa
            <b>K-Nearest Neighbor (K-NN)</b> memperoleh nilai akurasi tertinggi
            sebesar <b>85,07%</b>, diikuti oleh <b>Decision Tree</b> sebesar
            <b>84,65%</b> dan <b>Naïve Bayes</b> sebesar <b>77,83%</b>.

            Pada metrik <b>Precision</b> untuk kelas negatif, Naïve Bayes
            memperoleh nilai tertinggi yaitu <b>100%</b>. Namun demikian,
            nilai <b>Recall</b> yang sangat rendah sebesar <b>3,70%</b>
            menunjukkan bahwa model hanya mampu mengenali sebagian kecil
            ulasan negatif yang terdapat dalam dataset. Kondisi ini
            mengindikasikan bahwa Naïve Bayes cenderung mengklasifikasikan
            sebagian besar ulasan ke dalam kelas positif sehingga kurang
            optimal dalam mendeteksi sentimen negatif.

            Algoritma <b>K-NN</b> menunjukkan performa yang baik dengan
            kombinasi akurasi tertinggi serta nilai Precision, Recall,
            dan F1-Score yang relatif tinggi. Model ini mampu mengenali
            ulasan positif dengan sangat baik, meskipun kemampuan dalam
            mendeteksi ulasan negatif masih berada pada tingkat sedang.

            Sementara itu, <b>Decision Tree</b> menghasilkan performa yang
            paling seimbang dalam mengklasifikasikan kedua kelas sentimen.
            Hal ini terlihat dari nilai Recall kelas negatif sebesar
            <b>61,11%</b> dan F1-Score sebesar <b>64,71%</b>, yang merupakan
            nilai tertinggi dibandingkan algoritma lainnya untuk kelas negatif.
            Hasil tersebut menunjukkan bahwa Decision Tree lebih mampu
            mengenali ulasan negatif tanpa terlalu mengorbankan performa
            pada kelas positif.

            Secara keseluruhan, <b>K-NN</b> memberikan nilai akurasi tertinggi,
            sedangkan <b>Decision Tree</b> menunjukkan keseimbangan performa
            yang lebih baik dalam mendeteksi sentimen positif maupun negatif.

            </div>
            """,
            unsafe_allow_html=True
)
        
        st.divider()

        st.subheader(
            "Grafik Akurasi Per Algoritma"
        )

        st.image(
            r"data/03_Hasil_Analisis/grafik_akurasi.png",
            use_container_width=True
        )

        st.divider()

        st.subheader(
            "Grafik F1-Score Per Kelas"
        )

        st.image(
            r"data/03_Hasil_Analisis/f1_score_per_kelas.png",
            use_container_width=True
        )



# =====================================================
# HASIL ANALISIS
# =====================================================

# =====================================================
# HASIL ANALISIS
# =====================================================

elif selected == "Hasil Analisis":

    st.title("📌 Hasil Analisis")

    st.success(
        "Decision Tree dipilih sebagai model terbaik berdasarkan evaluasi menyeluruh terhadap akurasi, recall, dan F1-Score pada dataset."
    )

    st.markdown("""
    <div style="text-align: justify;">

    <h3>Hasil Perbandingan Algoritma</h3>

    Berdasarkan proses pelatihan dan pengujian terhadap tiga algoritma
    klasifikasi sentimen, yaitu Naïve Bayes, K-Nearest Neighbor (K-NN),
    dan Decision Tree, diperoleh hasil performa yang berbeda pada setiap
    model. Naïve Bayes menghasilkan tingkat akurasi sebesar 77,83%,
    K-NN memperoleh akurasi tertinggi sebesar 85,07%, sedangkan
    Decision Tree menghasilkan akurasi sebesar 84,65%.

    Secara umum, nilai akurasi menunjukkan bahwa seluruh algoritma
    mampu melakukan klasifikasi sentimen dengan cukup baik. Akan tetapi,
    perbedaan performa mulai terlihat ketika dilakukan evaluasi yang
    lebih mendalam menggunakan metrik precision, recall, dan f1-score,
    khususnya pada kelas negatif yang memiliki jumlah data lebih sedikit
    dibandingkan kelas positif.

    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style="text-align: justify;">

    <h3>Pengaruh Ketidakseimbangan Data (Class Imbalance)</h3>

    Dataset ulasan hotel yang digunakan dalam penelitian ini memiliki
    distribusi kelas yang tidak seimbang (imbalanced data), di mana
    jumlah ulasan positif jauh lebih banyak dibandingkan ulasan negatif.
    Kondisi ini menyebabkan model klasifikasi cenderung lebih mudah
    mempelajari pola pada kelas mayoritas dan berisiko mengabaikan
    karakteristik kelas minoritas.

    Pada kondisi data yang tidak seimbang, penggunaan metrik akurasi
    sebagai satu-satunya indikator performa dapat menghasilkan
    interpretasi yang menyesatkan. Sebuah model dapat memperoleh
    akurasi tinggi hanya karena berhasil mengklasifikasikan sebagian
    besar data positif, meskipun gagal mendeteksi ulasan negatif.

    Oleh karena itu, penelitian ini tidak hanya menggunakan akurasi
    sebagai dasar evaluasi, tetapi juga mempertimbangkan nilai recall
    dan f1-score pada kelas negatif. Kedua metrik tersebut digunakan
    untuk mengukur kemampuan model dalam mengenali ulasan yang berisi
    kritik, keluhan, atau ketidakpuasan pelanggan terhadap layanan hotel.

    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style="text-align: justify;">

    <h3>Interpretasi Hasil Pengujian</h3>

    Hasil pengujian menunjukkan bahwa Naïve Bayes memperoleh akurasi
    sebesar 77,83%, namun mengalami kesulitan yang sangat signifikan
    dalam mendeteksi ulasan negatif. Hal ini terlihat dari nilai recall
    negatif yang hanya mencapai 0,04. Dengan kata lain, sebagian besar
    ulasan negatif gagal dikenali dan justru diprediksi sebagai ulasan
    positif. Kondisi tersebut menunjukkan adanya bias yang tinggi terhadap
    kelas mayoritas.

    Algoritma K-Nearest Neighbor (K-NN) berhasil menghasilkan akurasi
    tertinggi sebesar 85,07%. Hasil ini menunjukkan bahwa pendekatan
    berbasis kedekatan data mampu mengenali pola sentimen secara efektif.
    Namun demikian, kemampuan model dalam mendeteksi kelas negatif masih
    berada di bawah Decision Tree, sehingga performa yang dihasilkan
    belum sepenuhnya optimal untuk menangani ketidakseimbangan data.

    Sementara itu, algoritma Decision Tree menghasilkan akurasi sebesar
    84,65%, sedikit di bawah K-NN. Meskipun demikian, Decision Tree
    menunjukkan performa yang lebih seimbang dengan nilai Recall Negatif
    sebesar 0,61 dan F1-Score Negatif sebesar 0,65. Hasil tersebut
    menunjukkan bahwa model mampu mengenali ulasan negatif secara lebih
    konsisten dibandingkan dua algoritma lainnya.

    Kemampuan tersebut diperoleh karena Decision Tree membangun aturan
    keputusan (decision rules) berdasarkan fitur-fitur penting yang
    terbentuk dari representasi TF-IDF. Mekanisme ini memungkinkan model
    untuk menangkap pola-pola tekstual yang berkaitan dengan keluhan
    pelanggan secara lebih efektif dibandingkan pendekatan probabilistik
    maupun pendekatan berbasis jarak.

    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style="text-align: justify;">

    <h3>Model Terpilih</h3>

    Berdasarkan evaluasi komprehensif terhadap seluruh metrik performa,
    algoritma Decision Tree ditetapkan sebagai model terbaik dalam
    penelitian ini. Keputusan tersebut tidak hanya mempertimbangkan
    tingkat akurasi, tetapi juga mempertimbangkan kemampuan model dalam
    mendeteksi kelas minoritas yang menjadi fokus utama penelitian.

    Dengan akurasi sebesar 84,65%, Recall Negatif sebesar 0,61,
    dan F1-Score Negatif sebesar 0,65, Decision Tree menunjukkan
    keseimbangan performa yang paling baik di antara seluruh algoritma
    yang diuji. Hasil ini membuktikan bahwa Decision Tree merupakan
    model yang paling robust dalam menghadapi permasalahan class
    imbalance pada data ulasan hotel di Sulawesi Utara.

    Oleh karena itu, Decision Tree dipilih sebagai model klasifikasi
    utama karena mampu menghasilkan prediksi sentimen yang lebih
    representatif, objektif, dan reliabel dalam mengidentifikasi
    ulasan positif maupun negatif dari pelanggan hotel.

    </div>
    """, unsafe_allow_html=True)

