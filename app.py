import streamlit as st
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Judul Aplikasi Web
st.markdown('''
# **APPAREL ANALYZER : Implementasi Market Basket Analysis untuk Produk Fashion**
---

**Credit:** App built in `Python` + `Streamlit` by [Kelompok 2 ATLAS Teams]

---
''')
st.balloons()

image = Image.open('images/MBA.jpg')

st.image(image, caption='Gambar Ilustrasi Market Basket Analysis')

with st.sidebar.header('Upload your CSV data'):
    uploaded_files = st.file_uploader("Choose CSV file", type=["csv"])

st.markdown('''
Sistem Apparel Analyzer adalah sistem yang kami kembangkan untuk industri fashion dimana perusahaan akan menerapkan teknologi mutakhir untuk mengoptimalkan manajemen inventaris. 
Sistem ini memanfaatkan kecerdasan buatan (AI) dan analisis data yang canggih untuk mencapai efisiensi yang lebih besar dalam operasi kami.

Pastikan dataset yang diunggah mengandung kolom yang diperlukan, seperti 'transaction', 'Items', 'No transaksi', 'Jenis' dan 'Qty'; untuk hasil yang akurat.
''')

# Membaca file CSV
if uploaded_files is not None:
    data = pd.read_csv(uploaded_files)
    st.subheader("Tampilan Data Anda: ")
    st.write("Data from CSV file:")
    st.write(data)

    st.subheader("Nama - Nama Kolom:")
    st.write(data.columns.tolist())

    # Menampilkan nilai unik dari satu kolom saja
    column_name = 'Jenis'
    unique_values = data[column_name].unique()

    st.subheader(f"Nilai unik di Kolom '{column_name}':")
    for value in unique_values:
        st.write(value)

    if 'Items' in data.columns and 'Qty' in data.columns:
        product_quantity = data.groupby('Items')['Qty'].sum()
        top_products = product_quantity.sort_values(ascending=False).head(10)
        colors = plt.cm.viridis(range(len(top_products)))

        st.subheader('Grafik Produk Paling Diminati')
        fig, ax = plt.subplots()
        bars = ax.bar(top_products.index, top_products.values, color=colors)
        plt.xticks(rotation=45)
        plt.xlabel('Items')
        plt.ylabel('Frekuensi')

        for i, value in enumerate(top_products.values):
            ax.text(i, value + 1, str(value), ha='center', va='bottom')

        st.pyplot(fig)
    else:
        st.write("Columns 'Items' and 'Qty' not found in the uploaded file. Please check the column names.")

    st.subheader('Hasil Perhitungan Data Anda')
    transactions = [transaction.split(', ') for transaction in data['Items']]

    if transactions:
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        oht = pd.DataFrame(te_ary, columns=te.columns_)

        frequent_itemsets = apriori(oht, min_support=0.01, use_colnames=True)
        st.write("Frequent Itemsets :")
        st.markdown('''Frequent itemsets adalah himpunan item yang muncul bersama secara teratur dalam dataset transaksi atau dalam konteks market basket analysis. Dalam analisis keranjang belanja, frequent itemsets mengacu pada kombinasi item yang sering dibeli bersama-sama oleh pelanggan.''')
        st.write(frequent_itemsets)

        selected_columns = ['transaction', 'Items']
        df = data[selected_columns]

        # Loop through product categories for hot encoding
        product_categories = data['Jenis'].unique()
        for category in product_categories:
            basket_category = (data[data['Jenis'] == category]
                               .groupby(['No transaksi', 'Items'])['Qty']
                               .sum().unstack().reset_index().fillna(0)
                               .set_index('No transaksi'))

            basket_encoded = basket_category.applymap(lambda x: 1 if x > 0 else 0)

            frq_items = apriori(basket_encoded, min_support=0.01, use_colnames=True)
            rules = association_rules(frq_items, metric="lift", min_threshold=0.1)
            rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
            st.write(f"Model Association Rule Berdasarkan Jenis {category} :")
            st.write(rules.head())

    else:
        st.write("No transaction data found.")
else:
    st.write("No file uploaded or invalid file.")

# Susunan anggota kelompok
st.subheader("Kelompok 2 ATLAS")

# Path ke folder tempat gambar disimpan
folder_path = 'C:/Users/Lenovo/OneDrive/Documents/Apparel Analyzer/images'

def tampilkan_jejeran_gambar(gambar_paths, ukuran=720):
    col1, col2, col3, col4, col5 = st.columns(5)

    # Tampilkan gambar di setiap kolom
    with col1:
        img1 = Image.open(gambar_paths[0])
        st.image(img1, width=ukuran, use_column_width=True)
        st.markdown(
            f'<p style="text-align:center; font-family:poppins;">Nadia</p>',
            unsafe_allow_html=True)
        st.markdown(f'<style>div.stImage img {{width: {ukuran}px; height: {ukuran}px;}}</style>', unsafe_allow_html=True)

    with col2:
        img2 = Image.open(gambar_paths[1])
        st.image(img2, width=ukuran, use_column_width=True)
        st.markdown(
            f'<p style="text-align:center; font-family:poppins;">Fena</p>',
            unsafe_allow_html=True)
        st.markdown(f'<style>div.stImage img {{width: {ukuran}px; height: {ukuran}px;}}</style>', unsafe_allow_html=True)

    with col3:
        img3 = Image.open(gambar_paths[2])
        st.image(img3, width=ukuran, use_column_width=True)
        st.markdown(
            f'<p style="text-align:center; font-family:poppins;">Yafi</p>',
            unsafe_allow_html=True)
        st.markdown(f'<style>div.stImage img {{width: {ukuran}px; height: {ukuran}px;}}</style>', unsafe_allow_html=True)

    with col4:
        img4 = Image.open(gambar_paths[3])
        st.image(img4, width=ukuran, use_column_width=True)
        st.markdown(
            f'<p style="text-align:center; font-family:poppins;">Tohir</p>',
            unsafe_allow_html=True)
        st.markdown(f'<style>div.stImage img {{width: {ukuran}px; height: {ukuran}px;}}</style>', unsafe_allow_html=True)

    with col5:
        img5 = Image.open(gambar_paths[4])
        st.image(img5, width=ukuran, use_column_width=True)
        st.markdown(
            f'<p style="text-align:center; font-family:poppins;">Hilda</p>',
            unsafe_allow_html=True)
        st.markdown(f'<style>div.stImage img {{width: {ukuran}px; height: {ukuran}px;}}</style>', unsafe_allow_html=True)

# Daftar path gambar yang ingin ditampilkan
gambar_list = [
    'images/Nadia.jpg',
    'images/fena.jpg',
    'images/yafi.jpeg',
    'images/tohir.jpeg',
    'images/hilda.jpg'
]

# Panggil fungsi untuk menampilkan 5 gambar bersebelahan dengan ukuran yang sama
tampilkan_jejeran_gambar(gambar_list)
