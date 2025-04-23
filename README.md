# Autoencoder-DeepLearning-3008

Pada percobaan ini melakukan Autoencoder untuk menaikkan exposure kecerahan gambar

**Dataset**

Pada dataset yang saya gunakan merupakan dataset image awan yang berisi kurang lebih 30 gambar

**Cara Kerja Code:**

pada saat code di running maka tampilan terminalnya:

**1. Input Gambar dari Pengguna**

Proses dimulai dengan meminta pengguna memasukkan path gambar yang ingin mereka tingkatkan kecerahannya.
Gambar yang dimasukkan harus dalam format PNG atau JPEG, dan secara otomatis dikonversi ke mode RGB menggunakan PIL.Image.
Jika file gambar tidak ditemukan, program akan memberikan notifikasi agar pengguna memasukkan path yang benar.

**2. Menentukan Tingkat Kecerahan**

Setelah gambar diinput, pengguna dapat memilih tingkat kecerahan yang diinginkan.
Program menggunakan input() untuk meminta angka, misalnya 1.8 atau 2.0, yang akan digunakan dalam ImageEnhance.Brightness().
Jika pengguna memasukkan angka di luar rentang yang direkomendasikan (misalnya terlalu kecil atau terlalu besar), mereka akan diminta untuk memasukkan angka yang lebih sesuai.

**Hasil terminal:**

**user diminta untuk menginputkan path gaambar yang telah tersedia di folder dataset**

**lalu user diminta menginputkan tingkat kecerahannya yang diinginkan:**
![image](https://github.com/user-attachments/assets/7fd5ba8f-e009-481f-93a4-a8db429136c3)

**Setelah itu, program akan mentraining dengan epoch 3000**
![image](https://github.com/user-attachments/assets/b199e7a3-b6d1-4c96-bc7d-26b0b4072fa4)

**Loss yang terus menurun menunjukkan bahwa model berhasil belajar meningkatkan kecerahan gambar**

**3. Pembuatan Dataset Target**

Program kemudian membuat gambar versi terang dari gambar asli berdasarkan nilai kecerahan yang dipilih oleh pengguna.
Proses ini dilakukan dengan ImageEnhance.Brightness() dari Pillow, yang meningkatkan tingkat pencahayaan pada gambar.
Gambar hasil ini disimpan ke direktori data/target dan nantinya digunakan sebagai target pelatihan model.

**NOTED : KEMUDIAN HASIL OUTPUTNYA AKAN SEPERTI INI DENGAN EPOCH 3000**
![image](https://github.com/user-attachments/assets/769e30f1-c1ac-4710-b6d7-644b8a60aa72)

**DAPAT DIKETAHUI BAHWA HASIL DARI MODEL TERSEBUT :**

**- INPUT : MERUPAKAN GAMBAR ASLI/NORMAL DARI DATASET, YAITU GAMABR YANG BELUM MENGALAMI PENINGKATAN KECERAHAN**

**- PREDICTED : MERUPAKAN GAMBAR YANG TELAH DIPROSES AUTOENCODER, DIMANA MODEL MENCOBA MENINGKATKAN KECERAHAN SESUAI POLA YANG TELAH DIPELAJARI DARI DATA PELATIHAN**

**- TARGET 1.5 : MERUPAKAN GAMBAR HASIL PENINGKATAN KECERAHAN DENGAN BRIGHTNESS FACTOR = 1.5**



**4. Arsitektur Model Autoencoder**
Proyek ini menggunakan Simple UNet, yaitu arsitektur autoencoder dengan dua tahap utama:
- Encoder:
Menggunakan dua lapisan konvolusi untuk menangkap fitur gambar.
Max pooling digunakan untuk mengecilkan ukuran gambar agar lebih mudah diproses.

- Decoder:
Upsampling dilakukan untuk mengembalikan ukuran gambar ke bentuk awal.
Lapisan akhir konvolusi menghasilkan gambar dengan kecerahan yang telah ditingkatkan.

**5. Proses Training Model**

Setelah dataset (gambar asli dan target terang) siap, model dilatih menggunakan MSELoss sebagai fungsi evaluasi.
Model dilatih selama 100 epoch, dengan optimasi menggunakan Adam Optimizer agar hasilnya lebih stabil.
Dalam setiap epoch, model mencoba memperbaiki prediksi gambar terang berdasarkan target yang telah dibuat sebelumny

**6. Evaluasi Performa Model**

Selama training, model mencetak nilai loss di setiap epoch menunjukkan bagaimana model semakin baik dalam memprediksi gambar terang.
Loss yang terus menurun menunjukkan bahwa model berhasil belajar meningkatkan kecerahan gambar

**7️. Prediksi dan Visualisasi**

Setelah training selesai, model digunakan untuk memprediksi gambar terang dari gambar asli yang diberikan pengguna.
Gambar input, prediksi model, dan target terang kemudian ditampilkan dalam matplotlib, sehingga pengguna bisa melihat perbandingan hasil.
**Program ini akan melihat perbandingan antara gambar asli, hasil prediksi model, dan target terang.**








