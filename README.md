# Projek Game PCV
Ahmad Hanif Al-Fatih
5024241048

# Snake Camera Game

Snake Camera Game adalah game ular yang dimainkan melalui webcam laptop. Kepala ular mengikuti objek berwarna merah, misalnya sarung tangan merah, yang digerakkan di depan kamera. Projek ini sendiri memakai teknik pengolahan citra klasik dengan OpenCV.

## Cara Menjalankan

Pertama Pastikan Python 3.8 atau lebih baru dan webcam sudah tersedia. Install library yang dibutuhkan dengan perintah `pip install opencv-python numpy`, lalu mainkan game dengan `py main.py`. Program akan membuka dua windows, yaitu "Deteksi Kamera" yang menampilkan feed webcam beserta penanda objek merah, dan "Snake Game" yang menampilkan gamenya.

## Cara Bermain

Mulail dengan mengeklik objek merah di window Deteksi Kamera untuk mengkalibrasi warna, lalu pastikan lingkaran hijau mengikuti objeknya. Tekan SPASI untuk memulai, kemudian gerakkan objek untuk mengarahkan ular. Ular akan memanjang dan score bertambah 10 setiap memakan apel, sedangkan memakan bom membuat ular memendek dan score berkurang 10. Di layar selalu ada 3 apel dan 1 bom, dan apel yang dimakan langsung diganti dengan apel baru. Bom akan hilang setelah 5 detik lalu muncul lagi di tempat lain. Ular akan KO jika menabrak dinding di tepi layar atau menabrak badannya sendiri, dan kamu bisa main lagi dengan menekan R.

## Kontrol Permainan
Tekan M untuk melihat mask, dan pastikan hanya objek merah yang menyala putih. Jika bibir atau kulit ikut terdeteksi, naikkan batas saturasi dengan tombol +. Jika objek tidak terdeteksi, turunkan batas saturasi dengan - atau batas kecerahan.

## Konsep Pengolahan Citra

Setiap frame kamera diproses dengan Gaussian blur untuk mengurangi noise, lalu dikonversi dari BGR ke HSV agar warna lebih gampang dipisahkan dari kecerahan. Warna merah kemudian disegmentasi dengan thresholding `inRange`. Karena merah berada di kedua ujung hue, dua rentang digabung. Hasilnya dibersihkan dengan operasi morfologi (erosi lalu dilasi), kemudian titik pusat objek terbesar dicari menggunakan `findContours` dan `moments`. Terakhir, posisi dihaluskan dengan rata-rata bergerak supaya gerakan ular tidak patah-patah.

## Tampilan Permainan

<img width="999" height="413" alt="image" src="https://github.com/user-attachments/assets/e4ffb8f3-f776-4272-a325-0fdbf0fd3259" />

<img width="1002" height="434" alt="image" src="https://github.com/user-attachments/assets/3a3b8aeb-ebca-4e21-9575-d354fb743505" />
