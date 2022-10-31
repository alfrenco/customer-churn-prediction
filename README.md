[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=8750431&assignment_repo_type=AssignmentRepo)

## Dataset

Data yang digunakan adalah data berisi informasi customer perusahaan Telco.Dataset yang digunakan bisa dilihat [disini](https://www.kaggle.com/blastchar/telco-customer-churn)

---

## Objective

Membuat model deep learning untuk menentukan apakah seorang customer akan churn atau tidak.

---

## Kesimpulan

Model yang dibuat untuk melakukan klasifikasi churn customer kali ini ada dua yaitu sequential model dan juga functional model. Kedua model dibentuk dengan arsitektur yang sama yaitu menggunakan 1 hidden layer dengan jumlah neuron 4 dan 1 output layer dengan 1 neuron. Kedua model menghasilkan performa yang cukup baik dilihat dari grafik loss dan accuracy saat training yang cukup baik dan juga tidak overfit. Untuk classification report saya mendapatkan accuracy yang sama untuk kedua model yaitu 0.82.

Model sequential dan functional yang sudah dibuat tadi dicoba untuk diimprove dengan mengganti hyperparameternya. Pada model sequential saya tambahkan jumlah hidden layer dari 1 hidden layer menjadi 3 hidden layer dimana pada hidden layer pertama ditambahkan initializer glorot, selain itu jumlah neuron juga diubah menjadi 32 pada hidden layer pertama, 16 pada hidden layer kedua, dan 8 pada hidden layer ketiga. Saya juga mengganti optimizer model dari adam menjadi nadam dan juga jumlah epoch dari 10 menjadi 20.  Perubahan yang sama dilakukan juga pada functional model dengan perbedaan pada functional model tidak menggunakan initializer glorot. Perubahan kedua arsitektur tersebut merubah jumlah **parameter** dari **189 menjadi 2145** untuk kedua model.

Hasil Improve :
* Dari classification report sequential kita bisa melihat bahwa ternyata ada peningkatan pada precision label 0 dari 0.85 menjadi 0.86 dan recall label 1 dari 0.55 menjadi 0.57.
* Dari classification report di functional kita bisa melihat terjadi improvement dimana pada recall label 0 meningkat dari 0.92 menjadi 0.93. Pada label 1 juga mengalami peningkatan pada precision dari 0.71 menjadi 0.72.


**Saran** :

Pada data ini kita berfokus pada label 0 dimana datanya lebih banyak sehingga kita bisa mendapatkan hasil precision dan recall yang sudah bagus, namun jika kita ingin lebih fokus ke label 1 atau orang yang churn disarankan untuk menghandling imbalance data terlebih dahulu sehingga precision dan recall label 1 bisa meningkat.

