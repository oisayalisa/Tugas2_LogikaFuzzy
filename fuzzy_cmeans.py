import pandas as pd
import skfuzzy as fuzz
import seaborn as sns
import matplotlib.pyplot as plt

# Membaca data dari CSV yang berada dalam folder yang sama dengan skrip Python
file_path = 'Tingkat Kemiskinan.csv'
df = pd.read_csv(file_path)

# Menentukan kolom yang akan digunakan
columns_to_use = ['Persentase Penduduk Miskin', 'Tingkat Pengangguran']  # Ubah sesuai dengan nama kolom yang sebenarnya
X = df[columns_to_use].values.T  # Transpose data untuk memenuhi format input FCM

# Meminta input dari pengguna untuk jumlah cluster
jumlah_cluster = int(input("Masukkan jumlah cluster: "))

# Menerapkan algoritma Fuzzy C-Means
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X, c=jumlah_cluster, m=2, error=0.005, maxiter=1000, init=None)

# Menyimpan hasil ke dalam kolom Cluster
df['Cluster'] = u.argmax(axis=0)

# Pengaturan tema Seaborn
sns.set(style="whitegrid")

# Visualisasi hasil cluster dalam satu diagram
plt.figure(figsize=(8, 6))

# Menampilkan semua cluster dalam satu diagram
for cluster in range(jumlah_cluster):
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['Persentase Penduduk Miskin'], cluster_data['Tingkat Pengangguran'], label=f'Cluster {cluster}', cmap='viridis')

#plt.scatter(cntr[0], cntr[1], s=100, c='red', label='Centroids')
# Menampilkan posisi centroid untuk setiap cluster
plt.scatter(cntr[:, 0], cntr[:, 1], s=100, c='red', marker='X', label='Centroids')
    
plt.title('Hasil Fuzzy C-Means Clustering')
plt.xlabel('Persentase Penduduk Miskin')
plt.ylabel('Tingkat Pengangguran')
plt.legend()
plt.show()

#Nama : Sitti Nur Haliza
#NIM  : E1E120051