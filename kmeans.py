import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# Membaca data dari CSV yang berada dalam folder yang sama dengan skrip Python
file_path = 'Tingkat Kemiskinan.csv'
df = pd.read_csv(file_path)

# Menentukan kolom yang akan digunakan
columns_to_use = ['Persentase Penduduk Miskin', 'Tingkat Pengangguran']  

X = df[columns_to_use]

# Meminta input dari pengguna untuk jumlah cluster
jumlah_cluster = int(input("Masukkan jumlah cluster: "))

# Menerapkan algoritma K-Means
kmeans = KMeans(n_clusters=jumlah_cluster, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Pengaturan tema Seaborn
sns.set(style="whitegrid")

# Visualisasi hasil cluster dalam satu diagram
plt.figure(figsize=(8, 6))

# Menampilkan semua cluster dalam satu diagram
for cluster in range(jumlah_cluster):
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['Persentase Penduduk Miskin'], cluster_data['Tingkat Pengangguran'], label=f'Cluster {cluster}', cmap='viridis')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', label='Centroids')
    
plt.title('Hasil K-Means Clustering')
plt.xlabel('Persentase Penduduk Miskin')
plt.ylabel('Tingkat Pengangguran')
plt.legend()
plt.show()

#Nama : Sitti Nur Haliza
#NIM  : E1E120051