import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import tensorflow as tf
import numpy as np
import os
import datetime
import time

# Mendeklarasikan variabel global
cap = None
model = None
video_writer = None
filename = None
start_time = None
recording = False

# Fungsi untuk memuat model yang sudah dilatih
def load_model():
    global model
    model = tf.keras.models.load_model('model_merokok.h5')  # Ganti dengan path model Anda

# Fungsi untuk memprediksi dari kamera dan menyimpan video jika merokok terdeteksi
def predict_and_capture():
    global cap, model, video_writer, filename, start_time, recording
    cap = cv2.VideoCapture(0)  # Buka kamera default (biasanya index 0)

    def update_frame():
        global cap, video_writer, filename, start_time, recording
        ret, frame = cap.read()
        if ret:
            # Mengubah format gambar dari BGR (OpenCV) ke RGB (PIL)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            pil_image = ImageTk.PhotoImage(image=pil_image)

            # Update canvas dengan gambar dari kamera
            canvas.create_image(0, 0, anchor=tk.NW, image=pil_image)
            canvas.image = pil_image

            # Jika sedang merekam, hanya menulis frame ke video dan tidak melakukan prediksi
            if recording:
                video_writer.write(frame)

                # Periksa apakah rekaman telah mencapai batas waktu 10 detik
                if time.time() - start_time >= 10:
                    video_writer.release()
                    video_writer = None
                    recording = False
                    print(f"Perekaman selesai setelah 10 detik, video disimpan sebagai {filename}")
            else:
                # Mengubah ukuran gambar sesuai dengan ukuran input model
                image = cv2.resize(frame, (150, 150))  # Sesuaikan dengan ukuran input model Anda
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Normalisasi gambar dan siapkan untuk prediksi
                image = np.array(image) / 255.0
                image = np.expand_dims(image, axis=0)

                # Melakukan prediksi menggunakan model yang sudah dimuat
                prediction = model.predict(image)
                label = "Merokok" if prediction[0][0] > 0.5 else "Tidak Merokok"

                # Menyimpan video jika terdeteksi merokok
                if label == "Merokok":
                    # Inisialisasi VideoWriter jika belum ada
                    if not recording:
                        if not os.path.exists('captured_videos'):
                            os.makedirs('captured_videos')

                        # Buat nama file unik berdasarkan timestamp
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f'captured_videos/captured_video_{timestamp}.avi'

                        # Inisialisasi VideoWriter
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))

                        # Catat waktu mulai rekaman
                        start_time = time.time()
                        recording = True
                        print(f"Mulai merekam video: {filename}")

                # Update label prediksi pada layar
                result_label.config(text=f'Hasil: {label}')

            # Memanggil fungsi itu sendiri setiap 10 milidetik untuk memperbarui frame
            canvas.after(10, update_frame)
        else:
            # Jika tidak ada frame yang diperoleh dari kamera, panggil update_frame lagi setelah 10 milidetik
            canvas.after(10, update_frame)

    # Memanggil fungsi update_frame secara initial
    update_frame()

# Fungsi untuk menghentikan kamera dan menutup aplikasi
def stop_camera():
    global cap, video_writer
    if cap:
        cap.release()  # Menutup kamera
    if video_writer:
        video_writer.release()  # Menutup penyimpanan video
    root.destroy()  # Menutup jendela aplikasi

# Membuat GUI
root = tk.Tk()
root.title('Deteksi Merokok')

# Membuat frame untuk area kamera
frame_camera = tk.Frame(root)
frame_camera.pack()

# Membuat canvas untuk menampilkan gambar dari kamera
canvas = tk.Canvas(frame_camera, width=640, height=480)
canvas.pack()

# Label untuk menampilkan hasil prediksi dari kamera
result_label = tk.Label(root, text='')
result_label.pack(pady=10)

# Tombol untuk menghentikan kamera dan menutup aplikasi
stop_button = ttk.Button(root, text="Stop Aplikasi", command=stop_camera)
stop_button.pack(pady=20)

# Memuat model saat aplikasi dimulai
load_model()

# Memanggil fungsi untuk memprediksi dan menampilkan gambar dari kamera
predict_and_capture()

# Menjalankan aplikasi GUI
root.mainloop()