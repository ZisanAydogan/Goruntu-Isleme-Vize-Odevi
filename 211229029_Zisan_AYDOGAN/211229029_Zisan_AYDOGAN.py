import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd 


# Sigmoid fonksiyonu
def sigmoid(x, a=1):
    return 1 / (1 + np.exp(-a*x))

# Yatay kaydırılmış sigmoid fonksiyonu
def shifted_sigmoid(x, a=1, b=0):
    return 1 / (1 + np.exp(-a*(x - b)))

# Eğimli sigmoid fonksiyonu
def tilted_sigmoid(x, a=1, b=0, c=1):
    return 1 / (1 + np.exp(-a*(x - b))) + c

# S-Curve kontrast arttırma fonksiyonu
def s_curve_contrast(image_path, func, **kwargs):
    # Görüntüyü yükle
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Normalize et
    normalized_image = image / 255.0  

    # İşlevi görüntü üzerine uygula
    transformed_image = func(normalized_image, **kwargs)

    # İşlem sonrası görüntüyü [0, 255] aralığına geri getir
    transformed_image = (transformed_image * 255).astype(np.uint8)

    return transformed_image

# Kontrast arttırma fonksiyonu
def contrast_enhancement(image_path):
    # Görüntüyü yükle
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Adaptif histogram eşitleme işlemini uygula
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))  
    enhanced_image = clahe.apply(image)

    return enhanced_image

class AnaSayfa(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ödev Takip Uygulaması")
        self.geometry("400x400")

        self.menu = tk.Menu(self)
        self.config(menu=self.menu)

        # Ana Sayfa menüsü
        self.ana_sayfa_menu = tk.Menu(self.menu, tearoff=False)
        self.ana_sayfa_menu.add_command(label="Ana Sayfa", command=self.ana_sayfa)
        self.menu.add_cascade(label="Ana Sayfa", menu=self.ana_sayfa_menu)

        # Ödev menüleri
        self.odev1_menu = tk.Menu(self.menu, tearoff=False)
        self.menu.add_cascade(label="   Ödev 1", menu=self.odev1_menu)
        self.odev1_menu.add_command(label="Detayları Göster", command=lambda: self.odev1_detaylari("Ödev 1: Temel İşlevselliği Oluştur"))

        self.odev2_menu = tk.Menu(self.menu, tearoff=False)
        self.menu.add_cascade(label="   Vize Ödevi", menu=self.odev2_menu)
        self.odev2_menu.add_command(label="S-Curve", command=self.s_curve_uygula)
        self.odev2_menu.add_command(label="Yoldaki Çizgileri Bulma", command=self.yol_cizgileri_bulma)
        self.odev2_menu.add_command(label="Yüz ve Göz Bulma", command=self.göz_bulma)
        self.odev2_menu.add_command(label="Deblurring", command= self.deblurring)
        self.odev2_menu.add_command(label="Resimdeki nesneler", command=self.nesne_bul)

        self.odev3_menu = tk.Menu(self.menu, tearoff=False)
        self.menu.add_cascade(label="    Ödev 3", menu=self.odev3_menu)
        self.odev3_menu.add_command(label="Detayları Göster", command=lambda: self.odev3_detaylari("Ödev 3: "))

        self.odev4_menu = tk.Menu(self.menu, tearoff=False)
        self.menu.add_cascade(label="   Ödev 4", menu=self.odev4_menu)
        self.odev4_menu.add_command(label="Detayları Göster", command=lambda: self.odev4_detaylari("Ödev 4: "))

        # Ödev 1 penceresi
        self.detaylar_penceresi = None

        # Başlık
        self.baslik_label = tk.Label(self, text="", font=("Arial", 14))
        self.baslik_label.pack(pady=10)

        # Ana sayfa içeriği
        self.ana_sayfa()
        
    def s_curve_uygula(self):
        image_path = 'C:/Users/Zisann/Desktop/bulanik.jpg'
        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        enhanced_image = contrast_enhancement(image_path)
    
        # Standart Sigmoid Fonksiyonu ile işlem
        selected_function = sigmoid
        function_params = {'a': 5}  
        sigmoid_transformed_image = s_curve_contrast(image_path, selected_function, **function_params)
    
        # Yatay Kaydırılmış Sigmoid Fonksiyonu ile işlem
        selected_function = shifted_sigmoid
        function_params_shifted = {'a': 10, 'b': 0.5}  
        shifted_sigmoid_transformed_image = s_curve_contrast(image_path, selected_function, **function_params_shifted)
    
        # Eğimli Sigmoid Fonksiyonu ile işlem
        selected_function = tilted_sigmoid
        function_params = {'a': 18, 'b': 0.5, 'c': 0.1}  
        tilted_sigmoid_transformed_image = s_curve_contrast(image_path, selected_function, **function_params)
    
        # Resimleri göster
        plt.figure(figsize=(20, 10))
    
        plt.subplot(2, 3, 1)
        plt.imshow(original_image, cmap='gray')
        plt.title('Orijinal Görüntü')
    
        plt.subplot(2, 3, 2)
        plt.imshow(enhanced_image, cmap='gray')
        plt.title('Kendi Eşitleme Yöntemi ile Güçlendirilmiş Görüntü')
    
        plt.subplot(2, 3, 4)
        plt.imshow(sigmoid_transformed_image, cmap='gray')
        plt.title('Standart Sigmoid ile Güçlendirilmiş Görüntü')
    
        plt.subplot(2, 3, 5)
        plt.imshow(shifted_sigmoid_transformed_image, cmap='gray')
        plt.title('Yatay Kaydırılmış Sigmoid ile Güçlendirilmiş Görüntü')
    
        plt.subplot(2, 3, 6)
        plt.imshow(tilted_sigmoid_transformed_image, cmap='gray')
        plt.title('Eğimli Sigmoid ile Güçlendirilmiş Görüntü')
    
        plt.show()  
        
    def yol_cizgileri_bulma(self):            
        # Görüntüyü yükle
        image = cv2.imread("road.jpg")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Kenar tespiti
        edges = cv2.Canny(gray, 50, 150)
        
        # Hough Dönüşümü
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=50)
        
        # Tüm çizgileri çizme
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Sonucu görselleştirme
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Road Lines Detection")
        plt.axis('off')
        plt.show()
        
    def göz_bulma(self):
        # Görüntüyü yükle
        image = cv2.imread("face.jpg")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Yüz tespiti için sınıflandırıcıyı yükle
        face_cascade = cv2.CascadeClassifier("C:\\Users\\Zisann\\anaconda3\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml")

        # Göz tespiti için sınıflandırıcıyı yükle
        eye_cascade = cv2.CascadeClassifier("C:\\Users\\Zisann\\anaconda3\\Library\\etc\\haarcascades\\haarcascade_eye.xml")

        # Yüz tespiti
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Tüm yüzleri çizme
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]
            
            # Gözleri tespit etme
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # Görüntüyü Matplotlib ile gösterme
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Face with Eyes Detection")
        plt.axis('off')
        plt.show()
        
    def deblurring(self):
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt

        def sharpen_image(image, kernel_size=(5, 5)):
            # Kenar belirginleştirme için filtre oluşturma
            kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
            sharpened_image = cv2.filter2D(image, -1, kernel)
            return sharpened_image

        # Görüntüyü yükle
        image = cv2.imread("blurlu.jpg")

        # Görüntüyü keskinleştir
        sharpened_image = sharpen_image(image)

        # Görüntüleri görselleştirme
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB))
        plt.title("Sharpened Image")
        plt.axis('off')

        plt.show()
        
    
    def nesne_bul(self):
        file_path = "image.jpg"
        # Resmi işle
        image = cv2.imread(file_path)
        # Renk filtresi uygula (koyu yeşil)
        lower_green = np.array([0, 100, 0])
        upper_green = np.array([100, 255, 100])
        mask = cv2.inRange(image, lower_green, upper_green)
        # Konturları bul
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Özellikleri saklamak için boş bir DataFrame oluştur
        data = []
        # Her kontur için işlem yap
        for i, contour in enumerate(contours):
            # Alanı hesapla
            area = cv2.contourArea(contour)
            # Konturun dış dikdörtgenini al
            x, y, w, h = cv2.boundingRect(contour)
            # Diagonal hesapla
            diagonal = np.sqrt(w*2 + h*2)
            # Momentleri hesapla
            moments = cv2.moments(contour)
            # Momentlardan enerji ve entropi hesapla
            energy = -1 * np.sum([p * np.log(p + 1e-6) for p in moments.values() if p > 0])
            entropy = -1 * np.sum([p / np.sum(list(moments.values())) * np.log(p / np.sum(list(moments.values()))) for p in moments.values() if p > 0])
            # Gri ton ortalama ve medianını hesapla
            mean_val = np.mean(image[y:y+h, x:x+w])
            median_val = np.median(image[y:y+h, x:x+w])
            # Ortalama, median ve moment değerlerini sakla
            data.append([i+1, (x+w/2, y+h/2), w, h, diagonal, energy, entropy, mean_val, median_val])
        # DataFrame oluştur
        df = pd.DataFrame(data, columns=["No", "Center", "Length", "Width", "Diagonal", "Energy", "Entropy", "Mean", "Median"])
        # Excel dosyasına yaz
        excel_path = "koyu_yesil_bolgeler.xlsx"
        df.to_excel(excel_path, index=False) 



    
    def ana_sayfa(self):
        self.baslik_label.config(text="DERS ADI: Dijital Görüntü İşleme\n\nÖĞRENCİ NO: 211229029\n\nAD SOYAD: Zişan AYDOĞAN")

    def odev1_detaylari(self, baslik):
        self.detaylar_penceresi = tk.Toplevel(self)
        self.detaylar_penceresi.title("Ödev Detayları")
        self.detaylar_penceresi.geometry("400x400")

        detaylar_label = tk.Label(self.detaylar_penceresi, text=baslik, font=("Arial", 12))
        detaylar_label.pack(pady=10)

        # Görüntü yükleme alanı
        goruntu_yukle_buton = tk.Button(self.detaylar_penceresi, text="Görüntü Yükle", command=self.goruntu_yukle)
        goruntu_yukle_buton.pack(pady=5)

        self.canvas = tk.Canvas(self.detaylar_penceresi, bg="white")
        self.canvas.pack(expand=True, fill="both")
        self.canvas.config(width=400, height=200)

        # Interpolasyon yöntemi seçimi için diyalog kutusu
        self.interpolasyon_label = tk.Label(self.detaylar_penceresi, text="Interpolasyon Yöntemi:", font=("Arial", 10, "bold"))
        self.interpolasyon_label.pack()
        self.interpolasyon_secim = tk.StringVar()
        self.interpolasyon_secim.set("seçiniz")  
        self.interpolasyon_menu = tk.OptionMenu(self.detaylar_penceresi, self.interpolasyon_secim, "Bilinear", "Bicubic", "Average")
        self.interpolasyon_menu.pack()
        self.bk_label5 = tk.Label(self.detaylar_penceresi, text="\n")
        self.bk_label5.pack()
        
        
        self.image_pil = Image.open(dosya_yolu)
        self.image = ImageTk.PhotoImage(self.image_pil)
        self.canvas.config(width=self.image_pil.width, height=self.image_pil.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)

        # Büyültme/Küçültme araçları
        self.bk_label = tk.Label(self.detaylar_penceresi, text="Büyültme/Küçültme", font=("Arial", 10, "bold"))
        self.bk_label2 = tk.Label(self.detaylar_penceresi, text="(Küçültme ise oranı 0.5 vs şeklinde ondalıklı giriniz)", font=("Arial", 8, "bold"))
        self.bk_label.pack()
        self.bk_label2.pack()

        self.boyut_label = tk.Label(self.detaylar_penceresi, text="Oran:", font=("Arial", 10))
        self.boyut_label.pack()
        self.boyut_entry = tk.Entry(self.detaylar_penceresi)
        self.boyut_entry.pack()

        self.bk_button = tk.Button(self.detaylar_penceresi, text="Uygula", command=self.bk_uygula)
        self.bk_button.pack()
        self.bk_label3 = tk.Label(self.detaylar_penceresi, text="\n")
        self.bk_label3.pack()

        # Zoom in/out araçları
        self.zoom_label = tk.Label(self.detaylar_penceresi, text="Zoom In/Out", font=("Arial", 10, "bold"))
        self.zoom_label.pack()

        self.zoom_label = tk.Label(self.detaylar_penceresi, text="Yüzde:", font=("Arial", 10))
        self.zoom_label.pack()
        self.zoom_entry = tk.Entry(self.detaylar_penceresi)
        self.zoom_entry.pack()

        self.zoom_in_button = tk.Button(self.detaylar_penceresi, text="Yakınlaştır", command=lambda: self.zoom("in"))
        self.zoom_in_button.pack(padx=5)
        self.zoom_out_button = tk.Button(self.detaylar_penceresi, text="Uzaklaştır", command=lambda: self.zoom("out"))
        self.zoom_out_button.pack(padx=5)
        self.bk_label4 = tk.Label(self.detaylar_penceresi, text="\n")
        self.bk_label4.pack()

        # Döndürme araçları
        self.dondur_label = tk.Label(self.detaylar_penceresi, text="Döndürme", font=("Arial", 10, "bold"))
        self.dondur_label.pack(pady=5)

        self.aci_label = tk.Label(self.detaylar_penceresi, text="Açı:", font=("Arial", 10))
        self.aci_label.pack()
        self.aci_entry = tk.Entry(self.detaylar_penceresi)
        self.aci_entry.pack()

        self.dondur_button = tk.Button(self.detaylar_penceresi, text="Uygula", command=self.dondur_uygula)
        self.dondur_button.pack()
        
    def odev2_detaylari(self, baslik):
        self.detaylar_penceresi = tk.Toplevel(self)
        self.detaylar_penceresi.title("S-Curve")
        self.detaylar_penceresi.geometry("400x400")
        detaylar_label = tk.Label(self.detaylar_penceresi, text=baslik, font=("Arial", 12))
        detaylar_label.pack(pady=10)

        # Görüntü yükleme alanı
        goruntu_yukle_buton = tk.Button(self.detaylar_penceresi, text="Görüntü Yükle", command=self.goruntu_yukle)
        goruntu_yukle_buton.pack(pady=5)

        self.canvas = tk.Canvas(self.detaylar_penceresi, bg="white")
        self.canvas.pack(expand=True, fill="both")
        self.canvas.config(width=400, height=200)

    def bilinear_interpolasyon(self, goruntu, x, y):
        x1 = int(x)
        x2 = x1 + 1
        y1 = int(y)
        y2 = y1 + 1
        
        if x2 >= goruntu.shape[1]:
            x2 = goruntu.shape[1] - 1
        if y2 >= goruntu.shape[0]:
            y2 = goruntu.shape[0] - 1
        
        P11 = goruntu[y1, x1]
        P12 = goruntu[y2, x1]
        P21 = goruntu[y1, x2]
        P22 = goruntu[y2, x2]
        
        alpha = x - x1
        beta = y - y1
        
        deger = (1 - alpha) * (1 - beta) * P11 + alpha * (1 - beta) * P21 + (1 - alpha) * beta * P12 + alpha * beta * P22
        
        return deger

    def bicubic_interpolasyon(self, goruntu, x, y):
        x1 = int(x)
        x2 = x1 + 1
        y1 = int(y)
        y2 = y1 + 1
        
        if x2 >= goruntu.shape[1]:
            x2 = goruntu.shape[1] - 1
        if y2 >= goruntu.shape[0]:
            y2 = goruntu.shape[0] - 1
        
        P = np.zeros((4, 4, 3), dtype=np.uint8)

        for j in range(4):
            for i in range(4):
                j_index = min(max(y1 - 1 + j, 0), goruntu.shape[0] - 1)
                i_index = min(max(x1 - 1 + i, 0), goruntu.shape[1] - 1)
                P[j, i] = goruntu[j_index, i_index]

        # Bicubic interpolasyon katsayıları
        a = -0.5 * P[0, 0] + 1.5 * P[1, 0] - 1.5 * P[2, 0] + 0.5 * P[3, 0]
        b = P[0, 0] - 2.5 * P[1, 0] + 2 * P[2, 0] - 0.5 * P[3, 0]
        c = -0.5 * P[0, 0] + 0.5 * P[2, 0]
        d = P[1, 0]
        
        nx = x - x1
        
        # Bicubic interpolasyon formülü
        deger = a * nx ** 3 + b * nx ** 2 + c * nx + d
        
        return deger

    def average_interpolasyon(self, goruntu, x, y):
        x1 = int(x)
        x2 = x1 + 1
        y1 = int(y)
        y2 = y1 + 1
        
        if x2 >= goruntu.shape[1]:
            x2 = goruntu.shape[1] - 1
        if y2 >= goruntu.shape[0]:
            y2 = goruntu.shape[0] - 1
        
        P = np.zeros((3,), dtype=np.uint8)
        
        for j in range(y1, y2 + 1):
            for i in range(x1, x2 + 1):
                P += goruntu[j, i]
        
        deger = P / ((y2 - y1 + 1) * (x2 - x1 + 1))
        
        return deger

    def goruntu_yukle(self):
        dosya_yolu = filedialog.askopenfilename(title="Görüntü Seç", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.jfif")])
        if dosya_yolu:
            self.goruntu_goster(dosya_yolu)

    def goruntu_goster(self, dosya_yolu):
        self.image_pil = Image.open(dosya_yolu)
        self.image = ImageTk.PhotoImage(self.image_pil)
        self.canvas.config(width=self.image_pil.width, height=self.image_pil.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)

       
    def bk_uygula(self):
        oran = float(self.boyut_entry.get())
        secilen_yontem = self.interpolasyon_secim.get()
        
        # Seçilen interpolasyon yöntemine göre görüntüyü yeniden boyutlandır
        if secilen_yontem == "Bilinear":
            image_resized = self.image_pil.resize((int(self.image_pil.width * oran), int(self.image_pil.height * oran)), Image.BILINEAR)
        elif secilen_yontem == "Bicubic":
            image_resized = self.image_pil.resize((int(self.image_pil.width * oran), int(self.image_pil.height * oran)), Image.BICUBIC)
        elif secilen_yontem == "Average":
            image_resized = self.image_pil.resize((int(self.image_pil.width * oran), int(self.image_pil.height * oran)), Image.ANTIALIAS)
        else:
            messagebox.showerror("Hata", "Geçersiz interpolasyon yöntemi!")
       
                    
        self.image = ImageTk.PhotoImage(image_resized)
        self.canvas.config(width=image_resized.width, height=image_resized.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)

    def zoom(self, mode):
        try:
            zoom_yuzdesi = float(self.zoom_entry.get())
            if mode == "in":
                width = int(self.image_pil.width * (1 + zoom_yuzdesi / 100))
                height = int(self.image_pil.height * (1 + zoom_yuzdesi / 100))
            elif mode == "out":
                width = int(self.image_pil.width * (1 - zoom_yuzdesi / 100))
                height = int(self.image_pil.height * (1 - zoom_yuzdesi / 100))
            else:
                return
            image_zoomed = self.image_pil.resize((width, height), Image.ANTIALIAS)
            self.image = ImageTk.PhotoImage(image_zoomed)
            self.canvas.config(width=width, height=height)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
        except ValueError:
            messagebox.showerror("Hata", "Geçersiz yüzde değeri!")

    def dondur_uygula(self):
        try:
            aci = float(self.aci_entry.get())
            image_rotated = self.image_pil.rotate(aci, expand=True)
            self.image = ImageTk.PhotoImage(image_rotated)
            self.canvas.config(width=image_rotated.width, height=image_rotated.height)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
        except ValueError:
            messagebox.showerror("Hata", "Geçersiz açı değeri!")

    def odev3_detaylari(self, baslik):
        self.detaylar_penceresi = tk.Toplevel(self)
        self.detaylar_penceresi.title("Ödev Detayları")
        self.detaylar_penceresi.geometry("400x400")

        detaylar_label = tk.Label(self.detaylar_penceresi, text=baslik, font=("Arial", 12))
        detaylar_label.pack(pady=10)

    def odev4_detaylari(self, baslik):
        self.detaylar_penceresi = tk.Toplevel(self)
        self.detaylar_penceresi.title("Ödev Detayları")
        self.detaylar_penceresi.geometry("400x400")

        detaylar_label = tk.Label(self.detaylar_penceresi, text=baslik, font=("Arial", 12))
        detaylar_label.pack(pady=10)

if __name__ == "__main__":
    app = AnaSayfa()
    app.mainloop()