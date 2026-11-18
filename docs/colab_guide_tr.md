# Colab ile başlangıç

1. `notebooks/01_colab_training.ipynb` dosyasını Google Colab'da **Dosya > Not defteri yükle** yoluyla aç.
2. **Çalışma zamanı > Çalışma zamanı türünü değiştir > T4 GPU** seç. Ücretsiz GPU müsaitliği garanti değildir.
3. Hücreleri sırayla çalıştır. Drive bağlama hücresindeki Google izin ekranını kendi hesabınla tamamla.
   Kurulum hücresi sanal ortamı `--without-pip` ile oluşturur ve paketleri ana pip'in `--python` seçeneğiyle o ortama kurar; böylece Colab'da `ensurepip` bulunmasına ihtiyaç duymaz. Yarım kalan ortamı silmeden onarabilir.
4. Varsayılan deney adı `run_001`. Tamamlanan aşamalar aynı deneyde tekrar eğitilmez. Kod veya ayar değişirse yeni deney adı kullan.
5. Önce veri hazırlanır, sonra klasik modeller, Word2Vec/sinir ağı ve Transformer eğitilir. Bunların hepsi Colab makinesinde çalışır.
6. INT8 dönüşümü ve ortak test ölçümleri de Colab'da yapılır. Ölçümler GPU yerine CPU kullanır.
7. Arayüz hücresi çalışan uygulamayı not defterinin içinde açar. En az iki yeni mesaj dene; birini zor örnek olarak seç.
8. Son hücreler sonuçları ve ağırlık paketini Drive'a kaydeder. Bilgisayarında eğitim başlatılmaz.

Çıktılar `MyDrive/SWE015_Moderation/runs/run_001/` altında saklanır. Özellikle `reports/results.csv`, `reports/evaluation.json`, `reports/failure_examples.json` ve `moderation_run_bundle.zip` dosyalarını koru.

Bir oturum koparsa tamamlanan aşamalar Drive'da kalır. Yarım kalan eğitim aşaması yeniden başlar; tam optimizer durumu ile kaldığı batch'ten devam etme uygulanmadı. Aynı oturumda uygulama hücresini tekrar çalıştırmak sunucuyu yeniler.

Henüz ölçülmüş doğruluk/hız sonucu yoktur. Bunları eğitim ve değerlendirme hücreleri üretir. Sonraki teslim işleri gerçek sonuçlarla PDF sunumu, sesli iki dakikalık video ve erişilebilir depo/bağlantılardır.

