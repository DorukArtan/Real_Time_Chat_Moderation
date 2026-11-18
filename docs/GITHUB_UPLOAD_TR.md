# GitHub'a gönderme

Hedef: **DorukArtan/Real_Time_Chat_Moderation**. Mevcut proje klasörü `C:\Users\USER\Desktop\LLM_HW1` olarak kalabilir; yeniden klonlamak gerekmez.

Bu bilgisayarda yerel Git bağlantısı ve commit hazırlandıktan sonra, IDE'nin PowerShell terminalinde:

```powershell
cd C:\Users\USER\Desktop\LLM_HW1
git remote -v
git push -u origin main
```

`origin` hedefinin `DorukArtan/Real_Time_Chat_Moderation.git` olduğunu kontrol et. Giriş istenirse **DorukArtan** hesabını kullan. `Everything up-to-date` veya başarılı `main -> main` çıktısı gönderimin sonucunu gösterir. Gönderim tamamlanınca GitHub sayfasını açıp kaynak klasörlerinin göründüğünü doğrula.

Yeni depodaki başlangıç README commit'inin geçmişi korunarak proje eklenir. `.gitignore`, ortamı, büyük ağırlıkları, verileri ve yerel hesap/izin ekran görüntülerini Git dışında tutar.

Tarayıcıdan elle yüklemeyi tercih edersen hazırlanan `dist/Real_Time_Chat_Moderation_github.zip` paketini boş bir klasöre açıp içindeki dosya ve klasörleri yükleyebilirsin. Kök dizinde `README.md`, `app.py`, `run.cmd`, `.gitignore`, `.gitattributes` ve `moderation/` görünmeli. Gizli dosyaları ve `.vscode/` klasörünü de ekle.

Model dosyalarının erişilebilir indirme bağlantısı ayrıca README'ye eklenecek. Bu yapılana kadar yeni bir bilgisayardaki klon, eğitilmiş modeli doğrudan çalıştıramaz. Sunum ve video da ayrı teslim işleridir.

Colab not defteri güncel kaynakla yeniden paketlendi. Drive'daki tamamlanmış eski `run_001` kaynak arşivini değiştirme. Bu yeni not defteriyle eğitimi baştan yapacaksan yeni bir `RUN_NAME` seç; mevcut eğitilmiş paket için yalnızca restore adımını kullan. Bu hazırlık sırasında yeniden eğitim yapılmadı.
