# Faz 0 raporu (26 Ekim 2026)

## 1. Doğrulanan teknik bulgular

Kaynaklar sabitlenmiş commit'lerde satır satır okundu (ayrıntı ve referanslar:
[araştırma özeti](../research/SUMMARY.md)). Yukarı akış (upstream) kodu çalıştırılmadı;
yalnızca model dosyaları bizim kodumuzla yüklendi.

- **Microduck:** ~25 cm, simülasyon modelinde 0,737 kg. 15 Dynamixel XL330 servo var;
  yapay zekâ bunlardan **14'ünü** sürüyor (bacak başına 5, boyun/kafa 4). Gövde IMU'su
  servo hattında (LSM6DSV16X). **LiDAR yok**; ToF mesafe sensörü, kamera, NFC var.
- **Kontrol döngüsü:** saniyede 50 kez; politika 61 sayı alıp 14 hedef açı üretiyor.
  Bu "dil" eğitim kodunda, robot yazılımında ve tarayıcı simülatöründe aynı.
- **Eğitim:** resmi eğitim (microduck_rl) **NVIDIA CUDA ekran kartı istiyor**; Mac'te
  olduğu gibi çalışmaz. Topluluk projesi microduck-lab, Mac'te CPU ile PPO eğitimi
  yapıyor (kodu doğrulandı, performans iddiaları doğrulanmadı).
- **Lisans:** kod Apache-2.0; Microduck'ın 3B parça dosyaları **CC BY-NC-SA 4.0**
  (ticari kullanım yok). Bu yüzden kendi robotumuz tamamen özgün geometriyle tasarlanacak.
  Microduck'ın CAD dosyaları, malzeme listesi (BOM) veya devre şemaları yayımlanmamış.
- **Robot yazılımı:** Rust ile yazılmış, ONNX Runtime kullanıyor. Güvenlik katmanı
  ince: eklem başına limit yok, acil durdurma yok. Kendi robotumuzda bunu katmanlı
  güvenlik mimarisiyle çözüyoruz ([ADR-0006](../decisions/ADR-0006-layered-safety.md)).

## 2. Önerilen mimari

Simülasyon ve öğrenme Python + MuJoCo 3.10 (CPU) ile; politika dosyaları ONNX; robot
üzerindeki gerçek zamanlı yazılım Faz 4'te Rust; motorları her zaman ayrı bir
mikrodenetleyicideki güvenlik denetleyicisi yönetir; yapay zekâ/LLM motorlara asla
doğrudan komut vermez. Ayrıntı: [mimari](../architecture/ARCHITECTURE.md),
[teknoloji kararları](../architecture/TECH_DECISION_MATRIX.md).

## 3. Yol haritası (özet)

0 Araştırma → 1 Microduck simülasyonu → 2 Gözlemlenebilirlik (grafikler, raporlar) →
3 Pekiştirmeli öğrenme (Mac'te) → 4 Özgün robot (simülasyonda) → 5 Fiziksel prototip
(**onayınızla**) → 6 Otonom davranışlar. Her fazın ölçülebilir çıkış kapısı var:
[yol haritası](../roadmap/ROADMAP.md).

## 4. Temel riskler

Resmi eğitimin CUDA gerektirmesi; simülasyondaki motor modelinin eğitimdekinden farklı
olması (aşağıdaki bulgu); bulut ortamının Hugging Face'e erişememesi; ileride donanım
ve batarya güvenliği. Tamamı: [risk kaydı](../roadmap/RISKS.md).

## 5. Tamamlanan işler ve testler

| Test | Sonuç | Ortam |
|---|---|---|
| Kod kalitesi (ruff, mypy strict) | geçti | bulut + GitHub Linux |
| Birim + simülasyon testleri | 25 geçti, 2 atlandı (politika indirilemedi) | bulut |
| Aynı testler | geçti | GitHub Linux ve **GitHub macOS arm64** (sizin M5'iniz değil) |
| Resmi politikalar sha256 doğrulamasıyla indirildi, 61→14 sözleşmesi | 9 test geçti | GitHub Linux |
| Sır taraması, bağımlılık güvenlik açığı taraması | temiz | GitHub Linux |

**Önemli bulgu:** Resmi yürüme ve ayakta durma politikaları robotu 10 saniye boyunca
dengede tutuyor (politikasız robot 0,9 saniyede düşüyor). Ancak "0,2 m/s ileri yürü"
komutunda robot neredeyse hiç ilerlemiyor (0,001 m/s). Muhtemel neden: politikalar
eğitimde **BAM** adlı gerçekçi bir motor modeliyle eğitildi; biz şimdilik MuJoCo'nun
basit motor modelini kullanıyoruz. [Kanıt](../experiments/2026-10-26-official-policies-mjcf-actuators.md)

## 6. Mac gerektiren işler

[Mac görevleri](../handoffs/MAC_LOCAL_HANDOFF.md): MAC-01 kurulum ve testler, MAC-02
3B görüntüleyici, MAC-03 M5 hız ölçümü, MAC-04 resmi tarayıcı simülatörüyle karşılaştırma.
Hiçbiri henüz çalıştırılmadı.

## 7. Sıradaki otonom iş paketi

BAM motor modelinin entegrasyonu (onayınıza bağlı), başlangıç durumunun resmi koda
eşitlenmesi, oturma/kalkma ve diğer davranışların raporlanması, ardından telemetri
kaydedici (Faz 2).

## 8. Sizin kararınız gereken konular

1. **BAM motor modeli:** Rhoban'ın açık kaynak (Apache-2.0) BAM kütüphanesini sabit
   bir commit'ten bağımlılık olarak eklemek (önerilen; içe aktarılan kısmı incelendi:
   ağ, dosya yazma, komut çalıştırma yok) ya da modeli kendimiz yeniden yazmak.
2. (İsteğe bağlı) `main` dalına "CI geçmeden birleştirme yok" koruması eklemek
   (GitHub → Settings → Branches).
3. (İsteğe bağlı) Bulut ortamının ağ ayarlarına `huggingface.co` eklemek; böylece
   politika testleri bulutta da çalışır.
