# Robotun kontrol programı (Rust çalışma zamanı): özet (WP4-04)

**Ortam:** bulut + GitHub Linux CI. Yalnızca simülasyon; donanım güvenliği için
kanıt değildir.

Robotun üzerinde çalışacak kontrol döngüsünün ilk hali yazıldı (`runtime/`).
Saniyede 50 kez şu zinciri çalıştırıyor:

komut kapısı → kontrolcü → güvenlik denetçisi → servolar

- **Komut kapısı:** hız komutlarını sınırlar (en çok 0.3 m/s ileri), ani
  değişimi yumuşatır, 0.5 s yeni komut gelmezse robotu durdurur.
- **Güvenlik denetçisi:** robot 45°'den fazla eğilirse, eklem çok hızlı dönerse
  ya da veri gelmezse motorları gevşetir ve bu durumu kilitler. Hatalı (NaN)
  hedefleri yok sayar, eklem sınırlarını ve hız sınırını uygular.
- Eğitilmiş yürüme politikası artık bu programda çalışabiliyor (WP4-10): bulutta
  kısa eğitilmiş bir politika 30 s boyunca hatasız koştu. Gerçek yürüme
  politikası Mac'teki eğitimden (MAC-08, G4.3) gelecek.

Simülasyonda denendi: itilince düşen robotta motorlar aynı döngüde kapandı;
veri kesilince 0.2–0.5 s içinde motorlar kapandı.

**Kayıt (WP4-13):** program her döngüde ne ölçtüğünü, ne istendiğini ve
denetçinin ne gönderdiğini bir dosyaya yazabiliyor (`--log`). Bu dosyadan aynı
HTML raporu üretiliyor; raporda motorların açık olduğu anlar ve denetçinin
devreye girdiği döngüler görünüyor. Kayıt yazılamazsa robot durmaz, sadece kayıt
kapanır.

**Durum:** güvenlik katmanı kodunu 2026-11-06'da onayladın ve birleştirildi.
Denetçinin sınırlarında yapılacak her değişiklik yine senin onayını gerektirir.
Mac'te çalıştırma adımları `MAC-09` olarak eklendi.
