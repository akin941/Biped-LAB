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
- Şimdilik robot sadece ayakta duruyor; yürüme politikası Mac'teki eğitimden
  (G4.3) sonra eklenecek.

Simülasyonda denendi: itilince düşen robotta motorlar aynı döngüde kapandı;
veri kesilince 0.2–0.5 s içinde motorlar kapandı.

**Senden istenen:** bu, güvenlik katmanı kodu olduğu için birleştirmeden önce
senin onayın gerekiyor (PR'daki açıklamaya bakman yeterli). Mac'te çalıştırma
adımları `MAC-08` olarak eklendi.
