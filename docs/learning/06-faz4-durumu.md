# Faz 4 (kendi robotumuz simülasyonda): durum özeti

**Tarih:** 2026-11-23. **Ortam:** her satırda yazıyor. Hepsi simülasyon; donanım
için kanıt değildir. Hiçbir şey satın alınmadı.

## Kapılar
| Kapı | Ne istiyor | Durum |
|---|---|---|
| G4.1 | Boyutlandırma hesapları (tork, hız, güç; 2× pay) | Hesaplar hazır (bulut + Mac). **Senin incelemen bekliyor** (`03-boyutlandirma-ozeti.md`). |
| G4.2 | Kendi MJCF modelimiz, 10 eklem, 20–30 cm | Tamam (bulut, GitHub Linux CI). |
| G4.3 | Rastgeleleştirmeyle yürüyen politika, itme/sürtünme/gecikme raporu | Tamam (Mac, MAC-14). Sonra STS3215 servo eğrisiyle (MAC-37) ve 10 mm engebeli zeminde (MAC-42) de geçti. |
| G4.4 | Rust programı simülasyonu 50 Hz'de, güvenlik denetçisiyle sürüyor | Tamam (GitHub Linux CI + Mac). |
| G4.5 | Malzeme listesi ve servo seçimi, fiyatıyla | **Senin kararın.** Servo kararı "şimdilik bekle". |

## Son durum: yürüme politikası
- Seçilen politika **k3_6900** (engebeli zeminde eğitilmiş, üç komşu kontrol
  noktasının ortalaması).
- Rust programı üzerinden, STS3215 servo modeliyle, 8 farklı başlangıçtan 0.15 ve
  0.25 m/s'de yürüdü: düşme yok, hata yok. Sola/sağa kayma en fazla 0.056 rad/s
  (Mac, MAC-46).
- Bundan önce program, politikaya eklem hızlarını eğitimden farklı (bir adım
  erken) veriyordu; robot hep sola dönüyordu. Bu düzeltildi (WP4-37/38).
- Gerçek zamanlı koşu da geçti (Mac, Mac yüklüyken): düşme yok, hata yok, 1000
  döngünün hiçbiri gecikmedi; dönme en fazla 0.041 rad/s.

## Servo konusunda bilinenler
- `04-servo-secenekleri.md` eski hesaba dayanıyor. Kendi yürüyüşümüzden
  ölçünce (MAC-36) asıl sınırın **hız** olduğu çıktı: kalça ve dizde ~7 rad/s.
- Simülasyonda gerçek eğrisiyle yürüyen aday **Feetech STS3215, 7.4 V (2S pil)**:
  tüm eklemlerde 2× tork payı var; kalça ve dizde hız payı 2×'in altında
  (gereken ~4.8, servonun yüksüz hızı 5.45 rad/s). Yürüyüş yine de tutuyor.
- Programda STS3215 haberleşme kodu hazır, ama gerçek cihazla denenmedi.

## Senden beklenenler (acele yok)
1. G4.1 hesaplarına göz atıp "tamam" demen.
2. G4.5: servo ve pil voltajı kararı. Satın alma ancak senin açık onayınla olur.
