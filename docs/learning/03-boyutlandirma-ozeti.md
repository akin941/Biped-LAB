# Kendi robotumuzun boyutlandırması: özet (WP4-01, G4.1)

**Ortam:** statik hesaplar bulutta, yürüme tabanlı tahminler GitHub Linux CI'da
(gha-linux) yapıldı. Hepsi simülasyon; donanım ölçümü yok. Ayrıntı:
`docs/experiments/2026-10-26-wp4-01-sizing.md`.

## Tasarım (biped_v1, taslak)
- Boy ~25 cm, kütle ~0.51 kg, 10 eklem (her bacakta kalça yaw/roll/pitch, diz, bilek).
- Sadece basit kutu geometrisi; Microduck parçası yok.
- Servo yeri tutucu: XL330 sınıfı (0.52 N·m, 18 g). Gerçek seçim WP4-03'te.

## Nasıl hesapladık
1. Robot dururken her eklemin taşıdığı tork (tek ayak ve çift ayak üzerinde).
2. Microduck'ın resmi yürüme politikasının simülasyondaki eklem torklarından bir
   "dinamik çarpan" çıkardık ve bizim robota ölçekledik.
3. Gereken değer = tahmini yürüme tepe değeri × 2 (güvenlik payı).

## Sonuç
| Eklem | Gereken tork (2× pay) | Yer tutucu servo yeter mi? |
|---|---|---|
| Kalça yaw | 0.21 N·m | evet |
| Kalça roll | 0.35 N·m | evet |
| Kalça pitch | 0.37 N·m | evet |
| **Diz** | **0.68 N·m** | **hayır** (0.52 N·m; ısınma riski de var) |
| Bilek | 0.44 N·m | evet (pay az) |

- Ortalama elektrik gücü ~7.7 W; 2S 500 mAh pil ile ~23 dk yürüme (varsayımlı).
- **Karar gerektiren nokta:** diz için en az ~0.7 N·m ve ~10 rad/s verebilen daha
  güçlü bir servo sınıfı gerekiyor. Bu, satın alma değil; WP4-03'te seçenekleri
  fiyatıyla birlikte sana sunacağım.
- Bu tahminler, kendi robotumuz için politika eğitildiğinde (G4.3, Mac) gerçek
  simülasyon ölçümleriyle yenilenecek.
