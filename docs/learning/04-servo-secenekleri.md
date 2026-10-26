# Servo seçenekleri: özet (WP4-03)

**Ortam:** bulut. Üretici sayfalarından alınan veriler ve hesap; donanım ölçümü
yok, hiçbir şey satın alınmadı. Ayrıntı:
`docs/experiments/2026-10-26-wp4-03-actuator-survey.md`.

## Ana bulgu
- Diz dışındaki tüm eklemler XL330-M288-T (18 g, ~€40) ile 2× payla karşılanıyor.
- Dizde yalnızca **XC330-M288-T, 6 V'ta** hem tork hem hız payını sağlıyor
  (tork ×3.2, hız ×2.0). 5 V'ta hız payı ×1.7'ye düşüyor.

## Öneri (karar senin, BOM aşamasında)
1. **Karma set, 6 V tek hat:** dizlerde 2 × XC330-M288-T, diğer 8 eklemde
   XL330-M288-T. Kütle +10 g. Fiyat görüntüsü: 8 × €40.20 + 2 × $103.39.
   Dezavantaj: 6 V servoların üst sınırı; 2S pilden regülatör gerekir.
2. **Aynı set 5 V'ta:** dizde hız payı ×1.7 kabul edilir; elektrik daha basit.
3. **Bacakları kısaltmak:** diz yükü azalır, belki tüm eklemler XL330 olur;
   hesabı yeniden yaparım.

Bu tahminler Microduck'ın yürüyüşünden türetildi. Kendi robotumuz için politika
eğitilince (Mac, G4.3) dizi yeniden kontrol edeceğim; satın alma ondan sonra.
