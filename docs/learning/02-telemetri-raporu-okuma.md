# Telemetri raporu nasıl okunur?

`bipedlab record ... --html` komutu ya da CI'daki `policy-reports` dosyası bir HTML
raporu üretir. Tarayıcıda açın; internet gerekmez.

## Üstteki tablolar

- **Ortam etiketi (`env_label`)**: Sonucun nerede üretildiği. `gha-linux` = GitHub'ın
  Linux makinesi, `mac-m5` = sizin Mac'iniz. Bir ortamdaki sonuç başka ortam için
  kanıt sayılmaz.
- **Düştü mü?**: Gövde 60°'den fazla eğildiyse "Evet".
- **İleri hız / hız hatası**: Komut 0,2 m/s ise ve ölçülen 0,001 m/s ise robot yerinde
  sayıyor demektir.
- **Ayak yerde oranı**: Ayakta duran robotta (1, 1) olur. Yürüyen robotta her ayak bir
  süre havada kalır; yaklaşık 0,5–0,7 beklenir.
- **Ortalama güç / en yüksek tork**: Motorların ne kadar zorlandığı. Kendi robotumuz
  için motor seçerken bu sayılar temel alınacak (Faz 4).
- **Gerçek zamana göre hız**: Simülasyonun o bilgisayarda kaç kat hızlı koştuğu.

## Grafikler

| Grafik | Ne görmeliyiz? | Sorun işareti |
|---|---|---|
| Gövde eğimi | Birkaç derecenin altında, sakin bir çizgi | 60°'ye tırmanan çizgi = düşme |
| İleri hız: komut ve gerçekleşen | İki çizgi birbirine yakın | Ölçülen çizgi sıfırda kalıyorsa robot yürümüyor |
| Dönüş hızı | Aynı mantık, sola dönüş pozitif | |
| Mekanik güç | Yürürken düzenli dalgalanma | Ani sıçramalar = sarsıntı / çarpma |
| Ayak teması | Sol ve sağ sırayla 0–1 arasında değişir | İkisi hep 1 ise adım atılmıyor |
| Sol bacak eklemleri | Yürürken tekrarlayan (periyodik) dalgalar | Düz çizgiler = hareket yok |
| En yüksek eklem torku | Motor sınırının altında | Sınıra yapışık çizgi = motor yetersiz |
| Kontrol adımı süresi | 20 ms'nin çok altında | 20 ms'yi aşmak gerçek robotta gecikme demek |

## Örnek: bugünkü bulgu

Resmi yürüme politikası ile kayıtta eğim düşük, ayaklar sürekli yerde ve ileri hız
neredeyse sıfır. Yani robot dengede ama adım atmıyor. Bu, motor modelinin (BAM)
eksikliğine işaret ediyor; ayrıntı için
[deney kaydı](../experiments/2026-10-26-official-policies-mjcf-actuators.md).
