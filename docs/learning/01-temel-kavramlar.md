# Temel kavramlar

Bu sayfa projede geçen teknolojileri kod okumadan anlamanız için yazıldı.

## Robot nasıl yürür? Büyük resim

Microduck'ı düşünün: 14 küçük motor (servo), bir denge sensörü (IMU) ve küçük bir
bilgisayar. Saniyede **50 kez** şu döngü çalışır:

1. **Algıla:** IMU "öne 3° eğildim, şu hızla dönüyorum" der; motorlar "eklemim şu açıda" der.
2. **Düşün:** Bu 61 sayı bir yapay sinir ağına (politika) verilir.
3. **Hareket et:** Ağ 14 sayı üretir: her motorun gideceği yeni açı. Motorlar oraya gider.

Ağ bu işi deneme yanılmayla, binlerce kez düşerek **simülasyonda** öğrenmiştir.

## MuJoCo: fizik simülatörü

Gerçek dünyanın yerçekimi, sürtünme ve çarpışma kurallarını bilgisayarda hesaplayan
bir "oyun motoru" gibi düşünün; ama oyun değil, mühendislik için hassas yapılmış.
Robotu bir XML dosyasıyla tarif ederiz (MJCF): parçalar, eklemler, kütleler, motorlar.
Simülasyonda robotu düşürmek bedavadır; gerçekte bir motor kırılabilir.

**Projeden bir gözlem:** Microduck'ı simülasyonda "ayakta dur" pozuna koyup motorları
sabit tuttuğumuzda **yaklaşık 0,9 saniyede öne düşüyor**. Yani robot pasif olarak
duramaz; dengeyi sürekli düzelten bir kontrol (politika) şart. İnsan da böyledir:
ayakta dururken farkında olmadan sürekli küçük düzeltmeler yaparız.

## Pekiştirmeli öğrenme (RL) ve PPO

Bir köpeğe ödül maması ile numara öğretmek gibi: robot bir şey dener, iyi giderse
**ödül**, düşerse ceza alır. "İstenen hızda ilerle: +2 puan", "dik dur: +2", "motorları
sarsma: −0,1" gibi. Binlerce paralel simülasyonda milyonlarca deneme sonrası ağ,
ödülü en yüksek yapan hareketleri öğrenir.

**PPO** (Proximal Policy Optimization) bu öğrenmenin en yaygın yöntemlerinden biri.
Özelliği temkinli olması: her güncellemede politikayı "çok fazla" değiştirmez; böylece
öğrenme kararlı ilerler.

**Domain randomization (alan rastgeleleştirme):** Simülasyon gerçeğin birebir aynısı
değildir. Bu yüzden eğitimde sürtünme, kütle, motor gücü, sensör gürültüsü her seferinde
biraz değiştirilir. Böylece politika "her türlü dünyada" çalışmayı öğrenir ve gerçek
robota geçince şaşırmaz (sim-to-real).

## ONNX: eğitilmiş beynin taşınabilir dosyası

Eğitilen ağ bir `.onnx` dosyasına kaydedilir. Bu, Python'da da, Rust'ta da, tarayıcıda
da, robotun içindeki küçük bilgisayarda da aynı şekilde çalışan standart bir formattır.
Microduck'ın yürüme beyni yaklaşık 1 MB'lık böyle bir dosyadır.

Biz bu dosyaları Hugging Face'ten **sabit bir sürümden** indirip **sha256 parmak izini**
kontrol ediyoruz: dosya bir bit bile değişmişse kullanmıyoruz.

## Rust: robotun içindeki yazılım

Robotun üzerinde çalışan program hızlı, öngörülebilir ve hatasız olmalı; bir anlık
takılma robotun düşmesi demek. Rust, bellek hatalarını derleme anında yakalayan modern
bir dildir. Microduck'ın robot yazılımı Rust'tadır. Biz Rust'ı **Faz 4'te**, kendi
robotumuzun kontrol yazılımı için kullanacağız; şimdilik gerek yok (ADR-0007).

## Neden önce simülasyon?

- Bedava ve güvenli: düşmek bir şey kırmaz.
- Hızlı: bulut makinesinde Microduck simülasyonu gerçek zamandan ~90 kat hızlı çalıştı
  (Mac'te ayrıca ölçülecek).
- Ölçülebilir: her eklemin torkunu ölçüp kendi robotumuz için hangi motorun yeteceğini
  hesaplayabiliriz.

## Bulut ve Mac neden ayrı?

Bulut oturumunda ekran ve Apple çipi yok. Kodu, testleri ve belgeleri bulutta yapıyoruz;
3B pencere, Apple Silicon hız ölçümü ve uzun eğitimler sizin Mac'inizde yapılacak.
Bir sonuç hangi ortamda ölçüldüyse yalnızca o ortam için geçerli sayılır.
