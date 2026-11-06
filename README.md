# Biped Robotics Lab

İki ayaklı robotlar için simülasyon öncelikli bir laboratuvar. İlk hedef, açık kaynak
**Microduck** robotunu (Pollen Robotics) simülasyonda anlamak ve çalıştırmak. Asıl
hedef, buradan öğrenilenlerle **tamamen özgün** bir iki ayaklı robotu önce simülasyonda,
sonra gerçek donanımda geliştirmek.

> Agents / English readers: start with [AGENTS.md](AGENTS.md).

## Şu an neredeyiz?

**Faz 4 (kendi robotumuz `biped_v1`, simülasyonda) bulutta ilerliyor; Faz 3 ve 4'ün tam
eğitimleri Mac'te çalıştırılmayı bekliyor.** Güncelleme: 2026-11-06.
Ayrıntılı durum: [yol haritası](docs/roadmap/ROADMAP.md) ve [görev listesi](docs/roadmap/BACKLOG.md).

| Ne | Durum | Nerede doğrulandı |
|---|---|---|
| Microduck'ın resmi modeli ve resmi politikaları (yürüme, ayakta durma, oturma, top vurma) bizim çalıştırıcımızda | ✅ çalışıyor; resmi yürüme politikası G1.3 hız eşiğini geçemiyor (politikanın sınırı, bizim hatamız değil) | GitHub Linux |
| Telemetri kaydı ve tek dosyalık HTML rapor | ✅ | GitHub Linux |
| Pekiştirmeli öğrenme ortamı + her gece kısa PPO eğitimi | ✅ ödül 3 tohumda artıyor | GitHub Linux |
| Microduck için tam PPO eğitimi (G3.3) | ⏳ Mac'te çalıştırılmayı bekliyor (MAC-06) | — |
| Kendi robotumuz `biped_v1`: model, boyutlandırma, servo araştırması | ✅ | bulut / GitHub Linux |
| `biped_v1` eğitim ortamı, titreme düzeltmesi, sağlamlık testleri (itme, sürtünme, gecikme) | ✅ kısa eğitimde robot sadece ayakta duruyor; yürüme tam eğitimi bekliyor | bulut / GitHub Linux |
| `biped_v1` tam eğitimi ve diz torku ölçümü (G4.3) | ⏳ Mac'te çalıştırılmayı bekliyor (MAC-08) | — |
| Rust çalışma zamanı ve güvenlik denetleyicisi (G4.4) | ⏳ PR #25, sizin güvenlik onayınızı bekliyor | — |
| Malzeme listesi (BOM) taslağı | ⏳ taslak hazır; voltaj, acil durdurma ve satın alma zamanı sizin kararınız | — |
| Mac'te kurulum, görsel simülatör, hız ölçümü | ⏳ bekliyor | [Mac görevleri](docs/handoffs/MAC_LOCAL_HANDOFF.md) |
| Fiziksel donanım | ❌ başlamadı (onayınız olmadan satın alma yok) | — |

## Hızlı başlangıç (Mac'te)

```bash
brew install uv
git clone https://github.com/akin941/Biped-LAB.git && cd Biped-LAB
uv sync --all-extras
uv run python scripts/fetch_upstream.py microduck_rl     # resmi modeli sabit sürümden indir
uv run pytest                                            # testler
uv run bipedlab inspect microduck                        # robotun eklemleri, motorları
uv run bipedlab stand toy                                # bizim basit robot ayakta duruyor mu?
uv run mjpython -m bipedlab.cli view microduck           # 3B pencere (yalnızca Mac/ekranlı makine)
```

## Belgeler

- [Temel kavramlar (Türkçe)](docs/learning/01-temel-kavramlar.md): MuJoCo, PPO, ONNX, Rust neden var?
- [Faz 0 raporu (Türkçe)](docs/reports/PHASE0_RAPOR.md)
- [Araştırma özeti](docs/research/SUMMARY.md) ve ayrıntılı raporlar
- [Mimari](docs/architecture/ARCHITECTURE.md), [teknoloji kararları](docs/architecture/TECH_DECISION_MATRIX.md), [ADR'ler](docs/decisions/README.md)
- [Riskler](docs/roadmap/RISKS.md), [güvenlik ve emniyet](docs/security/SECURITY_AND_SAFETY.md), [test stratejisi](docs/testing/TESTING_STRATEGY.md)

## Lisans

Bu depo Apache-2.0'dır. Microduck'ın 3B mesh dosyaları (CC BY-NC-SA 4.0, ticari olmayan)
bu depoya **konmaz**; testler sırasında sabitlenmiş sürümden indirilir. Bkz. [NOTICE](NOTICE),
[ADR-0004](docs/decisions/ADR-0004-licensing.md).
