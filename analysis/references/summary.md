# 참고문헌 요약

## Siciliano et al. (2005) **Comparison of PVT and NaI(Tl) scintillators for vehicle portal monitor applications**  

*NIM A 550 (2005) 647–674*

Cite : `@Sicilliano2005Comparision

### 1. 연구 목적

본 논문은 방사선 포털 모니터(RPM)에서 널리 쓰이는 **PVT(Plastic Scintillator)** 와 **NaI(Tl)** 검출기를 비교하여, 국경 검문소에서 핵물질 탐지 성능을 평가하는 데 목적이 있다.

### 2. 두 검출기의 특성 비교

|  NaI(Tl) | PVT |
|:------|:--------|
| - 고밀도, 고원자번호 →  </br>  **높은 intrinsic efficiency** **뛰어난 에너지 분해능**, photopeak 명확 | - 저밀도 → intrinsic 효율은 낮지만 **대형 제작 가능 </br>→ absolute efficiency는 NaI보다 커질 수 있음**|
| - 동위원소 식별 가능 | - 에너지 정보는 제한적(Compton continuum) |
|- 고가, 온도·충격에 취약 | - 저가, 내환경성 우수

</br>

### 3. 실험 스펙트럼 비교

- NaI(Tl): 명확한 photopeak  
- PVT: 저에너지에 몰린 Compton 분포  
- 일부 에너지 창(Window)을 활용한 분석 가능

</br>

### 4. MCNP 시뮬레이션 결과

- NaI(Tl) intrinsic efficiency: **200 keV 근처에서 90%**
- PVT intrinsic efficiency: **100 keV 근처 40–50%**
- 절대 효율은 PVT가 더 높음(Fig. 5)
- 최적 두께:
  - NaI(Tl): 약 **1 cm**
  - PVT: **10–20 cm**

</br>

### 5. Cargo 감쇠 분석

- 100 keV 이하: **감쇠 10<sup>5</sup> 배**
- 1.5 MeV 이상: **감쇠 10 배**
- PVT: 원래 스펙트럼이 featureless → cargo 영향 상대적으로 덜 치명적
- Bare / Cargo 모두 PVT:NaI 비는 **5:1**

</br>

## 6. 결론 요약

| 항목 | PVT | NaI(Tl) |
|------|-----|----------|
| 비용 | ⭐ 매우 낮음 | 높음 |
| 크기 확장성 | ⭐ 대형 제작 용이 | 제한적 |
| 분해능 | 낮음 | ⭐ 매우 높음 |
| 동위원소 식별 | 제한적 | ⭐ 가능 |
| 내환경성 | ⭐ 강함 | 취약 |
| 1차 검사 | ⭐ 적합 | 비경제적 |
| 2차 검사 | 일부 가능 | ⭐ 최적 |

**요약:**

1차 검문에는 **PVT**, 정밀 식별이 필요한 2차 검문에는 **NaI(Tl)** 이 적합하다.

</br>

### 7. 활용도

- 논문 Introduction의 리뷰 자료  
- RPM detector 비교 근거  
- Background compensation, spectral analysis 논문 인용 근거로 활용 가능  

</br>

## Summary of Bertrand (2014): *Current Status on Plastic Scintillators Modifications*

CITE : `@Bertrand2014Current`


플라스틱 섬광체(Plastic Scintillators, PS)의 화학적·구조적 개선 현황을 광범위하게 정리한 리뷰 논문이다. PS는 빠른 응답속도, 낮은 비용, 대형 제작 용이성 등의 장점이 있으나, 고전적으로 낮은 광출력과 낮은 에너지 분해능이 한계로 지적되어 왔다. 이 논문은 이러한 한계를 극복하기 위한 최신 개선 방향을 정리한다.

---

### 1. 플라스틱 섬광체의 기본 구조

플라스틱 섬광체는 다음과 같은 성분으로 이루어진다.

- **고분자 매트릭스(polymer matrix)**: PVT, PS(polystyrene), PMMA 등  
- **1차 형광체(primary fluor)**: 에너지를 흡수해 UV 광 방출  
- **2차 형광체(secondary fluor, wavelength shifter)**: UV → 가시광 변환  
- **첨가제(additives)**: 광출력 향상, 기능성 강화 등  

</br>

### 2. 주요 성능 개선 전략

#### 2.1 광출력(light yield) 향상

- 형광체 농도 최적화  
- 높은 양자효율을 가진 새로운 WLS(wavelength shifter) 도입  
- 공중합(co-polymerization)을 통한 투명도 개선  
- 자기흡수(self‑absorption) 감소 설계  

---

### 3. 펄스 형태 판별(PSD) 기능을 갖춘 플라스틱 섬광체

과거에는 액체 섬광체만 가능했던 **중성자/감마 구분(PSD)** 기능이  
고분자 화학의 발전으로 PS에서도 가능해졌다.

- 느린 성분(slow component)과 빠른 성분(fast component)이 공존하도록 도핑  
- 중성자 대비 감마의 파형 차이를 분석  
- 핵보안/핵물질 검출 분야에서 매우 중요한 기능  

</br>

### 4. 나노복합체(nanocomposite) 플라스틱 섬광체

나노입자 혼합을 통해 다음과 같은 특성이 향상됨:

- **고 Z-원소 나노입자**(Bi, Pb 계열) → 감마선 상호작용 확률 증가  
- **퀀텀닷(QD)** → 발광 파장 조정 및 광학 특성 향상  
- 기계적 강도 향상  
- 새로운 스펙트럼 엔지니어링 가능  

</br>

### 5. 기능성 도핑 및 파장공학

#### 5.1 Gd-도핑  

- 중성자 포획 단면적이 큰 Gd를 첨가해 중성자 검출 감도 향상  

#### 5.2 Cl-함유 고분자  

- 체렌코프 광과 섬광광 분리를 가능하게 함  
- 고에너지 물리 실험에 유리  

#### 5.3 UV-투명 고분자  

- 이중 판독(dual‑readout) 방식 가능  
- TOF-PET 등 의료 영상 분야에 적용  

</br>

### 6. 응용 분야

개량된 PS는 다음 분야에서 성능을 크게 향상시킴:

- 국경 핵물질 탐지(RPM, 보안 분야)  
- PET·TOF-PET 의료 영상장치  
- 입자 물리 실험의 대면적 검출기  
- 중성자/감마 듀얼 모드 검출 시스템  

</br>

### 7. 결론

플라스틱 섬광체는 전통적 한계(낮은 에너지 분해능)를 가지지만,  
최근의 화학적 수정·나노도핑·파장 엔지니어링을 통해:

- 광출력 증가  
- PSD 기능 확보  
- 고 Z-content 구현  
- 맞춤형 광학 특성 설계  

가 가능해졌으며, 저비용·대형 제작이 필요한 분야에서 경쟁력이 더욱 강화되었다.

</br>

## Ely et al. (2008) 한국어 요약  **The Use of Energy Information in Plastic Scintillator Material**

Cite : `@Ely2006EWindow`

### 1. 연구 목적

플라스틱 섬광체(PVT)는 RPM(방사선 포털 모니터)에서 가장 널리 사용되는 감마 검출기이다. 그러나 낮은 Z값과 컴프턴 산란이 지배적이기 때문에 **photopeak이 존재하지 않으며** **동위원소 식별이 불가능**하다. 이 논문은 **제한된 에너지 정보를 어떻게 활용하면 NORM(자연방사성물질)로 인한 nuisance alarm을 줄일 수 있는가**를 분석한다.

</br>

### 2. 플라스틱 섬광체의 감마 상호작용 특성  

- **컴프턴 산란이 지배적** → 연속적인 컴프턴 분포만 관측됨  
- **Photoelectric absorption**은 매우 저에너지(<50 keV)에서만 의미 있음  
- **Full-energy peak이 형성되지 않는 이유**  
  1. 낮은 Z → 대부분 컴프턴  
  2. 긴 mean free path → multiple scattering 부족  
  3. 광수 통계가 낮아 에너지 분해능이 매우 낮음  

이로 인해 PVT 기반 스펙트럼은 **에너지 구분이 매우 제한적**이다.

</br>

### 3. 에너지 윈도잉(Energy Windowing) 기법  

PVT에서 가능한 최소한의 스펙트럼 정보를 활용하여 **저에너지/고에너지 윈도 비율(Ratio)을 사용**하는 방법.

#### 3.1 단순 Gross Count의 한계  

배경 대비 총계수 증가만 비교하면  

- 비료(대표적 NORM)  
- 플루토늄  

이 **동일한 크기의 증가**를 보여 구분할 수 없음.

#### 3.2 Ratio 기반 판별  

저에너지 윈도 / 고에너지 윈도 비율을 사용하면:

- 비료(NORM): 배경과 비슷한 비율  

- 플루토늄(SNM): 스펙트럼 형태가 달라 **비율이 유의하게 증가**  

따라서 Ratio가 **NORM nuisance alarm을 크게 줄이는 핵심 기법**임.

</br>>

### 4. 실차량(fertilizer truck) 실험 결과  

(논문 Fig.3 기반)

- 차량 통과 시 Low/High window 모두 증가  
- 그러나 **Low/High 비율은 배경과 거의 동일**  
→ 비료 적재 차량은 **NORM으로 정확히 분류 가능**

</br>

### 5. 대규모 차량 데이터 기반 알고리즘 평가  

수천 대의 실제 RPM 차량 데이터에 **57Co 가상 신호를 주입(injection)**하여 Gross counting vs Ratio vs 고급 알고리즘을 비교.

결과:

- Gross count 알고리즘  
  - 높은 sensitivity  
  - 그러나 **NORM alarm(오경보) 비율 매우 높음**
- Ratio 알고리즘  
  - sensitivity 유지  
  - **NORM alarm 대폭 감소**
- 고급 알고리즘(PCA, Fisher discriminant, Neural network 등)  
  - Ratio보다 더 나은 가능성 존재  

</br>

### 6. 한계점  

- 의료 방사성핵종(예: Tc-99m)은 SNM과 PVT 스펙트럼 형태가 유사 → 구분 어려움  
- PMT, 전자회로는 환경 변화에 민감  
- pedestrian / 차량 검사 중 medical isotope 비중이 높으면 Ratio 방식 효과 떨어짐  

</br>

### 7. 결론

- PVT의 제한적 에너지 분해능에도 불구하고, **에너지 윈도 비율(Ratio) 분석만으로 대부분의 NORM을 효과적으로 제거할 수 있음**  
- SNM 탐지 sensitivity는 유지하면서 nuisance alarm 비율을 크게 줄임  
- 추가적인 PCA, Fisher-LDA, Neural Network 기반 다변량 알고리즘이 유망  
- 그러나 동위원소 식별은 NaI(Tl) 같은 스펙트로스코픽 검출기에 비해 제한적임  
- 넓은 설치 규모가 필요한 RPM에서는 **비용·신뢰성을 고려할 때 여전히 PVT가 실용적 선택**
