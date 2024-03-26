﻿# SISE의 계산량 감소를 위한 Attribution mask Compress-SISE 기법 제안

## Abstract
본 논문은 SISE 기법에서 유사한 속성 마스크 들을 압축하여 설명맵 생성시 연산량을 줄이는 AC-SISE 기법을 제안한다. AC-SISE의 성능을 측정하기 위해 VGG16 모델로 실험한 결과 EBPG, mIoU에서 SISE 보다 각 1.40%, 0.07% 낮은 성능을 보이지만 속성 마스크는 17.47% 감소했다. ResNet50 모델은 EBPG에서 SISE 보다 1.90% 낮은 성능을 mIoU에서는 0.89% 높은 성능을 보이고 속성 마스크는 34.69% 더 감소한 성능을 보였다.

### Figure 1. AC-SISE 기법으로 1ayer에서 visualization map을 생성하는 과정
<img width="451" alt="image" src="https://github.com/kyh980909/AC-SISE/assets/14137708/94a6c1f5-b59a-4e75-b4f6-739551931175">

### Figure 2. 속성 마스크 압축 방법
<img width="237" alt="image" src="https://github.com/kyh980909/AC-SISE/assets/14137708/87befcef-4c3f-4833-85bb-9d05fdddd6c8">

### Table 1. 실험 결과
<img width="449" alt="스크린샷 2024-03-26 13 16 59" src="https://github.com/kyh980909/AC-SISE/assets/14137708/2aa1b27d-deb2-40e1-9fcc-ae6428149b33">

## Conclusion
본 논문에서는 SISE의 속성 마스크를 압축하는 기법인 AC-SISE를 제안했다. 실험 결과 SISE의 속성 마스크를 압축했을 때 모델 설명 성능이 크게 낮아지지 않으면서 연산량은 크게 감소된 것을 확인할 수 있다.
향후 연구에서는 속성 마스크 압축 기법을 다양한 방식으로 연구하여 성능을 개선할 계획이다.
