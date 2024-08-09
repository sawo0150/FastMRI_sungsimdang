fastmri_sungsim V2 코드

smplayer 모델에 있는 모든 요소 넣어봄 (일단 debugging 완료)

넣은 요소들
1. gradient checkpointing / gradient accumulation
--> 다음의 vram 최적화를 통해 cascade 값 올릴 수 있었음 (but 현재 사용량 : 11GB)
2. MRAugment (Augment 방법들을 여러 방법을 통해 조절할 수 있음)
3. MRAugment scheduler (이것도 상수값을 조절할 수 있음)

2024/08/02 15시 수정
1. Model checkpointing, Loading 구현
2. Adam 대신 RAdam 적용
3. 기존의 SSIM 적용 대신 Mix Loss 적용 (MS_SSIM + L1_Loss)

2024/08/03 0시 수정
위의 요소들 모두 디버깅 완료

현재 모델 구성 : 
batch : 1
cascade : 20
chans : 9
sens_chans : 4

원래 구성 (Vram 5GB정도 사용하는 모델)
batch : 1
cascade : 1
chans : 9
sens_chan : 4

Vessl Server 1080 8GB GPU 기준 최대 cascade : 22 (chans : 9, sens_chan : 4)

--> 아마 cascade * chans * sens_chan의 숫자랑 Vram 사용량이랑 비례하지 않을 수도 있을 듯
==> 한번 계속 돌리면서 out of memory 에러가 뜨지 않는 최대값을 확인해보는게 좋을 것 같음

3개의 값을 잘 조합해서 최고의 성능을 이끌어내는 것이 중요할 것 같음

test 결과
SSIM
public : 0.9751
private : 0.9558

2024/08/05 14시 수정
mixed precision Learning 확인 및 50epoch 학습으로 V1 버전 final로 확정
+ mask Augment 비율 증가 함수 수정 (0.1(0epoch) -> 0.5(30epoch) -> 0.5(50epoch))
batch : 1
cascade : 20
chans : 9
sens_chan : 4

2024/08/05 23시 수정
CPU RAM 최적화를 통해서 .h5파일을 여러번 불러오지 말고 한번만 불러오도록 만들려고 했음
but 실험 결과 : 

원래 sungsimV1_final 
epoch : 5391.4s, valTime : 244.3260s
50iter : 47~48s
50epoch : 3.261

ram최적화 sungsimV1_final
epoch : 5350s
50iter : 33~55s

그래서 걍 안쓰기로 함

sungsimV2 : MRprompt 모델 기반

2024/08/10 02시 수정
sungsimV2.1 - cascade 6까지는 freezing 

cascade 10으로 만들고 끝 부분 4cascade와 sensitivity map만 새로 학습

--> 결국 cascade 개수 늘리는 코드 완성 밑 Debugging 완료!