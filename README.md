fastmri_sungsim V1 코드

smplayer 모델에 있는 모든 요소 넣어봄 (일단 debugging 완료)

넣은 요소들
gradient checkpointing

gradient accumulation

--> 다음의 vram 최적화를 통해 cascade 값 올릴 수 있었음 (but 현재 사용량 : 11GB)

MRAugment (Augment 방법들을 여러 방법을 통해 조절할 수 있음)

MRAugment scheduler (이것도 상수값을 조절할 수 있음)

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


--> 아마 cascade * chans * sens_chan의 숫자랑 Vram 사용량이랑 비례하지 않을 수도 있을 듯
==> 한번 계속 돌리면서 out of memory 에러가 뜨지 않는 최대값을 확인해보는게 좋을 것 같음


3개의 값을 잘 조합해서 최고의 성능을 이끌어내는 것이 중요할 것 같음


test 결과
SSIM

public : 0.9751

private : 0.9558

total : 0.9655
