import numpy as np
import random

class MaskAugmentor:
    def __init__(self, current_epoch_fn, initial_acc_values=[4, 5, 8], later_acc_range=(3, 12), step=0.01, total_epochs=100):
        """
        Args:
            current_epoch_fn: 현재 epoch를 반환하는 함수.
            initial_acc_values: 초기 가속 비율 값들의 리스트.
            later_acc_range: 후반부에 사용할 가속 비율의 범위.
            step: 가속 비율 증가량.
            total_epochs: 총 epoch 수.
        """
        self.current_epoch_fn = current_epoch_fn  # 현재 epoch를 반환하는 함수
        self.initial_acc_values = initial_acc_values
        self.later_acc_range = later_acc_range
        self.step = step
        self.total_epochs = total_epochs

    def generate_mask(self, acc, mask_length):
        mask = np.zeros(mask_length, dtype=np.uint8)
        center = mask_length // 2
        mask[center - 16:center + 16] = 1

        acc = int(acc)  # acc 값을 정수로 변환
        # print(acc)
        left_start = random.randint(0, int(acc-1))
        right_start = random.randint(0, int(acc-1))
        
        left_indices = list(range(left_start, center - 16, acc))
        right_indices = list(range(center + 16 + right_start, mask_length, acc))

        indices = left_indices + right_indices
        mask[indices] = 1
        # print(mask)
        return mask

    def update_acc_probability(self, epoch):
        progress = min(1, 0.1+(epoch / self.total_epochs) * 0.4)  # 마지막 epoch에서 0.8이 되도록 설정
        initial_weight = max(0, 1 - progress)
        later_weight = min(1, progress)

        return initial_weight, later_weight

    def get_acc(self):
        epoch = self.current_epoch_fn()  # 현재 epoch를 가져옴
        initial_weight, later_weight = self.update_acc_probability(epoch)
        
        if random.random() < initial_weight:
            return random.choice(self.initial_acc_values)
        else:
            return random.uniform(self.later_acc_range[0], self.later_acc_range[1])

    def augment(self, mask):
        epoch = self.current_epoch_fn()  # 현재 epoch를 가져옴
        # print("현재 epoch : ", epoch)
        # 80% 확률 이하로만 새로운 마스크를 생성함
        if random.random() <= self.update_acc_probability(epoch)[1]:
            acc = self.get_acc()
            augmented_mask = self.generate_mask(acc, len(mask))
        else:
            augmented_mask = mask

        return augmented_mask
