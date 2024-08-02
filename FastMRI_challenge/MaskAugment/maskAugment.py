import numpy as np
import random

class MaskAugmentor:
    def __init__(self, initial_acc_values=[4, 5, 8], later_acc_range=(3, 12), step=0.01, total_epochs=100):
        self.initial_acc_values = initial_acc_values
        self.later_acc_range = later_acc_range
        self.step = step
        self.current_epoch = 0
        self.total_epochs = total_epochs

    def generate_mask(self, acc, mask_length):
        mask = np.zeros(mask_length, dtype=np.uint8)
        center = mask_length // 2
        mask[center - 16:center + 16] = 1

        left_start = random.randint(0, acc-1)
        right_start = random.randint(0, acc-1)
        
        left_indices = list(range(left_start, center - 16, acc))
        right_indices = list(range(center + 16 + right_start, mask_length, acc))

        indices = left_indices + right_indices
        mask[indices] = 1

        return mask

    def update_acc_probability(self, epoch):
        progress = min(1, (epoch / self.total_epochs) * 0.8)  # 마지막 epoch에서 0.8이 되도록 설정
        initial_weight = max(0, 1 - progress)
        later_weight = min(1, progress)

        return initial_weight, later_weight

    def get_acc(self):
        # initial_weight, later_weight = self.update_acc_probability(self.current_epoch)
        
        # if random.random() < initial_weight:
        #     return random.choice(self.initial_acc_values)
        # else:
        return random.uniform(self.later_acc_range[0], self.later_acc_range[1])

    def augment(self, mask, current_epoch):
        self.current_epoch = current_epoch
        # 80% 확률 이하로만 새로운 마스크를 생성함
        if random.random() <= self.update_acc_probability(self.current_epoch)[1]:
            acc = self.get_acc()
            augmented_mask = self.generate_mask(acc, len(mask))
        else:
            augmented_mask = mask

        return augmented_mask

# Example usage
augmentor = MaskAugmentor(total_epochs=100)

# In your training loop
for epoch in range(total_epochs):
    for iter, data in enumerate(data_loader):
        mask, kspace, target, maximum, _, _ = data
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True).requires_grad_(True)
        target = target.cuda(non_blocking=True).requires_grad_(False)
        maximum = maximum.cuda(non_blocking=True).requires_grad_(False)
        
        # Apply mask augmentation
        mask = augmentor.augment(mask)
        
        # Apply gradient checkpointing
        output = checkpointed_forward(model, kspace, mask)
        loss = loss_type(output, target, maximum)
        
        loss = loss / args.gradient_accumulation_steps  # Scale the loss for gradient accumulation
        loss.backward()

        if (iter + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * args.gradient_accumulation_steps

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item() * args.gradient_accumulation_steps:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()

    total_loss = total_loss / len_loader
    print(f"Epoch {epoch + 1}/{total_epochs}, Total Loss: {total_loss}")
