import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
])


testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
testloader_full = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


indices = np.arange(len(testset))
np.random.seed(42)
np.random.shuffle(indices)
test_10_indices = indices[:1000]
test_20_indices = indices[:2000]
testloader_10 = torch.utils.data.DataLoader(Subset(testset, test_10_indices), batch_size=100, shuffle=False, num_workers=2)
testloader_20 = torch.utils.data.DataLoader(Subset(testset, test_20_indices), batch_size=100, shuffle=False, num_workers=2)


teacher_model = torchvision.models.resnet34(pretrained=False)
teacher_model.fc = nn.Linear(teacher_model.fc.in_features, 100)
teacher_model.load_state_dict(torch.load('best_resnet34_cifar100.pth', map_location=device))
teacher_model = teacher_model.to(device)
teacher_model.eval()


teacher_params = sum(p.numel() for p in teacher_model.parameters())
print(f"Teacher parameters: {teacher_params}")


class StudentCNN(nn.Module):
    def __init__(self, num_channels, num_classes=100):
        super(StudentCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(num_channels, num_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels * 2, num_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(num_channels * 2, num_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_channels * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 7, 7, 0, bias=False),  
        )

    def forward(self, z):
        return self.model(z.view(-1, 100, 1, 1))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


student_10 = StudentCNN(num_channels=48).to(device)  # ~10% parameters
student_20 = StudentCNN(num_channels=72).to(device)  # ~20% parameters
generator = Generator().to(device)
discriminator = Discriminator().to(device)


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, non_trainable

s10_trainable, s10_non_trainable = count_parameters(student_10)
s20_trainable, s20_non_trainable = count_parameters(student_20)
print(f"Student 10% params: {s10_trainable + s10_non_trainable} ({100 * (s10_trainable + s10_non_trainable) / teacher_params:.2f}% of teacher)")
print(f"Student 20% params: {s20_trainable + s20_non_trainable} ({100 * (s20_trainable + s20_non_trainable) / teacher_params:.2f}% of teacher)")


def kl_divergence(student_logits, teacher_logits, temperature=3.0):
    return nn.KLDivLoss(reduction='batchmean')(
        nn.functional.log_softmax(student_logits / temperature, dim=1),
        nn.functional.softmax(teacher_logits / temperature, dim=1)
    ) * (temperature ** 2)

criterion_gan = nn.BCELoss()
criterion_class = nn.CrossEntropyLoss()


optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_s10 = optim.Adam(student_10.parameters(), lr=0.001)
optimizer_s20 = optim.Adam(student_20.parameters(), lr=0.001)


def train_akd(student, optimizer_s, num_epochs=100, batch_size=64, latent_dim=100):
    student.train()
    generator.train()
    discriminator.train()
    real_label = 0.9  
    fake_label = 0.1

    for epoch in range(num_epochs):
        train_loader_tqdm = tqdm(range(100), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for _ in train_loader_tqdm:
         
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_images = generator(z)

        
            with torch.no_grad():
                teacher_logits = teacher_model(fake_images)
                pseudo_labels = torch.argmax(teacher_logits, dim=1)

         
            optimizer_d.zero_grad()
            real_output = discriminator(fake_images.detach() + 0.01 * torch.randn_like(fake_images))
            fake_output = discriminator(fake_images.detach())
            d_loss = criterion_gan(real_output, torch.full_like(real_output, real_label)) + \
                     criterion_gan(fake_output, torch.full_like(fake_output, fake_label))
            d_loss.backward()
            optimizer_d.step()

           
            optimizer_g.zero_grad()
            fake_output = discriminator(fake_images)
            g_gan_loss = criterion_gan(fake_output, torch.full_like(fake_output, real_label))
            g_class_loss = criterion_class(teacher_model(fake_images), pseudo_labels)
            g_loss = g_gan_loss + 0.5 * g_class_loss
            g_loss.backward()
            optimizer_g.step()

           
            optimizer_s.zero_grad()
            student_logits = student(fake_images.detach())
            teacher_logits = teacher_model(fake_images.detach())
            kl_loss = kl_divergence(student_logits, teacher_logits)
            kl_loss.backward()
            optimizer_s.step()

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] KL Loss: {kl_loss.item():.4f}, G Loss: {g_loss.item():.4f}, D Loss: {d_loss.item():.4f}")
          
            if epoch % 50 == 0:
                gen_images = fake_images[:16].cpu().detach()
                gen_images = (gen_images + 1) / 2
                grid = torchvision.utils.make_grid(gen_images, nrow=4)
                plt.figure(figsize=(8, 8))
                plt.imshow(grid.permute(1, 2, 0))
                plt.title(f"Generated Images Epoch {epoch}")
                plt.axis('off')
                plt.savefig(f'gen_images_epoch_{epoch}.png')
                plt.close()


def evaluate_model(model, testloader, title):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.title(f'{title} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{title.replace(" ", "_")}_cm.png')
    plt.close()
    
    return accuracy, cm


print("Training Student 10%...")
train_akd(student_10, optimizer_s10, num_epochs=100, batch_size=64)
print("Training Student 20%...")
train_akd(student_20, optimizer_s20, num_epochs=100, batch_size=64)


acc_10_10, cm_10_10 = evaluate_model(student_10, testloader_10, "Student 10% (10% Test Data)")
acc_10_20, cm_10_20 = evaluate_model(student_10, testloader_20, "Student 10% (20% Test Data)")
acc_20_10, cm_20_10 = evaluate_model(student_20, testloader_10, "Student 20% (10% Test Data)")
acc_20_20, cm_20_20 = evaluate_model(student_20, testloader_20, "Student 20% (20% Test Data)")


print("\nEvaluation Results:")
print(f"Student 10% (10% Test Data) Accuracy: {acc_10_10:.2f}%")
print(f"Student 10% (20% Test Data) Accuracy: {acc_10_20:.2f}%")
print(f"Student 20% (10% Test Data) Accuracy: {acc_20_10:.2f}%")
print(f"Student 20% (20% Test Data) Accuracy: {acc_20_20:.2f}%")
print(f"Student 10% Parameters: Trainable={s10_trainable}, Non-trainable={s10_non_trainable}")
print(f"Student 20% Parameters: Trainable={s20_trainable}, Non-trainable={s20_non_trainable}")


generator.eval()
z = torch.randn(16, 100).to(device)
gen_images = generator(z).cpu().detach()
gen_images = (gen_images + 1) / 2
grid = torchvision.utils.make_grid(gen_images, nrow=4)
plt.figure(figsize=(8, 8))
plt.imshow(grid.permute(1, 2, 0))
plt.title("Final Generated Images")
plt.axis('off')
plt.savefig('final_gen_images.png')
plt.show()