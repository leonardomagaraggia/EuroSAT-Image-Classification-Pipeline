import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from model import SimpleCNN, get_device
from prepare_dataset import test_loader


# Impostazioni

device = get_device()
num_classes = 10


# Carica modello salvato

model = SimpleCNN(num_classes=num_classes).to(device)
model.load_state_dict(torch.load("eurosat_cnn.pth", map_location=device))
model.eval()

# Predizioni sul test set

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


# Confusion matrix

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - EuroSAT Test Set")
plt.show()

# -----------------------
# Report classificazione
# -----------------------
class_names = test_loader.dataset.dataset.classes  # nomi classi EuroSAT
report = classification_report(all_labels, all_preds, target_names=class_names)
print(report)
