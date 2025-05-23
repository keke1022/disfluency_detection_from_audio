import os
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import Wav2Vec2FeatureExtractor
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
import pandas as pd
from tqdm import tqdm
import librosa
from torch.nn.utils.rnn import pad_sequence

from models import AcousticModel

torch.manual_seed(42)
np.random.seed(42)

DISFLUENCY_LABELS = ['FP', 'RP', 'RV', 'RS', 'PW']

class DisfluencyDataset(Dataset):
    def __init__(self, audio_dir, label_dir):
        """
        audio_dir: directory containing audio files, assumed to be in .wav format
        label_dir: directory containing label files, assumed to be in .npy format
        """
        self.audio_dir = audio_dir
        self.label_dir = label_dir
        self.audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_file = os.path.join(self.audio_dir, self.audio_files[idx])
        label_file = os.path.join(self.label_dir, self.audio_files[idx].replace('.wav', '.npy'))

        audio, sr = librosa.load(audio_file, sr=16000)
        audio = torch.tensor(audio, dtype=torch.float32)
        
        # load labels, 3:8 is the range of labels in the .npy file
        labels = np.load(label_file)
        labels = labels[:, 3:8]
        labels = torch.tensor(labels, dtype=torch.float32)
        return {
            "audio": audio,
            "labels": labels
        }
    
def collate_fn(batch):
    batch_audios = [b["audio"] for b in batch]
    batch_labels = [b["labels"] for b in batch]

    # use pad_sequence to pad the audio and labels
    batch_padded_audios = pad_sequence(batch_audios, batch_first=True, padding_value=0.0)
    batch_padded_labels = pad_sequence(batch_labels, batch_first=True, padding_value=0.0)

    return {
        "audio": batch_padded_audios,
        "labels": batch_padded_labels
    }

def train_model(model, train_dataloader, val_dataloader, device, num_epochs=15, lr=5e-5, patience=50):
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = BCEWithLogitsLoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        for step, batch in enumerate(tqdm(train_dataloader)):
            audio = batch["audio"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            _, logits = model(audio)

            # cut the labels to match the logits
            if labels.shape[1] != logits.shape[1]:
                labels = labels[:, :logits.shape[1]]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
            
            if step % 100 == 0 and step > 0:
                model.eval()
                val_loss = 0.0
                val_steps = 0
                
                with torch.no_grad():
                    for val_batch in val_dataloader:
                        val_audio = val_batch["audio"].to(device)
                        val_labels = val_batch["labels"].to(device)
                        
                        _, val_logits = model(val_audio)
                        if val_labels.shape[1] != val_logits.shape[1]:
                            val_labels = val_labels[:, :val_logits.shape[1]]
                        val_loss += criterion(val_logits, val_labels).item()
                        val_steps += 1
                
                avg_val_loss = val_loss / val_steps
                print(f"Epoch: {epoch}, Step: {step}, Val Loss: {avg_val_loss:.4f}")
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), "saved_model/best_wavlm_disfluency.pt")
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping after {epoch} epochs and {step} steps")
                    return
                model.train()
        
        avg_train_loss = train_loss / train_steps
        print(f"Epoch: {epoch}, Train Loss: {avg_train_loss:.4f}")
    
    return model

def evaluate_model(model, test_dataloader, device):
    model.to(device)
    model.eval()
    criterion = BCEWithLogitsLoss()
    test_loss = 0.0
    test_steps = 0
    
    with torch.no_grad():
        for batch in test_dataloader:
            audio = batch["audio"].to(device)
            labels = batch["labels"].to(device)
            _, logits = model(audio)
            if labels.shape[1] != logits.shape[1]:
                labels = labels[:, :logits.shape[1]]
            loss = criterion(logits, labels)
            test_loss += loss.item()
            test_steps += 1
    avg_test_loss = test_loss / test_steps
    print(f"Test Loss: {avg_test_loss:.4f}")

    return avg_test_loss

def main():
    batch_size = 2
    learning_rate = 5e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")

    dataset = DisfluencyDataset(
        audio_dir="/home/kekeliu/repos/disfluency_detection_from_audio/data/wav_sil",
        label_dir="/home/kekeliu/repos/disfluency_detection_from_audio/data/labels_framelevel",
        feature_extractor=feature_extractor
    )
    
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn
    )
    
    model = AcousticModel()
    
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        num_epochs=15,
        lr=learning_rate,
        patience=50
    )
    
    model.load_state_dict(torch.load("saved_model/best_wavlm_disfluency.pt", map_location=device))
    evaluate_model(model, test_dataloader, device)

if __name__ == "__main__":
    main()