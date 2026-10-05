import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, preds)}


# 1. Carregamento e Preparação
df_train = pd.read_excel('train.xlsx')
df_train['resp_text'] = df_train['resp_text'].fillna('').astype(str)

X = df_train['resp_text'].tolist()
y = LabelEncoder().fit_transform(df_train['clarity'])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Carregamento do Tokenizer e Modelo Base
nome_modelo = 'neuralmind/bert-base-portuguese-cased'
tokenizer = AutoTokenizer.from_pretrained(nome_modelo)
modelo = AutoModelForSequenceClassification.from_pretrained(
    nome_modelo,
    num_labels=3,
    use_safetensors=True
)

# 3. Tokenização dos Dados
train_encodings = tokenizer(
    X_train, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(X_val, truncation=True, padding=True, max_length=512)

# Classe utilitária para converter os dados para o formato do PyTorch


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = TextDataset(train_encodings, y_train)
val_dataset = TextDataset(val_encodings, y_val)

# 4. Configuração do Treinamento
training_args = TrainingArguments(
    output_dir='./resultados_bert_otimizado',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,   # Simula um batch efetivo de 32
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,              # Taxa ligeiramente maior suportada pelo warmup
    weight_decay=0.01,               # Combate direto ao overfitting
    warmup_steps=314,                # Protege os pesos iniciais
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

trainer = Trainer(
    model=modelo,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# 5. Iniciar o Fine-Tuning
print("A iniciar o fine-tuning do BERTimbau...")
trainer.train()

# 6. Avaliação Final
resultados = trainer.evaluate()
print("Resultados finais na validação:", resultados)
