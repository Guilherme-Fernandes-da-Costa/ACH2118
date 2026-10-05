import pandas as pd
import numpy as np
import torch
import os
import gc
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Preparação
df_train = pd.read_excel('train.xlsx')
X = df_train['resp_text'].fillna('').astype(str).tolist()
y = LabelEncoder().fit_transform(df_train['clarity'])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Treinamento do Complement Naive Bayes
vetorizador = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10000,
    sublinear_tf=True,       # Aplica escala logarítmica para atenuar palavras repetidas
    # Remove palavras que aparecem apenas 1 vez (ruído/erros)
    min_df=2,
    max_df=0.85,             # Remove palavras que estão presentes em mais de 85% dos textos
)
X_train_tfidf = vetorizador.fit_transform(X_train)
X_val_tfidf = vetorizador.transform(X_val)

cnb = ComplementNB(alpha=0.1)
cnb.fit(X_train_tfidf, y_train)
prob_cnb = cnb.predict_proba(X_val_tfidf)

# 3. Snapshot Ensembling do BERTimbau
checkpoints = [
    './resultados_bert_otimizado/checkpoint-1006',
    './resultados_bert_otimizado/checkpoint-1509'
]

tokenizer = AutoTokenizer.from_pretrained(
    'neuralmind/bert-base-portuguese-cased')
val_encodings = tokenizer(X_val, truncation=True,
                          padding=True, max_length=512, return_tensors='pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

probs_todos_checkpoints = []
lote_tamanho = 8

print("Iniciando a extração dos Checkpoints do BERT...\n")

for checkpoint in checkpoints:
    if not os.path.exists(checkpoint):
        print(f"Aviso: {checkpoint} não encontrado. Ignorando.")
        continue

    print(f"Processando {checkpoint}...")
    modelo_bert = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, use_safetensors=True)
    modelo_bert.to(device)
    modelo_bert.eval()

    prob_atual = []

    with torch.no_grad():
        for i in range(0, len(X_val), lote_tamanho):
            input_ids = val_encodings['input_ids'][i:i+lote_tamanho].to(device)
            attention_mask = val_encodings['attention_mask'][i:i +
                                                             lote_tamanho].to(device)

            outputs = modelo_bert(input_ids, attention_mask=attention_mask)
            probabilidades = torch.nn.functional.softmax(
                outputs.logits, dim=-1)
            prob_atual.extend(probabilidades.cpu().numpy())

    prob_atual_array = np.array(prob_atual)
    probs_todos_checkpoints.append(prob_atual_array)

    # Calcula e exibe a acurácia isolada deste checkpoint específico
    acuracia_checkpoint = accuracy_score(
        y_val, np.argmax(prob_atual_array, axis=1))
    print(f"Acurácia isolada deste checkpoint: {acuracia_checkpoint:.4f}\n")

    # Limpeza de memória da Placa de Vídeo antes de carregar o próximo
    del modelo_bert
    torch.cuda.empty_cache()
    gc.collect()

# Calcula a média das épocas do BERT
prob_bert_media = np.mean(probs_todos_checkpoints, axis=0)

# 4. Procura Automática do Melhor Peso para a Fusão
melhor_acuracia = 0
melhor_peso_bert = 0

print("\nProcurando a melhor distribuição matemática de pesos...")
# Testa pesos para o BERT de 50% até 100%, em intervalos de 5%
for peso_bert in np.arange(0.50, 1.01, 0.05):
    peso_cnb = 1.0 - peso_bert

    # Calcula o ensemble com a proporção atual
    prob_ensemble_teste = (prob_cnb * peso_cnb) + (prob_bert_media * peso_bert)
    predicoes_teste = np.argmax(prob_ensemble_teste, axis=1)

    acuracia_teste = accuracy_score(y_val, predicoes_teste)

    if acuracia_teste > melhor_acuracia:
        melhor_acuracia = acuracia_teste
        melhor_peso_bert = peso_bert

peso_final_cnb = 1.0 - melhor_peso_bert

print("\n--- RESULTADOS DA FUSÃO DINÂMICA ---")
print(
    f"Acurácia Isolada do CNB:             {accuracy_score(y_val, np.argmax(prob_cnb, axis=1)):.4f}")
print(
    f"Acurácia Média dos BERTs (Snapshot): {accuracy_score(y_val, np.argmax(prob_bert_media, axis=1)):.4f}")
print(f"Acurácia Final do Ensemble:          {melhor_acuracia:.4f}")
print(
    f"Pesos Utilizados:                    BERT {melhor_peso_bert*100:.0f}% | CNB {peso_final_cnb*100:.0f}%")

print("\nIniciando as previsões no conjunto de teste...")


print("\nIniciando a avaliação sobre o conjunto de treino completo (20.092 textos)...")

# 1. Extração Estatística (CNB) no conjunto inteiro
# Utiliza-se a variável X original definida no início do script
X_full_tfidf = vetorizador.transform(X)
prob_cnb_full = cnb.predict_proba(X_full_tfidf)

# 2. Extração Semântica (Snapshot BERT) no conjunto inteiro
val_encodings_full = tokenizer(
    X, truncation=True, padding=True, max_length=512, return_tensors='pt')
probs_todos_checkpoints_full = []

for checkpoint in checkpoints:
    print(f"Processando {checkpoint} para o conjunto de treino...")
    modelo_bert = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, use_safetensors=True)
    modelo_bert.to(device)
    modelo_bert.eval()

    prob_atual = []
    with torch.no_grad():
        for i in range(0, len(X), lote_tamanho):
            input_ids = val_encodings_full['input_ids'][i:i +
                                                        lote_tamanho].to(device)
            attention_mask = val_encodings_full['attention_mask'][i:i+lote_tamanho].to(
                device)

            outputs = modelo_bert(input_ids, attention_mask=attention_mask)
            probabilidades = torch.nn.functional.softmax(
                outputs.logits, dim=-1)
            prob_atual.extend(probabilidades.cpu().numpy())

    probs_todos_checkpoints_full.append(np.array(prob_atual))

    # Limpeza rigorosa da VRAM
    del modelo_bert
    torch.cuda.empty_cache()
    gc.collect()

prob_bert_media_full = np.mean(probs_todos_checkpoints_full, axis=0)

# 3. Fusão Final e Cálculo da Acurácia
# Utiliza a proporção exata descoberta na validação dinâmica
prob_ensemble_full = (prob_cnb_full * peso_final_cnb) + \
    (prob_bert_media_full * melhor_peso_bert)
predicoes_numericas_full = np.argmax(prob_ensemble_full, axis=1)

# Compara as predições do ensemble com os rótulos originais 'y'
acuracia_treino = accuracy_score(y, predicoes_numericas_full)

print(f"\n--- ACURÁCIA NO CONJUNTO DE TREINO COMPLETO ---")
print(f"Acurácia Final: {acuracia_treino:.4f}\n")


# Carrega os dados de teste
df_test = pd.read_excel('test1.xlsx')
X_test = df_test['resp_text'].fillna('').astype(str).tolist()

# 1. Extração Estatística (CNB)
X_test_tfidf = vetorizador.transform(X_test)
prob_cnb_test = cnb.predict_proba(X_test_tfidf)

# 2. Extração Semântica (Snapshot BERT)
val_encodings_test = tokenizer(
    X_test, truncation=True, padding=True, max_length=512, return_tensors='pt')
probs_todos_checkpoints_test = []

for checkpoint in checkpoints:
    print(f"Processando {checkpoint} para o teste...")
    modelo_bert = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, use_safetensors=True)
    modelo_bert.to(device)
    modelo_bert.eval()

    prob_atual = []
    with torch.no_grad():
        for i in range(0, len(X_test), lote_tamanho):
            input_ids = val_encodings_test['input_ids'][i:i +
                                                        lote_tamanho].to(device)
            attention_mask = val_encodings_test['attention_mask'][i:i+lote_tamanho].to(
                device)

            outputs = modelo_bert(input_ids, attention_mask=attention_mask)
            probabilidades = torch.nn.functional.softmax(
                outputs.logits, dim=-1)
            prob_atual.extend(probabilidades.cpu().numpy())

    probs_todos_checkpoints_test.append(np.array(prob_atual))
    del modelo_bert
    torch.cuda.empty_cache()
    gc.collect()

prob_bert_media_test = np.mean(probs_todos_checkpoints_test, axis=0)

# 3. Fusão Final (CNB + Média dos BERTs) com os pesos encontrados
prob_ensemble_test = (prob_cnb_test * peso_final_cnb) + \
    (prob_bert_media_test * melhor_peso_bert)
predicoes_numericas_test = np.argmax(prob_ensemble_test, axis=1)

# 4. Reversão dos Rótulos Numéricos para Texto (c1, c234, c5)
# Instancia um LabelEncoder idêntico ao do início do treino para mapear as classes corretamente
le = LabelEncoder()
le.fit(df_train['clarity'])
classes_finais = le.inverse_transform(predicoes_numericas_test)

# 5. Exportação
df_test['clarity'] = classes_finais
df_test.to_excel('submissao_final.xlsx', index=False)
print("\nPrevisões concluídas e guardadas no ficheiro 'submissao_final.xlsx'.")
