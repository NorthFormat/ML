from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import argrelextrema
import torch
import math
import numpy as np


class MLpart:
    def __init__(self):
        MODEL_NAME = 'UrukHan/t5-russian-spell'
        self.MAX_INPUT = 2024
        self.tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        self.modelFor = T5ForConditionalGeneration.from_pretrained("cointegrated/rut5-base-multitask")
        self.tokenizerFor = T5Tokenizer.from_pretrained("cointegrated/rut5-base-multitask")
        self.modelFor.load_state_dict(torch.load('F:\PythonTests\MLXakaton\pytorch_model.bin'))
        self.tokenizerPhar = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
        self.modelPhar = AutoModel.from_pretrained("cointegrated/rubert-tiny2")

    def do(self, text, g, p, f):
        # Токенизирование данных
        answer = text
        if g:
            answer = ""
            task_prefix = "Spell correct: "
            for i in text.strip().split("."):
                i = i.strip()
                encoded = self.tokenizer(
                    task_prefix + i,
                    padding="longest",
                    max_length=self.MAX_INPUT,
                    truncation=True,
                    return_tensors="pt",
                )
                predicts = self.model.generate(**encoded)
                answer += self.tokenizer.batch_decode(predicts, skip_special_tokens=True)[0].strip(".") + ". "
        if p:
            answer  = self.paragraph(answer)
        if f:
            answer = self.generate(answer)
        return answer

    def generate(self, text, **kwargs):
        inputs = self.tokenizerFor(text, max_length=512, return_tensors='pt')
        with torch.no_grad():
            hypotheses = self.modelFor.generate(**inputs.to("cpu"), max_length=512, do_sample=True, num_beams=5,
                                                **kwargs)
        return self.tokenizerFor.decode(hypotheses[0], skip_special_tokens=True)

    def paragraph(self, text):
        sentences = text.split('. ')
        embeddings = []
        for i in sentences:
            embedding = self.embed_bert_cls(i, self.modelPhar, self.tokenizerPhar)
            embeddings.append(embedding)
        embeddings = np.array(embeddings, dtype=float)
        similarities = cosine_similarity(embeddings)
        activated_similarities = self.activate_similarities(similarities, p_size=5)
        minmimas = argrelextrema(activated_similarities, np.less,
                                 order=2)
        split_points = [each for each in minmimas[0]]
        text = ''
        for num, each in enumerate(sentences):
            # Проверьте, является ли предложение минимумом (точкой разделения)
            if num in split_points:
                # Если да, то добавьте точку в конец предложения и абзац перед ним.
                text += f'\n\n {each}. '
            else:
                # Если это обычное предложение, просто поставьте точку в конце и продолжайте добавлять предложения.
                text += f'{each}. '
        return text[:-2]

    def embed_bert_cls(self, text, model, tokenizer):
        t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**{k: v.to(model.device) for k, v in t.items()})
        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings[0].cpu().numpy()

    def rev_sigmoid(self, x: float) -> float:
        return (1 / (1 + math.exp(0.5 * x)))

    def activate_similarities(self, similarities: np.array, p_size=10) -> np.array:
        # Чтобы создать веса для сигмоидной функции, сначала нужно создать пространство. P_size определяет количество используемых предложений и размер вектора весов.
        x = np.linspace(-10, 10, p_size)
        # Затем необходимо применить функцию активации к созданному пространству
        y = np.vectorize(self.rev_sigmoid)
        # Поскольку мы применяем активацию только к количеству предложений p_size, мы должны добавить нули, чтобы пренебречь эффектом каждого дополнительного предложения, а для соответствия длине вектора мы умножим
        activation_weights = np.pad(y(x), (0, similarities.shape[0] - p_size))
        ### 1. Возьмите каждую диагональ справа от главной диагонали
        diagonals = [similarities.diagonal(each) for each in range(0, similarities.shape[0])]
        ### 2. Заполните каждую диагональ нулями в конце. Поскольку каждая диагональ имеет разную длину, мы должны проводить заполнение нулями в конце.
        diagonals = [np.pad(each, (0, similarities.shape[0] - len(each))) for each in diagonals]
        ### 3. Сложите эти диагонали в новую матрицу
        diagonals = np.stack(diagonals)
        ### 4. Примените веса активации к каждой строке. Умножьте сходства на активацию.
        diagonals = diagonals * activation_weights.reshape(-1, 1)
        ### 5. Рассчитайте взвешенную сумму активированных сходств
        activated_similarities = np.sum(diagonals, axis=0)
        return activated_similarities
