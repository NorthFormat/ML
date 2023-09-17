# Импортируем библиотеки
from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast


class MLpart:
    def __init__(self):
        MODEL_NAME = 'UrukHan/t5-russian-spell'
        self.MAX_INPUT = 2024
        self.tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    def do(self, text, g, p, f):
        # Токенизирование данных
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
        return answer
