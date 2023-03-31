class Debug:
    def __init__(self, *args, **kwargs):
        pass

    def inference(self, *args, **kwargs):
        import random
        sample_outputs = [
            '我是杨开心。',
            '我两岁半了。',
            '我喜欢吃雪糕。',
        ]
        one_output = random.choice(sample_outputs)
        for i in range(len(one_output)):
            yield one_output[:i + 1]

    def predict_continue(self, *args, **kwargs):
        yield from self.inference(*args, **kwargs)
