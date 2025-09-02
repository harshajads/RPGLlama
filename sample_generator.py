import json
import fileinput
class Sample:
    def __init__(self, prompt, tilemap):
        self.prompt = prompt
        self.tilemap = tilemap
        self.can_train = False
        self.file_path = "C:\\Users\\harsh\\PycharmProjects\\Finetune_Llama\\sample_instance.json"
        self.gpt_recorder = "C:\\Users\\harsh\\PycharmProjects\\Finetune_Llama\\GPT_progress.txt"

    def create_sample(self):
        prompt = self.prompt
        map = ''
        for i in self.tilemap:
            map += i+'\n'
        map = map.rstrip()

        sample = {"prompt": prompt, "map": map}
        return sample

    def save_sample(self):
        s = self.create_sample()
        with open(self.file_path, 'a') as f:
            f.write(json.dumps(s) + '\n')


    def can_train_model(self):
        try:
            with open(self.file_path, 'r') as f:
                n = sum(1 for line in f)
                print(f"No of samples in the file: {n}")
                if n >= 1: # By the time the main program calls FineTune(), it checks if the instance file is nonempty
                    self.can_train = True
                else:
                    self.can_train = False
            return self.can_train
        except FileNotFoundError:
            return False


    def increment_success(self):
        file_path = self.gpt_recorder
        """Increment the successes value by 1."""

        for line in fileinput.input(file_path, inplace=True):
            if line.startswith("successes:"):
                current_value = int(line.strip().split(":")[1])
                print(f"successes: {current_value + 1}")
            else:
                print(line, end='')

    def increment_failure(self):
        file_path = self.gpt_recorder
        """Increment the failures value by 1."""
        for line in fileinput.input(file_path, inplace=True):
            if line.startswith("failures:"):
                current_value = int(line.strip().split(":")[1])
                print(f"failures: {current_value + 1}")
            else:
                print(line, end='')


#Test Case
"""s = Sample("Hi", "loser")
s.increment_success()
s.increment_failure()
print(s.can_train_model())"""