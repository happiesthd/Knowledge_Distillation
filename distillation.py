import os
import json
import yaml
import torch
import shutil
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from trl import SFTTrainer
from huggingface_hub import login

class GenerativeDistillationTrainer(SFTTrainer):
    def __init__(self, *args, teacher_model=None, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature
        
        if self.teacher_model is not None:
            self.teacher_model.to(self.args.device)
            self.teacher_model.eval()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Compute student's hard loss using the SFTTrainer's default method
        student_loss_hard, student_outputs = super().compute_loss(model, inputs, return_outputs=True)
        student_logits = student_outputs.logits 
        
        # Get teacher's predictions
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        # Get labels and mask out prompt tokens (-100)
        labels = inputs.get("labels")
        mask = labels != -100
        
        # Filter logits where labels are not masked
        filtered_student_logits = student_logits[mask]
        filtered_teacher_logits = teacher_logits[mask]
        
        # Soften distributions and compute KL Divergence(relative entropy) loss 
        student_log_softmax = F.log_softmax(filtered_student_logits / self.temperature, dim=-1)
        teacher_softmax = F.softmax(filtered_teacher_logits / self.temperature, dim=-1)
        loss_soft = F.kl_div(student_log_softmax, teacher_softmax, reduction="batchmean") * (self.temperature ** 2) 
        
        # Combine the losses
        loss = self.alpha * loss_soft + (1.0 - self.alpha) * student_loss_hard
        
        return (loss, outputs) if return_outputs else loss


class LegalDistillationPipeline:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.tokenizer = None
        self.teacher_model = None
        self.student_model = None
        self.dataset = None

    def load_config(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def setup_models_and_tokenizer(self):
        # It's critical that both models use the same tokenizer
        print(f"Loading tokenizer from teacher model path: {self.config['teacher_model']['path']}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['teacher_model']['path'])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading Teacher Model...")
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            self.config['teacher_model']['path'],
            torch_dtype=torch.bfloat16,
            # device_map="auto",
            attn_implementation="eager",
        )

        print("Loading Student Model...")
        self.student_model = AutoModelForCausalLM.from_pretrained(
            self.config['student_model']['path'],
            # torch_dtype=torch.float16,
            # device_map="auto",
            attn_implementation="eager",
        )
        print("All models and tokenizer loaded successfully.")

    # --- THIS IS YOUR EXACT DATA PROCESSING METHOD ---
    def extract_prompts_from_folder(self, folder_path):
        data_entries = []
        system_prompt = "You are a helpful Legal AI assistant that gives legal requirements based on provided law text and list of topic names."
        
        print(f"Extracting data from folder: {folder_path}")
        for fname in os.listdir(folder_path):
            if not fname.endswith(".json"):
                continue
            
            try:
                with open(os.path.join(folder_path, fname), encoding="utf-8") as f:
                    content = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                print(f"Skipping malformed file: {fname}")
                continue
    
            base_text = system_prompt + content.get("law_text_current", "").strip()
            if not base_text: continue
            
            requirements = content.get("requirements", [])
            if not requirements: continue
            
            assistant_response_parts = []
            all_topics = set()
            
            for req in requirements:
                req_name = req.get("requirement_name", "").strip()
                req_text = req.get("text", "").strip()
                req_topics = req.get("topics", [])
                
                if isinstance(req_topics, list):
                    all_topics.update(req_topics)
                else:
                    all_topics.add(str(req_topics))

                if req_name and req_text:
                    assistant_response_parts.append(
                        f"requirement name: {req_name}\n"
                        f"requirement text: {req_text}"
                    )
            
            if not assistant_response_parts: continue
    
            assistant_response = "\n\n".join(assistant_response_parts)
            topics_text = ", ".join(sorted(all_topics))
            user_prompt = f"{base_text}\n\nTopics: {topics_text}"
            
            messages = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response}
            ]
            
            full_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            data_entries.append({"text": full_text})
            
        self.dataset = Dataset.from_list(data_entries)
        print("First formatted entry:\n", self.dataset[0]['text'])
        print(f"Total entries created: {len(self.dataset)}")

    def run_distillation(self):
        # Setup training arguments from config file
        training_config = self.config["training"]
        distill_config = self.config["distillation"]

        training_args = TrainingArguments(
            output_dir=training_config["output_dir"],
            num_train_epochs=float(training_config["num_train_epochs"]),
            per_device_train_batch_size=int(training_config["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(training_config["gradient_accumulation_steps"]),
            learning_rate=float(training_config["learning_rate"]),
            max_grad_norm=float(training_config["max_grad_norm"]),
            logging_steps=int(training_config["logging_steps"]),
            save_steps=int(training_config["save_steps"]),
            fp16=training_config["fp16"],
            bf16=training_config["bf16"],
            optim=training_config["optim"],
            # report_to="tensorboard",
        )

        trainer = GenerativeDistillationTrainer(
            model=self.student_model,
            teacher_model=self.teacher_model,
            args=training_args,
            train_dataset=self.dataset,
            # dataset_text_field=self.config["dataset"]["text_column"],
            # max_seq_length=int(training_config["max_seq_length"]),
            # tokenizer=self.tokenizer,
            alpha=float(distill_config["alpha"]),
            temperature=float(distill_config["temperature"]),
        )

        print("Starting distillation training...")
        trainer.train()
        print("Training complete.")

        print(f"Saving final distilled model to {training_config['output_dir']}")
        trainer.save_model(training_config['output_dir'])
        self.tokenizer.save_pretrained(training_config['output_dir'])
        
        config_dest_path = os.path.join(training_config['output_dir'], "config.yaml")
        shutil.copy("config.yaml", config_dest_path)
        print("Model, tokenizer, and config saved.")


if __name__ == "__main__":
    pipeline = LegalDistillationPipeline("config.yaml")
    pipeline.setup_models_and_tokenizer()
    pipeline.extract_prompts_from_folder(pipeline.config["dataset"]["folder_path"])
    pipeline.run_distillation()