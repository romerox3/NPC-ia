from transformers import GPTJForCausalLM, AutoTokenizer
import torch
from accelerate import init_empty_weights

class GPTJService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        
        # Inicialización más eficiente en memoria
        with init_empty_weights():
            self.model = GPTJForCausalLM.from_pretrained(
                "EleutherAI/gpt-j-6B",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto"  # Esto distribuirá automáticamente el modelo en los dispositivos disponibles
            )

    async def generate_text(self, prompt: str, max_length: int = 100) -> str:
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.cuda.amp.autocast():  # Usar precisión mixta automática
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text