from src.model.model import load_TeMoLLM_Retrieval

model_dir = "/data/jw/motion/TextMotionRetrieval/TMR_LLM/result/train/64_0.8_32_256_20240720_1356"
TeMoLLM = load_TeMoLLM_Retrieval(model_dir=model_dir)

# prompt = " A man steps forward and does a handstand"

prompt = "Someone is swimming"

return_outputs = TeMoLLM.Text_Motion_Retrieval(prompt=prompt, max_mot_per_ret=7)

print(return_outputs)


