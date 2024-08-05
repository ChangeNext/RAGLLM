from src.model.model import load_TeMoLLM_Retrieval
import src.util.utils_model as utils_model

model_dir = "/data/jw/motion/TextMotionRetrieval/TMR_LLM/result/train/32_0.85_32_512_Qwen2-1.5B_True_20240729_0311"
logger = utils_model.get_logger(model_dir)
TeMoLLM = load_TeMoLLM_Retrieval(model_dir=model_dir,logger=logger)

# prompt = " A man steps forward and does a handstand"

prompt = "a person walks forward to the left, picks something up and walks back and then shakes what is in the hand."

return_outputs = TeMoLLM.Text_Motion_Retrieval(prompt=prompt, max_mot_per_ret=10)

print(return_outputs)


