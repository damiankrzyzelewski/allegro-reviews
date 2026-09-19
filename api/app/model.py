import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

HF_REPO_ID = (
    "Damkrzyz/herbert-polish-reviews-dk"
)


class SentimentClassifier:

  def __init__(self, repo_id: str = HF_REPO_ID):
    print(f"Downloading/loading artifacts from {repo_id}...")

    tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")
    self.tokenizer = Tokenizer.from_file(tokenizer_path)

    self.tokenizer.enable_truncation(max_length=256)
    self.tokenizer.enable_padding(length=256, pad_id=0, pad_token="<pad>")

    model_path = hf_hub_download(
        repo_id=repo_id, filename="model_quantized.onnx"
    )

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 2
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    self.session = ort.InferenceSession(
        model_path, sess_options, providers=["CPUExecutionProvider"]
    )

  def predict(self, text: str):
      encoding = self.tokenizer.encode(text)

      input_ids = np.array([encoding.ids], dtype=np.int64)
      attention_mask = np.array([encoding.attention_mask], dtype=np.int64)

      if hasattr(encoding, "type_ids") and encoding.type_ids:
        token_type_ids = np.array([encoding.type_ids], dtype=np.int64)
      else:
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

      ort_inputs = {
          "input_ids": input_ids,
          "attention_mask": attention_mask,
          "token_type_ids": token_type_ids,
      }

      outputs = self.session.run(None, ort_inputs)

      logits = outputs[0].squeeze(0)

      probs = 1.0 / (1.0 + np.exp(-logits))
      predicted_rating = int((probs > 0.5).sum() + 1)

      return predicted_rating, probs.round(4).tolist()


classifier = None


def get_classifier():
  global classifier
  if classifier is None:
    classifier = SentimentClassifier()
  return classifier