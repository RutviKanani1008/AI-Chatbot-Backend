from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)
CORS(app)

model_name = "microsoft/DialoGPT-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.route("/chatbot", methods=["POST"])
def chatbot():
    input_text = request.json.get("input_text")
    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")

    # Generate a response with adjusted parameters
    chat_history_ids = model.generate(
        input_ids,
        max_length=200,  # Increase max_length to allow longer responses
        temperature=0.7,  # Adjust temperature to control randomness (lower is less random)
        top_p=0.9,  # Nucleus sampling (top_p) to control response diversity
        top_k=50,  # Top-k sampling to limit the number of highest probability tokens considered
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    print('response--------', response)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
