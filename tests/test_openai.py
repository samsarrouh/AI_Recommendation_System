import openai

OPENAI_API_KEY = "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"  # Replace this with your actual API key

try:
    openai.api_key = OPENAI_API_KEY
    response = openai.models.list()
    print("✅ API Key is working! Models available:", [model.id for model in response.data])
except Exception as e:
    print("❌ API Key is NOT working. Error:", str(e))
