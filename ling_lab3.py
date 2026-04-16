from google import genai

API_KEY = "AIzaSyBRf9IGVRv8HIZzAFnV5GX7BeHrdZpb2Pk"

client = genai.Client(api_key=API_KEY)

print("\n🤖 Bot: Hello! I'm powered by Google Gemini 2.5 Flash.")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("🤖 Bot: Goodbye! Have a great day!")
        break
    
    if user_input:
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_input
            )
            print(f"Bot: {response.text}\n")
        except Exception as e:
            print(f"Bot: Error: {e}\n")