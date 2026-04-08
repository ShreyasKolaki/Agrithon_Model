import os
import time
from dotenv import load_dotenv
from google import genai

# 📥 Load env
load_dotenv()

# 🔑 Load API keys
api_keys = os.getenv("GEMINI_API_KEYS", "").split(",")

if not api_keys or api_keys == [""]:
    raise ValueError("❌ No Gemini API keys found in .env")

# 🔄 Key index
current_key_index = 0


def get_client():
    return genai.Client(api_key=api_keys[current_key_index])


def switch_key():
    global current_key_index
    current_key_index = (current_key_index + 1) % len(api_keys)
    print(f"⚠️ Switching to API key #{current_key_index + 1}")


def get_gemini_response(prompt: str) -> str:
    max_retries = len(api_keys)

    for _ in range(max_retries):
        try:
            client = get_client()

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            if response and response.text:
                print("✅ Success using gemini-2.5-flash")
                return response.text.strip()

            return "⚠️ Empty response from Gemini."

        except Exception as e:
            error_msg = str(e)

            print(f"❌ Error (key {current_key_index + 1}): {error_msg}")

            # 🔥 Rate limit
            if "429" in error_msg:
                print("⏳ Rate limit → waiting 8 seconds...")
                time.sleep(8)

            # 🔥 Server overload
            elif "503" in error_msg:
                print("⏳ Server busy → waiting 3 seconds...")
                time.sleep(3)

            # 🔥 Other errors
            else:
                time.sleep(2)

            switch_key()

    return "❌ All API keys exhausted. Try again later."