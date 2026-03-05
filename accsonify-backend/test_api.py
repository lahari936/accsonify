import httpx
import asyncio

async def test_api():
    base_url = "http://127.0.0.1:8000"
    audio_file = "test_audio.wav"

    print("Testing /detect-accent")
    try:
        with open(audio_file, "rb") as f:
            files = {"audio": (audio_file, f, "audio/wav")}
            response = httpx.post(f"{base_url}/detect-accent", files=files)
            print("Status:", response.status_code)
            print("Response:", response.json())
    except Exception as e:
        print("Failed:", e)

    print("\nTesting /transcribe")
    try:
        with open(audio_file, "rb") as f:
            files = {"audio": (audio_file, f, "audio/wav")}
            response = httpx.post(f"{base_url}/transcribe", files=files)
            print("Status:", response.status_code)
            print("Response:", response.json())
    except Exception as e:
        print("Failed:", e)

    print("\nTesting /convert-accent")
    try:
        with open(audio_file, "rb") as f:
            files = {"audio": (audio_file, f, "audio/wav")}
            data = {"target_accent": "indian"}
            response = httpx.post(f"{base_url}/convert-accent", files=files, data=data)
            print("Status:", response.status_code)
            print("Response:", response.json())
    except Exception as e:
        print("Failed:", e)

if __name__ == "__main__":
    asyncio.run(test_api())
