import getpass
import json
import urllib.request
import urllib.error

BOT_TOKEN = getpass.getpass("Nhập Bot Token: ").strip()
CHAT_ID = input("Nhập CHAT_ID: ").strip()

url = (
    "https://bot-api.zaloplatforms.com/"
    f"bot{BOT_TOKEN}/sendMessage"
)

payload = json.dumps({
    "chat_id": CHAT_ID,
    "text": "🔥 CANH BAO: He thong camera AI da ket noi thanh cong!"
}).encode("utf-8")

request = urllib.request.Request(
    url,
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST"
)

try:
    with urllib.request.urlopen(request, timeout=20) as response:
        result = json.loads(response.read().decode("utf-8"))

    print("\nKết quả từ Zalo:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if result.get("ok"):
        print("\n✅ Gửi cảnh báo Zalo thành công!")
    else:
        print("\n❌ Zalo không gửi được tin nhắn.")

except urllib.error.HTTPError as error:
    print("Lỗi HTTP:", error.code)
    print(error.read().decode("utf-8", errors="ignore"))

except Exception as error:
    print("Lỗi:", error)