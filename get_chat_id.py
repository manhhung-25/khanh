import getpass
import json
import time
import urllib.request
import urllib.error


def tim_chat_id(data):
    if isinstance(data, dict):
        chat = data.get("chat")

        if isinstance(chat, dict) and chat.get("id") is not None:
            return str(chat["id"])

        for value in data.values():
            chat_id = tim_chat_id(value)
            if chat_id:
                return chat_id

    elif isinstance(data, list):
        for value in data:
            chat_id = tim_chat_id(value)
            if chat_id:
                return chat_id

    return None


BOT_TOKEN = getpass.getpass("Nhập Bot Token: ").strip()

url = (
    "https://bot-api.zaloplatforms.com/"
    f"bot{BOT_TOKEN}/getUpdates"
)

print("\nĐang chờ tin nhắn mới...")
print("Bây giờ hãy mở Zalo và gửi một tin nhắn MỚI cho bot.")
print("Nhấn Ctrl+C nếu muốn dừng.\n")

while True:
    payload = json.dumps({
        "timeout": 30
    }).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=40) as response:
            result = json.loads(response.read().decode("utf-8"))

        if not result.get("ok"):
            if result.get("error_code") == 408:
                print("Chưa có tin nhắn, tiếp tục chờ...")
                continue

            print("Zalo trả về lỗi:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            break

        chat_id = tim_chat_id(result)

        if chat_id:
            print("\n==========================")
            print("CHAT_ID =", chat_id)
            print("==========================")
            break

        print("Có dữ liệu nhưng chưa tìm thấy CHAT_ID.")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    except urllib.error.HTTPError as error:
        if error.code == 408:
            print("Chưa có tin nhắn, tiếp tục chờ...")
            continue

        print("Lỗi HTTP:", error.code)
        print(error.read().decode("utf-8", errors="ignore"))
        break

    except Exception as error:
        print("Lỗi kết nối:", error)
        print("Thử lại sau 3 giây...")
        time.sleep(3)