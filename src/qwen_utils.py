

def to_chat_messages(video: dict | None = None, text: str = "Describe this video."):
    content = []
    if video is None:
        content.append({
            "type": "video",
            "video": "PLACEHOLDER",
        })
    else :
        content.append({
            "type": "video",
            "video": f"file://{video['path'].resolve()}",
            "fps": video["fps"],
            "max_pixels": video["max_pixels"],
        })

    content.append({
        "type": "text",
        "text": text,
    })

    return [ {
        "role": "user",
        "content": content
    } ]