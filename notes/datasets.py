
# Video-MME-v2:
def p(x) :
        return x["level"] == "1" and x["group_type"] == "relevance" and x["second_head"] == "Frame-Only"

# 312 questions over 78 videos | 112 has about 4 minutes [ids: 12-15]
# /export/data1/ramann/huggingface/hub/Video-MME-v2/xxx.mp4

dataset = ...
dataset = dataset.add_column("video_path", [ f"/export/data1/ramann/huggingface/hub/Video-MME-v2/{x}.mp4" for x in dataset["video_id"]])
dataset = dataset.map(lambda item: {
    'video_path': f"/export/data1/ramann/huggingface/hub/Video-MME-v2/{item["video_id"]}.mp4",
    'text': f"{item["question"]}\noptions:\n{item["options"]}"
}, remove_columns=["url", "group_type", "group_structure", "level", "second_head", "third_head"])

entry = dataset[12]


