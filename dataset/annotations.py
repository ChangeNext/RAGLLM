import json

def filter_annotations(annotations):
    filtered_annotations = {}
    for key, val in annotations.items():
        annots = val.pop("annotations")
        if annots:
            # 첫 번째 어노테이션만 선택
            first_annot = annots[0]
            filtered_annotations[key] = {
                "start": first_annot["start"],
                "end": first_annot["end"]
            }

    return filtered_annotations

def rearrange_annotations(annotations):
    rearranged_annotations = {}
    for key, val in annotations.items():
        annots = val["annotations"]
        if annots:
            start = annots[0]["start"]  # 첫 번째 어노테이션의 시작 시간
            end = annots[0]["end"]      # 첫 번째 어노테이션의 종료 시간
            text_list = [annot["text"] for annot in annots]
            rearranged_annotations[key] = {
                "start": start,
                "end": end,
                "text": text_list
            }
    return rearranged_annotations

# 필터링된 어노테이션 생성
with open("/data/motion/BLMP/dataset/humanml3d/annotations_1.json", "rb") as ff:
    original_data = json.loads(ff.read())
filtered_data = rearrange_annotations(original_data)


# 결과를 JSON 파일로 저장
with open("/data/motion/BLMP/dataset/humanml3d/filtered_annotations.json", "w") as outfile:
    json.dump(filtered_data, outfile, indent=4)