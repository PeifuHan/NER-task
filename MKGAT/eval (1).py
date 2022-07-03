def get_entities(sequence_tag):
    #     entity = {
    #         'begin': 0,
    #         'end': 0,
    #         'type': ''
    #     }
    entities = []
    ne_type = ''
    is_ne = False
    flag = 0
    for i, tag in enumerate(sequence_tag):
        j = 0
        if 'B-' in tag:
            if is_ne:
                flag = 0
                for j in range(begin, i):
                    if ne_type != sequence_tag[j].split("-")[1]:
                        entities.append({'begin': begin, 'end': j, 'type': ne_type})
                        is_ne = False
                        ne_type = ''
                        flag = 1
                        break
                if not flag:
                    entities.append({'begin': begin, 'end': i, 'type': ne_type})
                    is_ne = False
                    ne_type = ''
            begin = i + 1
            ne_type = tag.split('-')[1]
            is_ne = True
        elif 'O' == tag and is_ne:
            flag = 0
            for j in range(begin, i):
                if ne_type != sequence_tag[j].split("-")[1]:
                    entities.append({'begin': begin, 'end': j, 'type': ne_type})
                    is_ne = False
                    ne_type = ''
                    flag = 1
                    break
            if not flag:
                entities.append({'begin': begin, 'end': i, 'type': ne_type})
                is_ne = False
                ne_type = ''
    if is_ne:
        entities.append({'begin': begin, 'end': i + 1, 'type': ne_type})
    return entities


def get_actual_predict_labels(filepath):
    predict = []
    actual = []
    with open(filepath, "r", encoding="utf8") as fr:
        for line in fr:
            line = line.strip().split("\t")

            predict.append(line[0])
            actual.append(line[1])

    return predict, actual


def classify_entities_by_types(entities):
    per = []
    loc = []
    org = []
    misc = []

    for entity in entities:
        if entity["type"] == "PER":
            per.append(entity)
        if entity["type"] == "LOC":
            loc.append(entity)
        if entity["type"] == "LOC":
            org.append(entity)
        if entity["type"] == "MISC":
            misc.append(entity)
    return per, loc, org, misc

def classify_entities_by_types_NCBI(entities):
    begin = []
    inner = []
    outer = []

    for entity in entities:
        if entity[0] == "B":
            begin.append(entity)
        if entity[0] == "I":
            inner.append(entity)
        if entity[0] == "O":
            outer.append(entity)
    return begin, inner, outer


def compute_metrics_by_type(pre_entities, act_entites):


    print("比较预测与真实==================")
    crct = 0
    for i in range(len(pre_entities)):
        if pre_entities[i] == act_entites[i]:
            crct +=1
    print(crct)
    precision = crct/len(pre_entities)
    recall = crct/len(act_entites)
    print("=====precision:")
    print(precision)
    print("=====recall:")
    print(recall)
    print("=====f1:")
    print(2*precision*recall/(precision+recall))

    return precision, recall, 2*precision*recall/(precision+recall)




if __name__ == '__main__':
    # labels = ["PER", "LOC", "ORG", "MISC"]
    labels = ["B", "I", "O"]
    filepath = "./output_NCBI/4.txt"
    predict_labels, actual_labels = get_actual_predict_labels(filepath)
    predict_entities = get_entities(predict_labels)
    actual_entities = get_entities(actual_labels)

    pre_begin, pre_inner, pre_outer = classify_entities_by_types_NCBI(predict_entities)
    act_begin, act_inner, act_outer = classify_entities_by_types_NCBI(actual_entities)
    print("预测============")
    print(len(pre_begin))
    print(len(pre_inner))
    print(len(pre_outer))
    print("真实============")
    print(len(act_begin))
    print(len(act_inner))
    print(len(act_outer))

    per_scores = compute_metrics_by_type(pre_per, act_per)
    loc_scores = compute_metrics_by_type(pre_loc, act_loc)
    org_scores = compute_metrics_by_type(pre_org, act_org)
    misc_scores = compute_metrics_by_type(pre_misc, act_misc)
    #全部
    per_scores = compute_metrics_by_type(predict_entities, actual_entities)
    per_scores = compute_metrics_by_type(pre_begin, act_begin)
    loc_scores = compute_metrics_by_type(pre_inner, act_inner)
    org_scores = compute_metrics_by_type(pre_outer, act_outer)


    P = (per_scores[0] + loc_scores[0] + org_scores[0] + misc_scores[0])/4
    R = (per_scores[1] + loc_scores[1] + org_scores[1] + misc_scores[1])/4
    P = (per_scores[0] + loc_scores[0] + org_scores[0])/3
    R = (per_scores[1] + loc_scores[1] + org_scores[1])/3
    print(P)
    print(R)
    print("Macro-F1: {}".format(2*P*R/(P+R)))
    file = open('result_NCBI.txt','a')
    file.write("Precision: {} Recall: {} Macro-F1: {}".format(P,R,2*P*R/(P+R)))
    file.close()

