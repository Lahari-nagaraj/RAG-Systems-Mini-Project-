def keyword_score(answer, keywords):
    if not keywords:
        return 0

    answer = answer.lower()
    match = 0

    for k in keywords:
        if k.lower() in answer:
            match += 1

    return round(match / len(keywords), 2)