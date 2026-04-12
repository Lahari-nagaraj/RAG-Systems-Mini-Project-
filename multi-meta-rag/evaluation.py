def keyword_score(answer, keywords):
    if not keywords:
        return 0
    answer = answer.lower()
    match = sum(1 for k in keywords if k.lower() in answer)
    print(f"DEBUG SCORE: matched {match}/{len(keywords)} keywords {keywords} in answer")
    return round(match / len(keywords), 2)